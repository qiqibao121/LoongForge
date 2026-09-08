#!/usr/bin/env python3
"""Idempotently add calibration-driven per-tensor routing to the FP8 hook."""
from __future__ import annotations

import argparse
from pathlib import Path
import textwrap


INSERT = textwrap.dedent(r'''
class PerTensorGranularityState(FP8GranularityState):
    """Choose row/column/block once per tensor after exact calibration."""

    def __init__(self, *args, route_path="", calibration_steps=20,
                 mode="route", parameter_names=None, **kwargs):
        kwargs.setdefault("granularity", "block")
        super().__init__(*args, **kwargs)
        if mode not in {"calibrate", "route", "per_tensor"}:
            raise ValueError("invalid FP8_PER_TENSOR_MODE")
        if int(calibration_steps) <= 0:
            raise ValueError("FP8_PER_TENSOR_CALIBRATION_STEPS must be positive")
        self.route_path = route_path or os.getenv("FP8_PER_TENSOR_ROUTE_PATH", "")
        self.per_tensor_mode = mode
        self.calibration_steps = int(calibration_steps)
        self.parameter_names = parameter_names or {}
        self.route = {}
        self.scores = {}
        self.score_counts = {}
        self.metadata = {}
        self._finalized = False
        if mode != "calibrate":
            if not self.route_path:
                raise ValueError("FP8_PER_TENSOR_ROUTE_PATH is required in route mode")
            try:
                with open(self.route_path, encoding="utf-8") as stream:
                    payload = json.load(stream)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"unable to load FP8 per-tensor route {self.route_path!r}"
                ) from exc
            self.route = dict(payload.get("route", payload))
            if not self.route:
                raise ValueError("FP8 per-tensor route is empty")

    def add_scores(self, key, values, metadata):
        target = self.scores.setdefault(key, [0.0, 0.0, 0.0])
        for index, value in enumerate(values):
            target[index] += float(value)
        self.score_counts[key] = self.score_counts.get(key, 0) + 1
        self.metadata.setdefault(key, metadata)

    def maybe_increase_iter(self, bucket):
        # Some DDP builds do not propagate GradBucket.is_last() reliably;
        # bucket index 0 is the documented final bucket in that case.
        is_last = bucket.is_last()
        index_zero = hasattr(bucket, "index") and bucket.index() == 0
        if is_last or index_zero:
            self.iter += 1


def _fp8_per_tensor_score(view, mode, block_size):
    """Relative MSE using bounded workspaces and exact granularity boundaries."""
    error = torch.zeros((), dtype=torch.float64, device=view.device)
    energy = torch.zeros((), dtype=torch.float64, device=view.device)
    max_chunk_elems = 1 << 20
    if mode == "row":
        rows, cols = view.shape
        chunk_rows = max(1, min(rows, max_chunk_elems // max(cols, 1)))
        parts = (view[start:min(start + chunk_rows, rows)]
                 for start in range(0, rows, chunk_rows))
    elif mode == "column":
        rows, cols = view.shape
        chunk_cols = max(1, min(cols, max_chunk_elems // max(rows, 1)))
        parts = (view[:, start:min(start + chunk_cols, cols)]
                 for start in range(0, cols, chunk_cols))
    elif mode == "block":
        flat = view.reshape(-1)
        parts = (flat[start:min(start + max_chunk_elems, flat.numel())]
                 for start in range(0, flat.numel(), max_chunk_elems))
    else:
        raise ValueError(mode)
    for part in parts:
        q, scale, shape = _fp8_granularity_quantize(part, mode, block_size)
        dequantized = _fp8_granularity_dequantize(
            q, scale, mode, shape, block_size
        ).float()
        source = part.float()
        error.add_((dequantized - source).square().sum(dtype=torch.float64))
        energy.add_(source.square().sum(dtype=torch.float64))
    return float((error / energy.clamp_min(1e-12)).item())


def _fp8_per_tensor_finalize(state, device, group, world_size):
    if state._finalized or state.per_tensor_mode != "calibrate":
        return
    keys = sorted(state.scores)
    route = {}
    score_records = {}
    if keys:
        values = torch.tensor([state.scores[key] for key in keys],
                              dtype=torch.float64, device=device)
        counts = torch.tensor([state.score_counts[key] for key in keys],
                              dtype=torch.float64, device=device)
        dist.all_reduce(values, group=group)
        dist.all_reduce(counts, group=group)
        values.div_(counts[:, None].clamp_min(1.0))
        for key, row in zip(keys, values.tolist()):
            order = sorted(range(3), key=lambda index: row[index])
            best, second = order[0], order[1]
            chosen = ("row", "column", "block")[best]
            gap = (row[second] - row[best]) / max(row[best], 1e-12)
            if row[best] <= 0 or gap < 0.05:
                chosen = "block"
            route[key] = chosen
            score_records[key] = {
                "row": row[0], "column": row[1], "block": row[2],
                "chosen": chosen, "runner_up_gap": gap,
                **state.metadata[key],
            }
        state.route = route
        if dist.get_rank(group) == 0 and state.route_path:
            parent = os.path.dirname(state.route_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(state.route_path, "w", encoding="utf-8") as stream:
                json.dump({
                    "version": 1,
                    "calibration_steps": state.calibration_steps,
                    "route": route,
                    "scores": score_records,
                }, stream, indent=2, sort_keys=True)
    state._finalized = True


def _fp8_per_tensor_exact(view, world_size, group):
    reduced = view.detach().clone()
    dist.all_reduce(reduced, group=group)
    view.copy_(reduced.div_(world_size).to(view.dtype))


def fp8_per_tensor_hook(state, bucket):
    """Calibrate on exact training, then apply a fixed per-parameter route."""
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    input_tensor = bucket.buffer()

    # Match the two exact warm-up iterations used by the global row/block runs.
    # DDP may rebuild bucket layouts during these iterations; no residual must
    # be allocated until the layout is stable.
    if state.iter < state.start_powerSGD_iter:
        state.maybe_increase_iter(bucket)
        return dist.all_reduce(input_tensor, group=group, async_op=True).get_future().then(
            lambda future: future.value()[0].div_(world_size)
        )

    bucket_index = bucket.index()
    # This reference experiment intentionally has no model-sized residual:
    # quantization-only in-place updates avoid an extra 3–7 GB allocation and
    # match the effective semantics of the completed row/block reference runs.
    corrected = input_tensor
    parameters = list(bucket.parameters()) if hasattr(bucket, "parameters") else []
    offset = 0
    for tensor_index, tensor in enumerate(bucket.gradients()):
        parameter = parameters[tensor_index] if tensor_index < len(parameters) else None
        # With gradient_as_bucket_view, DDP may expose a flattened 1-D view
        # even when the underlying parameter is a matrix. Recover the
        # parameter shape before applying the per-tensor eligibility rule.
        tensor_shape = getattr(parameter, "shape", None)
        if tensor.ndim != 2 and tensor_shape is not None and len(tensor_shape) == 2:
            view = corrected[offset:offset + tensor.numel()].view(tuple(tensor_shape))
        else:
            view = corrected[offset:offset + tensor.numel()].view_as(tensor)
        fallback_key = f"bucket:{bucket_index}:tensor:{tensor_index}"
        if tensor_index < len(parameters):
            key = state.parameter_names.get(id(parameters[tensor_index]), fallback_key)
        else:
            key = fallback_key
        eligible = view.ndim == 2 and view.numel() * view.element_size() > state.min_bytes
        if not eligible:
            _fp8_per_tensor_exact(view, world_size, group)
        elif state.per_tensor_mode == "calibrate":
            scores = [_fp8_per_tensor_score(view, mode, state.block_size)
                      for mode in ("row", "column", "block")]
            state.add_scores(key, scores, {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "bytes": tensor.numel() * tensor.element_size(),
            })
            _fp8_per_tensor_exact(view, world_size, group)
        else:
            if key not in state.route:
                raise KeyError(f"missing FP8 route for eligible tensor {key!r}")
            _fp8_granularity_reduce_view(
                view, state.route[key], state.block_size, world_size, group
            )
        offset += tensor.numel()
    previous_iter = state.iter
    state.maybe_increase_iter(bucket)
    if (state.per_tensor_mode == "calibrate" and state.iter != previous_iter
            and state.iter >= state.calibration_steps):
        _fp8_per_tensor_finalize(state, input_tensor.device, group, world_size)
    future = torch.futures.Future()
    future.set_result(input_tensor)
    return future
''')


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    if old not in source:
        raise RuntimeError(f"{label} anchor not found")
    return source.replace(old, new, 1)


def patch(path: Path) -> None:
    source = path.read_text()
    if "class PerTensorGranularityState" not in source:
        source = replace_once(
            source, "class PowerSGDPlusErrorAveragingState",
            INSERT + "\n\nclass PowerSGDPlusErrorAveragingState",
            "class insertion",
        )
    source = replace_once(
        source,
        '_SUPPORTED_COMM_HOOKS["fp8_granularity_hook"] = fp8_granularity_hook',
        '_SUPPORTED_COMM_HOOKS["fp8_granularity_hook"] = fp8_granularity_hook\n'
        '_SUPPORTED_COMM_HOOKS["fp8_per_tensor_hook"] = fp8_per_tensor_hook',
        "hook registration",
    )
    source = replace_once(
        source,
        '    is_fp8_granularity = hook == "fp8_granularity_hook" or hook is fp8_granularity_hook',
        '    is_fp8_granularity = hook == "fp8_granularity_hook" or hook is fp8_granularity_hook\n'
        '    is_fp8_per_tensor = hook == "fp8_per_tensor_hook" or hook is fp8_per_tensor_hook',
        "hook predicate",
    )
    source = replace_once(
        source,
        '        or is_fp8_granularity\n    ):',
        '        or is_fp8_granularity\n        or is_fp8_per_tensor\n    ):',
        "state condition",
    )
    source = replace_once(
        source,
        '        if is_fp8_granularity:\n            state_cls = FP8GranularityState',
        '        if is_fp8_per_tensor:\n            state_cls = PerTensorGranularityState\n'
        '        elif is_fp8_granularity:\n            state_cls = FP8GranularityState',
        "state class",
    )
    old = '        if is_fp8_granularity:\n            state_kwargs.update(\n                granularity=os.getenv("FP8_GRANULARITY_MODE", "block"),'
    new = '        if is_fp8_per_tensor:\n            state_kwargs.update(\n                mode=os.getenv("FP8_PER_TENSOR_MODE", "route"),\n                route_path=os.getenv("FP8_PER_TENSOR_ROUTE_PATH", ""),\n                calibration_steps=int(os.getenv("FP8_PER_TENSOR_CALIBRATION_STEPS", "20")),\n                parameter_names=parameter_names,\n                min_bytes=int(os.getenv("FP8_GRANULARITY_MIN_BYTES", "65536")),\n                block_size=int(os.getenv("FP8_GRANULARITY_BLOCK", "128")),\n                log_path=os.getenv("FP8_PER_TENSOR_LOG", ""),\n            )\n        elif is_fp8_granularity:\n            state_kwargs.update(\n                granularity=os.getenv("FP8_GRANULARITY_MODE", "block"),'
    source = replace_once(source, old, new, "FP8 state kwargs")
    path.write_text(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    patch(parser.parse_args().path)
