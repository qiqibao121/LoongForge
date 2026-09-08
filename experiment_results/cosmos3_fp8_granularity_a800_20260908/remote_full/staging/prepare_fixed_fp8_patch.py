#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


STATE_ANCHOR = "        self.logged = set()\n"
STATE_REPLACEMENT = '''        self.logged = set()
        self.metrics_path = os.getenv("FP8_METRICS_PATH", "")
        self.global_step_offset = int(os.getenv("FP8_GLOBAL_STEP_OFFSET", "0"))
        self.quant_start_step = int(os.getenv("FP8_QUANT_START_STEP", "1"))
        self._metrics_mtime_ns = -1
        self._last_metrics_step = self.global_step_offset

    def current_global_step(self):
        if self.metrics_path:
            try:
                stat = os.stat(self.metrics_path)
                if stat.st_mtime_ns != self._metrics_mtime_ns:
                    with open(self.metrics_path, "rb") as stream:
                        lines = stream.readlines()[-64:]
                    for raw in reversed(lines):
                        try:
                            record = json.loads(raw.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        if "step" in record:
                            self._last_metrics_step = int(record["step"])
                            break
                    self._metrics_mtime_ns = stat.st_mtime_ns
            except OSError:
                pass
        return max(self.global_step_offset, self._last_metrics_step) + 1
'''

HOOK_START = "def fp8_granularity_hook(\n"
HOOK_END = '_SUPPORTED_COMM_HOOKS["dynamic_powersgd_hook"] = dynamic_powerSGD_hook\n'
HOOK = '''def fp8_granularity_hook(
    state: FP8GranularityState,
    bucket: dist.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Precision-only row/column/block gradient quantization reference."""
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    input_tensor = bucket.buffer()
    bucket_index = bucket.index()
    global_step = state.current_global_step()
    if global_step < state.quant_start_step:
        state.maybe_increase_iter(bucket)
        return dist.all_reduce(input_tensor, group=group, async_op=True).get_future().then(
            lambda fut: fut.value()[0].div_(world_size)
        )

    parameters = list(bucket.parameters()) if hasattr(bucket, "parameters") else []
    offset = 0
    selected = 0
    selected_bytes = 0
    for tensor_index, tensor in enumerate(bucket.gradients()):
        parameter = parameters[tensor_index] if tensor_index < len(parameters) else None
        parameter_shape = getattr(parameter, "shape", None)
        if tensor.ndim != 2 and parameter_shape is not None and len(parameter_shape) == 2:
            view = input_tensor[offset:offset + tensor.numel()].view(tuple(parameter_shape))
        else:
            view = input_tensor[offset:offset + tensor.numel()].view_as(tensor)
        use_quant = view.ndim == 2 and view.numel() * view.element_size() > state.min_bytes
        if use_quant:
            _fp8_granularity_reduce_view(
                view, state.granularity, state.block_size, world_size, group
            )
            selected += 1
            selected_bytes += view.numel() * view.element_size()
        else:
            reduced = view.detach().clone()
            dist.all_reduce(reduced, group=group)
            view.copy_(reduced.div_(world_size).to(view.dtype))
        offset += tensor.numel()
    if dist.get_rank(group) == 0 and state.log_path and bucket_index not in state.logged:
        state.logged.add(bucket_index)
        try:
            with open(state.log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps({"global_step": global_step, "bucket": bucket_index,
                    "granularity": state.granularity, "selected_tensors": selected,
                    "selected_bytes": selected_bytes}) + "\\n")
        except OSError:
            logger.warning("Unable to write FP8 granularity stats", exc_info=True)
    state.maybe_increase_iter(bucket)
    future = torch.futures.Future()
    future.set_result(input_tensor)
    return future


'''


def patch(path: Path) -> None:
    source = path.read_text()
    if "def current_global_step(self):" not in source:
        if STATE_ANCHOR not in source:
            raise RuntimeError("FP8 state anchor not found")
        source = source.replace(STATE_ANCHOR, STATE_REPLACEMENT, 1)
    start = source.find(HOOK_START)
    end = source.find(HOOK_END, start)
    if start < 0 or end < 0:
        raise RuntimeError("FP8 hook anchors not found")
    source = source[:start] + HOOK + source[end:]
    path.write_text(source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    patch(parser.parse_args().path)
