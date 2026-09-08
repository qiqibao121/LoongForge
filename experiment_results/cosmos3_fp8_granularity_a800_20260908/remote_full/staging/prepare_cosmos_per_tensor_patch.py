#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path


HOOK = r'''def fp8_per_tensor_hook(state, bucket):
    """Calibrate fixed per-tensor row/column/block routing by global step."""
    group = state.process_group if state.process_group is not None else dist.group.WORLD
    world_size = dist.get_world_size(group)
    input_tensor = bucket.buffer()
    bucket_index = bucket.index()
    current_step = state.current_global_step()
    calibration_end = int(os.getenv("FP8_PER_TENSOR_CALIBRATION_END_STEP", "620"))

    if state.per_tensor_mode == "calibrate" and current_step > calibration_end and not state._finalized:
        _fp8_per_tensor_finalize(state, input_tensor.device, group, world_size)

    parameters = list(bucket.parameters()) if hasattr(bucket, "parameters") else []
    offset = 0
    for tensor_index, tensor in enumerate(bucket.gradients()):
        parameter = parameters[tensor_index] if tensor_index < len(parameters) else None
        parameter_shape = getattr(parameter, "shape", None)
        if tensor.ndim != 2 and parameter_shape is not None and len(parameter_shape) == 2:
            view = input_tensor[offset:offset + tensor.numel()].view(tuple(parameter_shape))
        else:
            view = input_tensor[offset:offset + tensor.numel()].view_as(tensor)
        eligible = view.ndim == 2 and view.numel() * view.element_size() > state.min_bytes

        if current_step < state.quant_start_step:
            _fp8_per_tensor_exact(view, world_size, group)
        elif state.per_tensor_mode == "calibrate" and not state._finalized:
            if eligible and current_step <= calibration_end:
                scores = [_fp8_per_tensor_score(view, mode, state.block_size)
                          for mode in ("row", "column", "block")]
                key = state.parameter_names.get(
                    id(parameter), f"bucket:{bucket_index}:tensor:{tensor_index}"
                )
                state.add_scores(key, scores, {
                    "shape": list(view.shape),
                    "dtype": str(view.dtype),
                    "bytes": view.numel() * view.element_size(),
                })
            _fp8_per_tensor_exact(view, world_size, group)
        elif state.per_tensor_mode == "calibrate":
            # The route was finalized at the first bucket of step621; keep
            # the rest of the calibration step exact.
            _fp8_per_tensor_exact(view, world_size, group)
        elif not eligible:
            _fp8_per_tensor_exact(view, world_size, group)
        else:
            key = state.parameter_names.get(
                id(parameter), f"bucket:{bucket_index}:tensor:{tensor_index}"
            )
            if key not in state.route:
                raise KeyError(f"missing FP8 route for eligible tensor {key!r}")
            _fp8_granularity_reduce_view(
                view, state.route[key], state.block_size, world_size, group
            )
        offset += tensor.numel()

    future = torch.futures.Future()
    future.set_result(input_tensor)
    return future


'''


def patch(path: Path) -> None:
    source = path.read_text()
    start = source.find("def fp8_per_tensor_hook(")
    if start < 0:
        # The reusable Pi05 helper adds the per-tensor state and registration,
        # but intentionally leaves the hook body to the experiment-specific
        # launcher.  Insert our metrics-driven Cosmos hook before the common
        # build_comm_hook definition in that case.
        anchor = source.find("def build_comm_hook(")
        if anchor < 0:
            raise RuntimeError("build_comm_hook anchor not found")
        path.write_text(source[:anchor] + HOOK + source[anchor:])
        return
    # The reusable Pi05 helper places this hook next to the per-tensor state
    # classes near the top of the module, so replacing through build_comm_hook
    # would accidentally delete all later state classes and hooks.  Replace
    # only this function body.
    end = source.find("\n\nclass PowerSGDPlusErrorAveragingState", start)
    if end < 0:
        end = source.find("\n\ndef build_comm_hook(", start)
    if end < 0:
        raise RuntimeError("build_comm_hook anchor not found")
    path.write_text(source[:start] + HOOK + source[end + 2:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    patch(parser.parse_args().path)
