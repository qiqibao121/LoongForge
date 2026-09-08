#!/usr/bin/env python3
"""Eight-GPU smoke test for software row/column/block FP8-like payloads.

This is a communication/shape test only; it intentionally does not modify the
Pi0.5 training package or claim a training result.
"""
import json
import math
import os

import torch
import torch.distributed as dist


FP8_MAX = 448.0  # e4m3fn finite range used for the software surrogate
MIN_BYTES = 64 * 1024
BLOCK = 128


def _quantize(x: torch.Tensor, mode: str):
    flat = x.reshape(-1).float()
    if x.numel() * x.element_size() <= MIN_BYTES:
        return None, None, tuple(x.shape)
    if mode == "row":
        groups = x.reshape(x.shape[0], -1)
    elif mode == "column":
        groups = x.reshape(-1, x.shape[-1]).transpose(0, 1)
    elif mode == "block":
        padded_numel = ((flat.numel() + BLOCK - 1) // BLOCK) * BLOCK
        groups = torch.nn.functional.pad(flat, (0, padded_numel - flat.numel()))
        groups = groups.reshape(-1, BLOCK)
    else:
        raise ValueError(mode)
    scale = groups.abs().amax(dim=1).clamp_min(1e-12) / FP8_MAX
    q = (groups / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.contiguous().view(torch.uint8), scale, tuple(x.shape)


def _dequant(qbytes, scale, mode, shape):
    q = qbytes.view(torch.float8_e4m3fn).reshape(scale.numel(), -1).float()
    q = q * scale[:, None]
    if mode == "row":
        return q.reshape(shape)
    if mode == "column":
        return q.transpose(0, 1).reshape(shape)
    return q.reshape(-1)[: math.prod(shape)].reshape(shape)


def _exact_average(x):
    world = dist.get_world_size()
    gathered = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(gathered, x)
    return torch.stack(gathered).float().mean(0)


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")
    rank = dist.get_rank()
    world = dist.get_world_size()
    records = []
    tensors = [torch.randn(128, 256, device=device, dtype=torch.bfloat16),
               torch.randn(257, 129, device=device, dtype=torch.bfloat16),
               torch.randn(1024, device=device, dtype=torch.bfloat16)]
    for mode in ("exact", "row", "column", "block"):
        for index, tensor in enumerate(tensors):
            local = tensor + rank * 0.01
            exact = _exact_average(local)
            if mode == "exact" or local.numel() * local.element_size() <= MIN_BYTES or local.ndim < 2:
                got = exact
                selected = False
            else:
                q, scale, shape = _quantize(local, mode)
                q_all = [torch.empty_like(q) for _ in range(world)]
                s_all = [torch.empty_like(scale) for _ in range(world)]
                dist.all_gather(q_all, q)
                dist.all_gather(s_all, scale)
                got = torch.stack([_dequant(qi, si, mode, shape) for qi, si in zip(q_all, s_all)]).mean(0)
                selected = True
            err = (got - exact).abs()
            record = {"mode": mode, "tensor": index, "shape": list(tensor.shape),
                      "selected": selected, "max_abs_error": float(err.max()),
                      "relative_l2": float(err.norm() / exact.norm().clamp_min(1e-12)),
                      "finite": bool(torch.isfinite(got).all())}
            if rank == 0:
                print(json.dumps(record), flush=True)
            records.append(record)
    dist.barrier()
    if rank == 0:
        print(json.dumps({"event": "smoke_complete", "world_size": world,
                          "min_bytes": MIN_BYTES, "block": BLOCK}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
