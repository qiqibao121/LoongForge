# Pi0.5 FP8 gradient-granularity comparison

## Objective

Repeat the Cosmos3 granularity experiment on Pi0.5 with five strictly aligned
branches: dense, global Gradient-row, global Gradient-column, global
Gradient-block, and calibration-routed Per-tensor. The experiment is for
precision only; it must not be interpreted as a production FP8 bandwidth
benchmark because the software hook quantizes/dequantizes locally and then
uses exact all-reduce.

## Fixed inputs

- Model: `/raid0/fastwam/vla_artifacts/pi05/models/pi05_base`
- Dataset: `/raid0/fastwam/vla_artifacts/pi05/datasets/libero`
- Tokenizer: `/raid0/fastwam/vla_artifacts/pi05/tokenizers/paligemma-3b-pt-224`
- Common checkpoint: `/raid0/fastwam/experiments/pi05_fp8_granularity_long_a800_v1/gradient_row/checkpoints/steps_300`
- 8 A800 GPUs, seed 42, the existing Pi0.5 DDP finetune entrypoint, same
  per-device batch, worker count, dtype, optimizer, LR schedule, zero settings,
  and activation-checkpoint settings as the prior Pi0.5 long run.
- Target: step 1000; all formal branches start from the same step-300 state.

## Branches

| Branch | Communication hook | Quantized tensors |
|---|---|---|
| `dense` | `allreduce_hook` | none; exact all-reduce |
| `gradient_row` | `fp8_granularity_hook`, mode `row` | every eligible 2-D tensor |
| `gradient_column` | `fp8_granularity_hook`, mode `column` | every eligible 2-D tensor |
| `gradient_block` | `fp8_granularity_hook`, mode `block` | every eligible 2-D tensor |
| `per_tensor` | `fp8_per_tensor_hook` | fixed per-tensor route from calibration |

Eligibility is unchanged across the three global modes and the routed mode:
`ndim == 2` and `numel * element_size > 65536` bytes. All other gradients
remain exact. Rank is 1, minimum compression rate is 4.0, block size is 128,
and the first two resumed steps are exact warmup steps; the first potentially
compressed update is step 303.

## Calibration and formal run

1. Run a 20-step calibration from step 300 (steps 301--320) with exact
   all-reduce. For each eligible tensor, measure relative reconstruction MSE
   for row, column, and block-128 on the same local gradient.
2. All-reduce score sums and counts across ranks, choose the lowest mean score,
   and freeze one route map. If the runner-up gap is below 5%, choose block-128
   for stability. Record parameter name, shape, dtype, bytes, all three scores,
   chosen route, and route byte/count shares.
3. Run the five formal branches serially from the same step-300 checkpoint to
   step 1000. The calibration branch is not included in formal metrics.
4. Before the long run, perform a 10--30 step smoke of each branch and require
   matching collective order, no shape mismatch, no NaN/Inf, and successful
   metric emission.

## Metrics and acceptance

Align action loss by step 301--1000 against `dense` and report MAE, MAPE,
standard deviation of absolute and relative error, maximum absolute and
relative error with step, last-20-step mean, final loss, step time and
throughput. Also report NaN/skipped iterations, OOM/CUDA/NCCL/shape errors,
eligible tensor count, compressed tensor count, and byte coverage. For
`per_tensor`, include the complete route table and shapes.

The conclusion is limited to gradient communication granularity. No claim
about parameter all-gather is allowed unless the runtime exposes and logs an
actual parameter-communication hook.

## Safety and reproducibility

- Use only the existing A800 ControlMaster through `devbox-reuse-only`; never
  create a new SSH connection.
- Launch through an offline/nohup supervisor inside the Pod and run branches
  serially so GPUs are released between branches.
- Stage experiment-only hook and argument patches, run `py_compile` and
  `bash -n` before launch, and restore the shared code in an EXIT trap.
- Preserve the original remote experiment and logs. Any fix gets a new local
  version directory and must not overwrite successful results.
