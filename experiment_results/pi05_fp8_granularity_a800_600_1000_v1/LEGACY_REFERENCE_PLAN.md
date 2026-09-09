# Pi0.5 per-tensor FP8 granularity experiment

## Objective

The completed global row/block experiment answers only which single granularity is best on average. This follow-up chooses a granularity independently for every eligible gradient tensor, while preserving the same step-300 checkpoint, seed, optimizer, threshold and exact fallback.

## Eligibility and modes

- Eligible: two-dimensional tensors with `numel * element_size > 65536` bytes.
- Ineligible tensors: exact all-reduce.
- Candidate modes for each eligible tensor: row, column, block-128.
- The route uses the parameter name as the stable tensor key. Bucket/tensor position is only a fail-safe key, and the formal run stops if an eligible key is absent.

## Two-stage protocol

1. **Calibration (20 optimizer steps, 301–320)**
   - Start from the common step-300 checkpoint.
   - Use exact all-reduce for training so the calibration trajectory is not affected by a candidate choice.
   - For each eligible tensor, compute local row/column/block dequantization distortion on the same gradient. Accumulate a mean score and all-reduce score sums and sample counts across ranks before choosing.
   - Write one immutable route map and per-tensor scores. No route changes during the formal run.

2. **Formal comparison (301–1000, from step300)**
   - Run per-tensor routed row/column/block using the frozen calibration route map.
   - Reuse the already completed dense, global-row and global-block-128 branches as references. The launcher can reproduce them only when `RUN_REFERENCES=1` is explicitly set.

## Scores and safeguards

The calibration score is relative local quantization distortion,

`mean((dequant(quant(g)) - g)^2) / max(mean(g^2), eps)`.

The calibration log also records the winning mode, runner-up gap, tensor shape, dtype and bytes. If scores are tied or the winning margin is below 5%, the route defaults to block-128 for stability. This is a routing pilot, not an oracle using future loss or dense results.

## Fairness and limitations

- The new branch and the three completed references use the same step-300 checkpoint, seed, data, LR schedule, two exact initial steps, eligibility threshold and target step.
- The hook is gradient-only because the current DDP path exposes no parameter all-gather/broadcast hook.
- This is a precision reference implementation: local quantize/dequantize is followed by exact all-reduce, so timings are not production FP8 wire-bandwidth measurements.
- The reference hook does not retain a true pre-quantization residual, so this experiment isolates granularity under the same quantization-only semantics as the completed row/block runs; it does not claim error-feedback behavior.
- The calibration run is separate and is not included in the 301–1000 formal comparison.
