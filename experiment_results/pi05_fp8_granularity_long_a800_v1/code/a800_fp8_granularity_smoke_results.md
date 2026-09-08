# Pi0.5 FP8 granularity short smoke (A800)

- Reference: D0 dense, steps 301–310, from pi05_fp8_granularity_matrix_smoke_a800_v2.
- Runs: v6, gradient-only software reference hook; quantization starts at iteration 2.
- Threshold: only tensors with ndim > 1 and numel * element_size > 65536 bytes are quantized; all other gradients use exact all-reduce.
- Shared training_args.py and ddp_comm_hook.py were restored after the run.

| mode | final action loss | MAE vs dense | MAPE vs dense | max abs error | mean step time (302–310 s) | mean samples/s |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.149645746 | 0 | 0 | 0 | 1.018 | 12.09 |
| gradient_row | 0.149562880 | 2.51338e-05 | 0.01507% | 8.28654e-05 | 1.3626 | 9.2571 |
| gradient_column | 0.149627075 | 2.44305e-05 | 0.01426% | 1.51739e-04 | 1.3489 | 9.1183 |
| gradient_block (128) | 0.149683163 | 2.82556e-05 | 0.01692% | 9.21935e-05 | 1.6488 | 7.7758 |

Compression-selection logging recorded 461 selected tensor instances over 144 buckets and 9,351,534,592 selected bytes. Based on the run's mixed trainable dtypes (9,353,911,360 total gradient bytes), selected byte coverage is approximately 99.975% (the remainder is the exact small/1-D fallback).

All three modes reached step 310 with finite losses, zero NaN/skipped iterations, and no OOM/CUDA/NCCL/shape errors in the final v6 run. v4 is retained as the original OOM diagnostic (whole-rank dequantized stack, +15.7 GiB); v5 is retained as the non-contiguous column-view diagnostic.

This smoke validates gradient-side granularity behavior only. The current Pi0.5 DDP path has no parameter all-gather/broadcast communication hook, so a full gradient-by-parameter 3x3 matrix is not claimed. The hook uses chunked dequantize + exact all-reduce as a precision reference; its timing is not a production FP8 bandwidth measurement.

## Stage B: 200-step candidate run (steps 301–500)

The same four branches were run serially from the same step-300 checkpoint. The first three steps remained exact, and quantization began after the configured start iteration.

| mode | final action loss | MAE vs dense | MAPE vs dense | max abs error | last-20 mean | mean step time (302–500 s) |
|---|---:|---:|---:|---:|---:|---:|
| dense | 0.171968520 | 0 | 0 | 0 | 0.160799 | 0.998 |
| gradient_row | 0.171717703 | 1.89732e-04 | 0.11584% | 7.46131e-04 | 0.160818 | 1.343 |
| gradient_column | 0.171422482 | 2.90266e-04 | 0.17700% | 1.90972e-03 | 0.160928 | 1.452 |
| gradient_block (128) | 0.171518788 | 2.46249e-04 | 0.14989% | 1.52788e-03 | 0.160860 | 1.792 |

All four branches reached step 500 with finite losses, zero skipped/nan iterations, and no CUDA/NCCL/shape failures. The 200-step run is a candidate-screening stage; no 1000-step result has been claimed yet.

## Stage D: 1000-step formal precision run (steps 301–1000)

The formal run used the same step-300 checkpoint, seed, optimizer configuration, and threshold as Stage B. Column-wise quantization was screened out after Stage B; the formal comparison therefore ran dense, gradient-row, and gradient-block. The first gradient-block attempt hit CUDA OOM before producing a valid metric stream. Its log was retained, and a retry used only the allocator setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128`; the quantization semantics and training configuration were unchanged. The retry completed step 1000.

| mode | final action loss | MAE vs dense | MAPE vs dense | max abs error | max-error step | last-20 mean | mean step time (302–1000 s) | mean samples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.133954480 | 0 | 0 | 0 | — | 0.149313 | 0.989 | 12.141 |
| gradient_row | 0.133855477 | 2.43191e-04 | 0.15535% | 2.24522e-03 | 582 | 0.149343 | 1.347 | 8.920 |
| gradient_block (128) | 0.134350508 | 2.74373e-04 | 0.17441% | 1.68952e-03 | 838 | 0.149352 | 4.044 | 3.046 |

All valid formal branches reached step 1000 with zero NaN and skipped iterations. Relative to dense, row-wise quantization had the lower average error (about 11.4% lower MAE than block-wise), while block-wise had the lower single worst deviation. The last-20-step means were nearly identical to dense (absolute differences about 3.0e-5 for row and 4.0e-5 for block). The reported timing is for the chunked reference implementation (local quantize/dequantize followed by exact all-reduce), not production FP8 wire bandwidth.

Both formal quantized branches selected the same 461 tensor instances across 144 buckets, totaling 9,351,534,592 bytes (about 99.975% of the measured 9,353,911,360 gradient bytes); the remainder used exact fallback. Shared `training_args.py` and `ddp_comm_hook.py` were restored byte-for-byte to their backups after completion. The original block OOM log and allocator-optimized retry log remain in the remote experiment directory.
