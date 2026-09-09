# Pi0.5 FP8 gradient granularity accuracy experiment

Experiment: `/raid0/fastwam/experiments/pi05_fp8_granularity_a800_600_1000_v1`

All five branches resumed from the same step-300 checkpoint and completed steps 301–1000 on 8×A800. No NaN, OOM, CUDA, or NCCL error was observed. The shared training files were restored and verified against their backups after completion.

## Accuracy summary

| Variant | Final action loss | Mean action loss | Last-20 mean | MAE vs dense | MAPE vs dense | Max abs error |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.13395448 | 0.15787909 | 0.14931263 | — | — | — |
| Gradient-row | 0.13385548 | 0.15787994 | 0.14934299 | 0.00024319 | 0.15535% | 0.00224522 |
| Gradient-column | 0.13379221 | 0.15785169 | 0.14932995 | 0.00029021 | 0.18379% | 0.00302747 |
| Gradient-block | 0.13435051 | 0.15789945 | 0.14935243 | 0.00027437 | 0.17441% | 0.00168952 |
| Per-tensor | 0.13421889 | 0.15787030 | 0.14927381 | **0.00024279** | **0.15476%** | 0.00189766 |

Per-tensor has the lowest trajectory MAE and MAPE among the quantized branches, narrowly ahead of Gradient-row. Gradient-column has the lowest final-step loss, but a larger full-trajectory error; one final point is therefore not sufficient to rank accuracy.

## Per-tensor routing

The calibrated route covers 460 unique 2D tensors and 9,348,825,088 bytes:

| Granularity | Tensors | Bytes | Byte share |
|---|---:|---:|---:|
| Block | 455 | 8,290,549,760 | 88.68% |
| Column | 4 | 1,058,144,256 | 11.32% |
| Row | 1 | 131,072 | <0.01% |

The global row/column/block reference hooks selected 461 gradient views and 9,351,534,592 bytes. They use the intended row, column, and 128-element block scaling math, but their selector accepts tensors with rank greater than one, whereas the per-tensor route is restricted to 2D tensors. This one-tensor, 2,709,504-byte coverage difference is recorded as an experimental limitation.

This is an accuracy experiment: quantize/dequantize is followed by ordinary all-reduce, so the measured step time and communication volume should not be interpreted as production FP8 communication performance.

## Artifacts

- `comparison_summary.json` and `comparison_summary.csv`: aligned step-301–1000 statistics.
- `loss_comparison.png`: loss, absolute error, and relative error plots.
- `remote_full/`: downloaded metrics, logs, routing records, scripts, and status files.
- `analyze_results.py`: reproducible summary and plotting script.
