# Cosmos3 FP8 granularity precision experiment

All five formal branches resume from the same step600 DCP checkpoint and run
to step1000 with seed 42 and otherwise identical Cosmos3 training arguments.
Steps 601 and 602 use exact all-reduce; quantization starts at step603.

The fixed variants apply one granularity to every eligible gradient tensor:
2-D and strictly larger than 64 KiB. Other gradients use exact all-reduce.
The candidates are per-row, per-column, and contiguous block-128 scaling.

The per-tensor route is calibrated without updating the formal checkpoint.
Steps 603–620 calculate relative reconstruction MSE for all three candidates
while communicating the exact gradient. Step621 finalizes and writes the
cross-rank mean scores. The chosen route is fixed for the formal step600–1000
branch. If best and runner-up differ by less than 5%, block is chosen.

The route JSON and CSV record parameter name, shape, bytes, all candidate
scores, runner-up margin, and selected granularity for every quantized tensor.
