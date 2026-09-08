#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT="${RUN_ROOT:?RUN_ROOT required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:?BASE_CHECKPOINT required}"
CODE_ROOT="${CODE_ROOT:?CODE_ROOT required}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-/workspace/LoongForge/examples/embodied/pi05/run_pi05_ddp_finetune.sh}"
mkdir -p "$OUTPUT_DIR/checkpoints"
[[ -e "$OUTPUT_DIR/checkpoints/steps_300" ]] || ln -s "$BASE_CHECKPOINT" "$OUTPUT_DIR/checkpoints/steps_300"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128 \
FP8_GRANULARITY_MODE=block FP8_GRANULARITY_MIN_BYTES=65536 \
FP8_GRANULARITY_BLOCK=128 FP8_GRANULARITY_LOG="$OUTPUT_DIR/fp8_stats.jsonl" \
PYTHONPATH="$CODE_ROOT:/workspace/LoongForge:/workspace/Loong-Megatron" \
RUN_ROOT="$RUN_ROOT" OUTPUT_DIR="$OUTPUT_DIR" TENSORBOARD_DIR="$OUTPUT_DIR/tensorboard" \
TRAIN_ITERS=1000 GPUS_PER_NODE=8 PER_DEVICE_BATCH_SIZE=12 NUM_WORKERS=4 \
SEED=42 SAVE_INTERVAL=0 MASTER_PORT=29733 \
bash "$TRAIN_SCRIPT" --zero-optimizer --zero-parameters-as-bucket-view \
--ddp-comm-hook fp8_granularity_hook --ddp-powersgd-matrix-approximation-rank 1 \
--ddp-powersgd-start-iter 2 --ddp-powersgd-min-compression-rate 4.0 \
--resume --save-format dcp --lr-decay-iters 1000 \
model.compile_model=false 2>&1 | tee "$OUTPUT_DIR/train.log"
