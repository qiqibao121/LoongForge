#!/usr/bin/env bash
set -euo pipefail
TRAIN_SCRIPT="${TRAIN_SCRIPT:-/workspace/LoongForge/examples/embodied/pi05/run_pi05_ddp_finetune.sh}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT required}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:?BASE_CHECKPOINT required}"
CODE_ROOT="${CODE_ROOT:?CODE_ROOT required}"
TARGET_STEP="${TARGET_STEP:-1000}"
run_one() {
  local name="$1" mode="$2" port="$3"
  local out="$RUN_ROOT/$name"
  mkdir -p "$out/checkpoints"
  [[ -e "$out/checkpoints/steps_300" ]] || ln -s "$BASE_CHECKPOINT" "$out/checkpoints/steps_300"
  if [[ "$mode" == "exact" ]]; then
    env -u FP8_GRANULARITY_MODE -u FP8_GRANULARITY_LOG \
      RUN_ROOT="$RUN_ROOT" OUTPUT_DIR="$out" TENSORBOARD_DIR="$out/tensorboard" \
      TRAIN_ITERS="$TARGET_STEP" GPUS_PER_NODE=8 PER_DEVICE_BATCH_SIZE=12 NUM_WORKERS=4 \
      SEED=42 SAVE_INTERVAL=0 MASTER_PORT="$port" \
      bash "$TRAIN_SCRIPT" --zero-optimizer --zero-parameters-as-bucket-view \
      --ddp-comm-hook allreduce_hook --resume --save-format dcp \
      --lr-decay-iters 1000 model.compile_model=false 2>&1 | tee "$out/train.log"
  else
    FP8_GRANULARITY_MODE="$mode" FP8_GRANULARITY_MIN_BYTES=65536 \
      FP8_GRANULARITY_BLOCK=128 FP8_GRANULARITY_LOG="$out/fp8_stats.jsonl" \
      PYTHONPATH="$CODE_ROOT:/workspace/LoongForge:/workspace/Loong-Megatron" \
      RUN_ROOT="$RUN_ROOT" OUTPUT_DIR="$out" TENSORBOARD_DIR="$out/tensorboard" \
      TRAIN_ITERS="$TARGET_STEP" GPUS_PER_NODE=8 PER_DEVICE_BATCH_SIZE=12 NUM_WORKERS=4 \
      SEED=42 SAVE_INTERVAL=0 MASTER_PORT="$port" \
      bash "$TRAIN_SCRIPT" --zero-optimizer --zero-parameters-as-bucket-view \
      --ddp-comm-hook fp8_granularity_hook --ddp-powersgd-matrix-approximation-rank 1 \
      --ddp-powersgd-start-iter 2 --ddp-powersgd-min-compression-rate 4.0 \
      --resume --save-format dcp --lr-decay-iters 1000 \
      model.compile_model=false 2>&1 | tee "$out/train.log"
  fi
}
run_one dense exact 29730
run_one gradient_row row 29731
run_one gradient_block block 29732
