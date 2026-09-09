#!/usr/bin/env bash
set -euo pipefail

TRAIN_SCRIPT="${TRAIN_SCRIPT:-/workspace/LoongForge/examples/embodied/pi05/run_pi05_ddp_finetune.sh}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT required}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:?BASE_CHECKPOINT required}"
CODE_ROOT="${CODE_ROOT:?CODE_ROOT required}"
DATASET_PATH="${DATASET_PATH:-/raid0/fastwam/vla_artifacts/pi05/datasets/libero}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/raid0/fastwam/vla_artifacts/pi05/tokenizers/paligemma-3b-pt-224}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/raid0/fastwam/vla_artifacts/pi05/models/pi05_base}"
ROUTE_PATH="${ROUTE_PATH:?ROUTE_PATH required}"
TARGET_STEP="${TARGET_STEP:-1000}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-12}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
LR_DECAY_ITERS="${LR_DECAY_ITERS:-1000}"
RESULTS="$RUN_ROOT/variant_results.tsv"

if [[ ! -f "$RESULTS" ]]; then
  printf 'variant\tresult\tattempt\tlast_step\tdetail\n' > "$RESULTS"
fi

run_one() {
  local name="$1" hook="$2" mode="$3" port="$4"
  local out="$RUN_ROOT/$name"
  mkdir -p "$out/checkpoints"
  [[ -e "$out/checkpoints/steps_300" ]] || ln -s "$BASE_CHECKPOINT" "$out/checkpoints/steps_300"
  if [[ -f "$out/metrics.jsonl" ]] && tail -n 1 "$out/metrics.jsonl" | grep -q '"step": *'"$TARGET_STEP"; then
    echo "[$(date '+%F %T')] skip completed $name"
    printf '%s\tsuccess\t1\t%s\tcompleted\n' "$name" "$TARGET_STEP" >> "$RESULTS"
    return 0
  fi
  echo "[$(date '+%F %T')] starting $name"
  if [[ "$hook" == "allreduce_hook" ]]; then
    env -u FP8_GRANULARITY_MODE -u FP8_GRANULARITY_LOG \
      -u FP8_PER_TENSOR_MODE -u FP8_PER_TENSOR_ROUTE_PATH \
      PYTHONPATH="$CODE_ROOT:/workspace/LoongForge:/workspace/Loong-Megatron" \
      RUN_ROOT="$RUN_ROOT" OUTPUT_DIR="$out" TENSORBOARD_DIR="$out/tensorboard" \
      TRAIN_ITERS="$TARGET_STEP" GPUS_PER_NODE="$GPUS_PER_NODE" \
      PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" \
      SEED="$SEED" SAVE_INTERVAL=0 MASTER_PORT="$port" \
      bash "$TRAIN_SCRIPT" --dataset-path "$DATASET_PATH" \
      --tokenizer-path "$TOKENIZER_PATH" --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
      --zero-optimizer --zero-parameters-as-bucket-view \
      --ddp-comm-hook allreduce_hook --resume --save-format dcp \
      --lr-decay-iters "$LR_DECAY_ITERS" model.compile_model=false 2>&1 | tee "$out/train.log"
  elif [[ "$hook" == "fp8_per_tensor_hook" ]]; then
    FP8_PER_TENSOR_MODE=route FP8_PER_TENSOR_ROUTE_PATH="$ROUTE_PATH" \
      FP8_GRANULARITY_MIN_BYTES=65536 FP8_GRANULARITY_BLOCK=128 \
      FP8_PER_TENSOR_LOG="$out/per_tensor_stats.jsonl" \
      PYTHONPATH="$CODE_ROOT:/workspace/LoongForge:/workspace/Loong-Megatron" \
      RUN_ROOT="$RUN_ROOT" OUTPUT_DIR="$out" TENSORBOARD_DIR="$out/tensorboard" \
      TRAIN_ITERS="$TARGET_STEP" GPUS_PER_NODE="$GPUS_PER_NODE" \
      PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" \
      SEED="$SEED" SAVE_INTERVAL=0 MASTER_PORT="$port" \
      bash "$TRAIN_SCRIPT" --dataset-path "$DATASET_PATH" \
      --tokenizer-path "$TOKENIZER_PATH" --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
      --zero-optimizer --zero-parameters-as-bucket-view \
      --ddp-comm-hook fp8_per_tensor_hook --ddp-powersgd-matrix-approximation-rank 1 \
      --ddp-powersgd-start-iter 2 --ddp-powersgd-min-compression-rate 4.0 \
      --resume --save-format dcp --lr-decay-iters "$LR_DECAY_ITERS" \
      model.compile_model=false 2>&1 | tee "$out/train.log"
  else
    FP8_GRANULARITY_MODE="$mode" FP8_GRANULARITY_MIN_BYTES=65536 \
      FP8_GRANULARITY_BLOCK=128 FP8_GRANULARITY_LOG="$out/fp8_stats.jsonl" \
      PYTHONPATH="$CODE_ROOT:/workspace/LoongForge:/workspace/Loong-Megatron" \
      RUN_ROOT="$RUN_ROOT" OUTPUT_DIR="$out" TENSORBOARD_DIR="$out/tensorboard" \
      TRAIN_ITERS="$TARGET_STEP" GPUS_PER_NODE="$GPUS_PER_NODE" \
      PER_DEVICE_BATCH_SIZE="$PER_DEVICE_BATCH_SIZE" NUM_WORKERS="$NUM_WORKERS" \
      SEED="$SEED" SAVE_INTERVAL=0 MASTER_PORT="$port" \
      bash "$TRAIN_SCRIPT" --dataset-path "$DATASET_PATH" \
      --tokenizer-path "$TOKENIZER_PATH" --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
      --zero-optimizer --zero-parameters-as-bucket-view \
      --ddp-comm-hook fp8_granularity_hook --ddp-powersgd-matrix-approximation-rank 1 \
      --ddp-powersgd-start-iter 2 --ddp-powersgd-min-compression-rate 4.0 \
      --resume --save-format dcp --lr-decay-iters "$LR_DECAY_ITERS" \
      model.compile_model=false 2>&1 | tee "$out/train.log"
  fi
  printf '%s\tsuccess\t1\t%s\tcompleted\n' "$name" "$TARGET_STEP" >> "$RESULTS"
  echo "[$(date '+%F %T')] finished $name"
}

run_one dense allreduce_hook exact 29900
run_one gradient_row fp8_granularity_hook row 29901
run_one gradient_column fp8_granularity_hook column 29902
run_one gradient_block fp8_granularity_hook block 29903
run_one per_tensor fp8_per_tensor_hook per_tensor 29904
