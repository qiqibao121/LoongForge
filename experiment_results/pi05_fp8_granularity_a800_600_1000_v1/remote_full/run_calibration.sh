#!/usr/bin/env bash
set -euo pipefail
TRAIN_SCRIPT="${TRAIN_SCRIPT:-/workspace/LoongForge/examples/embodied/pi05/run_pi05_ddp_finetune.sh}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT required}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:?BASE_CHECKPOINT required}"
CODE_ROOT="${CODE_ROOT:?CODE_ROOT required}"
ROUTE_PATH="${ROUTE_PATH:?ROUTE_PATH required}"
DATASET_PATH="${DATASET_PATH:-/raid0/fastwam/vla_artifacts/pi05/datasets/libero}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/raid0/fastwam/vla_artifacts/pi05/tokenizers/paligemma-3b-pt-224}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/raid0/fastwam/vla_artifacts/pi05/models/pi05_base}"
TARGET_STEP="${TARGET_STEP:-320}"
OUT="$RUN_ROOT/calibration"
mkdir -p "$OUT/checkpoints"
[[ -e "$OUT/checkpoints/steps_300" ]] || ln -s "$BASE_CHECKPOINT" "$OUT/checkpoints/steps_300"
FP8_PER_TENSOR_MODE=calibrate FP8_PER_TENSOR_ROUTE_PATH="$ROUTE_PATH" \
FP8_PER_TENSOR_CALIBRATION_STEPS=20 FP8_PER_TENSOR_LOG="$OUT/per_tensor_stats.jsonl" \
PYTHONPATH="$CODE_ROOT:/workspace/LoongForge:/workspace/Loong-Megatron" \
RUN_ROOT="$RUN_ROOT" OUTPUT_DIR="$OUT" TENSORBOARD_DIR="$OUT/tensorboard" \
TRAIN_ITERS="$TARGET_STEP" GPUS_PER_NODE=8 PER_DEVICE_BATCH_SIZE=12 NUM_WORKERS=4 \
SEED=42 SAVE_INTERVAL=0 MASTER_PORT="${MASTER_PORT:-29740}" \
bash "$TRAIN_SCRIPT" --dataset-path "$DATASET_PATH" \
--tokenizer-path "$TOKENIZER_PATH" --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
--zero-optimizer --zero-parameters-as-bucket-view \
--ddp-comm-hook fp8_per_tensor_hook --ddp-powersgd-matrix-approximation-rank 1 \
--ddp-powersgd-start-iter 2 --ddp-powersgd-min-compression-rate 4.0 \
--resume --save-format dcp --lr-decay-iters 1000 model.compile_model=false 2>&1 | tee "$OUT/train.log"
