#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT="${RUN_ROOT:?RUN_ROOT required}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:?BASE_CHECKPOINT required}"
CODE_ROOT="${CODE_ROOT:-/workspace/LoongForge}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$CODE_ROOT/examples/embodied/cosmos3/run_cosmos3_nano_droid_ddp_finetune.sh}"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-/raid0/fastwam/vla_artifacts}"
DATA_PATH="${DATA_PATH:-$ARTIFACTS_ROOT/cosmos3/datasets/Cosmos3-DROID-subset}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ARTIFACTS_ROOT/cosmos3/tokenizers/Qwen3-VL-8B-Instruct}"
VAE_PATH="${VAE_PATH:-$ARTIFACTS_ROOT/cosmos3/models/Wan2.2_VAE/Wan2.2_VAE.pth}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ARTIFACTS_ROOT/cosmos3/models/Cosmos3-Nano}"
ROUTE_PATH="${ROUTE_PATH:?ROUTE_PATH required}"
OUT="$RUN_ROOT/calibration"
mkdir -p "$OUT/checkpoints"
ln -sfn "$BASE_CHECKPOINT" "$OUT/checkpoints/steps_600"
FP8_PER_TENSOR_MODE=calibrate \
FP8_PER_TENSOR_ROUTE_PATH="$ROUTE_PATH" \
FP8_PER_TENSOR_CALIBRATION_END_STEP=620 \
FP8_GRANULARITY_MIN_BYTES=65536 FP8_GRANULARITY_BLOCK=128 \
FP8_METRICS_PATH="$OUT/metrics.jsonl" FP8_GLOBAL_STEP_OFFSET=600 \
FP8_QUANT_START_STEP=603 \
PYTHONPATH="$CODE_ROOT:/workspace/Loong-Megatron" \
LOCAL_VLA_ARTIFACTS_ROOT="$ARTIFACTS_ROOT" RUN_ROOT="$RUN_ROOT" OUTPUT_DIR="$OUT" \
TENSORBOARD_DIR="$OUT/tensorboard" TRAIN_ITERS=621 GPUS_PER_NODE=8 \
PER_DEVICE_BATCH_SIZE=1 NUM_WORKERS=4 SEED=42 SAVE_INTERVAL=0 MASTER_PORT="${MASTER_PORT:-33200}" \
TOKENIZER_PATH="$TOKENIZER_PATH" VAE_PATH="$VAE_PATH" CHECKPOINT_PATH="$CHECKPOINT_PATH" \
DATA_PATH="$DATA_PATH" VIDEO_BACKEND=pyav \
bash "$TRAIN_SCRIPT" --ddp-comm-hook fp8_per_tensor_hook \
--ddp-powersgd-matrix-approximation-rank 1 --ddp-powersgd-start-iter 2 \
--ddp-powersgd-min-compression-rate 4.0 --ddp-powersgd-scheduled-start-step 603 \
--save-format dcp --lr-decay-iters 1000 --zero-optimizer \
--zero-parameters-as-bucket-view --resume > "$OUT/train.log" 2>&1
