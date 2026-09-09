#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/raid0/fastwam/experiments/pi05_fp8_granularity_a800_600_1000_v1}"
STAGING="$ROOT/staging"
SOURCE=/workspace/LoongForge/loongforge/embodied/distributed/ddp_utils/ddp_comm_hook.py
ARGS_SOURCE=/workspace/LoongForge/loongforge/embodied/train/training_args.py
mkdir -p "$STAGING"
# The shared checkout may be an unpatched base without the FP8 granularity
# registration anchor.  Reuse the previously validated Pi05 FP8 staging base
# when available; never modify that successful experiment directory.
FP8_BASE=/raid0/fastwam/experiments/pi05_fp8_granularity_per_tensor_a800_v1/staging/ddp_comm_hook.base.py
if [ -f "$FP8_BASE" ]; then
  cp -a "$FP8_BASE" "$STAGING/ddp_comm_hook.base.py"
else
  cp -a "$SOURCE" "$STAGING/ddp_comm_hook.base.py"
fi
cp -a "$ARGS_SOURCE" "$STAGING/training_args.base.py"
for script in launch_pi05_fp8_granularity_a800.sh run_calibration.sh \
    run_formal_comparison.sh prepare_per_tensor_fp8_patch.py \
    prepare_training_args_patch.py summarize_route.py; do
  cp -a "$ROOT/$script" "$STAGING/$script"
done
env ROOT="$ROOT" \
  SOURCE="$SOURCE" \
  ARGS_SOURCE="$ARGS_SOURCE" \
  PATCH_BASE="$STAGING/ddp_comm_hook.base.py" \
  PREPARE_PATCH="$STAGING/prepare_per_tensor_fp8_patch.py" \
  PREPARE_ARGS_PATCH="$STAGING/prepare_training_args_patch.py" \
  CALIBRATION_RUNNER="$STAGING/run_calibration.sh" \
  FORMAL_RUNNER="$STAGING/run_formal_comparison.sh" \
  SUMMARIZE_ROUTE="$STAGING/summarize_route.py" \
  BASE_CHECKPOINT=/raid0/fastwam/experiments/pi05_fp8_granularity_long_a800_v1/gradient_row/checkpoints/steps_300 \
  CODE_ROOT=/workspace/LoongForge \
  DATASET_PATH=/raid0/fastwam/vla_artifacts/pi05/datasets/libero \
  TOKENIZER_PATH=/raid0/fastwam/vla_artifacts/pi05/tokenizers/paligemma-3b-pt-224 \
  PRETRAINED_CHECKPOINT=/raid0/fastwam/vla_artifacts/pi05/models/pi05_base \
  MASTER_PORT="${MASTER_PORT:-29761}" \
  CALIBRATION_TARGET_STEP="${CALIBRATION_TARGET_STEP:-320}" \
  FORMAL_TARGET_STEP="${FORMAL_TARGET_STEP:-1000}" \
  bash "$STAGING/launch_pi05_fp8_granularity_a800.sh" > "$ROOT/controller.log" 2>&1
