#!/usr/bin/env bash
set -euo pipefail

# Run only after the user explicitly starts the A800 job.  The shared hook is
# restored in a trap; no code change is intentionally left in the workspace.
ROOT="${ROOT:?ROOT required}"
SOURCE="${SOURCE:-/workspace/LoongForge/loongforge/embodied/distributed/ddp_utils/ddp_comm_hook.py}"
ARGS_SOURCE="${ARGS_SOURCE:-/workspace/LoongForge/loongforge/embodied/train/training_args.py}"
PATCH_BASE="${PATCH_BASE:?PATCH_BASE required}"
PREPARE_PATCH="${PREPARE_PATCH:?PREPARE_PATCH required}"
PREPARE_ARGS_PATCH="${PREPARE_ARGS_PATCH:?PREPARE_ARGS_PATCH required}"
CALIBRATION_RUNNER="${CALIBRATION_RUNNER:?CALIBRATION_RUNNER required}"
FORMAL_RUNNER="${FORMAL_RUNNER:?FORMAL_RUNNER required}"
SUMMARIZE_ROUTE="${SUMMARIZE_ROUTE:?SUMMARIZE_ROUTE required}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:?BASE_CHECKPOINT required}"
CODE_ROOT="${CODE_ROOT:-/workspace/LoongForge}"
DATASET_PATH="${DATASET_PATH:-/raid0/fastwam/vla_artifacts/pi05/datasets/libero}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/raid0/fastwam/vla_artifacts/pi05/tokenizers/paligemma-3b-pt-224}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-/raid0/fastwam/vla_artifacts/pi05/models/pi05_base}"
ROUTE_PATH="$ROOT/route/per_tensor_route.json"
CALIBRATION_TARGET_STEP="${CALIBRATION_TARGET_STEP:-320}"
FORMAL_TARGET_STEP="${FORMAL_TARGET_STEP:-1000}"

mkdir -p "$ROOT/route" "$ROOT/staging"
printf 'variant\tresult\tattempt\tlast_step\tdetail\n' > "$ROOT/variant_results.tsv"
if [[ "$SUMMARIZE_ROUTE" != "$ROOT/staging/summarize_route.py" ]]; then
  cp -a "$SUMMARIZE_ROUTE" "$ROOT/staging/summarize_route.py"
fi
BACKUP="$ROOT/staging/ddp_comm_hook.original.$(date +%Y%m%d_%H%M%S).py"
ARGS_BACKUP="$ROOT/staging/training_args.original.$(date +%Y%m%d_%H%M%S).py"
cp -a "$SOURCE" "$BACKUP"
cp -a "$ARGS_SOURCE" "$ARGS_BACKUP"
ORIGINAL_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
ORIGINAL_ARGS_SHA="$(sha256sum "$ARGS_SOURCE" | awk '{print $1}')"
RESTORED=0
restore_shared_source() {
  if [[ "$RESTORED" == "1" ]]; then return; fi
  cp -a "$BACKUP" "$SOURCE"
  cp -a "$ARGS_BACKUP" "$ARGS_SOURCE"
  RESTORED=1
  AFTER_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
  AFTER_ARGS_SHA="$(sha256sum "$ARGS_SOURCE" | awk '{print $1}')"
  [[ "$AFTER_SHA" == "$ORIGINAL_SHA" ]] || {
    echo "ERROR shared hook restore checksum mismatch" >&2
    return 1
  }
  [[ "$AFTER_ARGS_SHA" == "$ORIGINAL_ARGS_SHA" ]] || {
    echo "ERROR shared training args restore checksum mismatch" >&2
    return 1
  }
}
trap restore_shared_source EXIT INT TERM

cp -a "$PATCH_BASE" "$SOURCE"
python3 "$PREPARE_PATCH" --path "$SOURCE"
python3 -m py_compile "$SOURCE"
python3 "$PREPARE_ARGS_PATCH" --path "$ARGS_SOURCE"
python3 -m py_compile "$ARGS_SOURCE"

export BASE_CHECKPOINT CODE_ROOT DATASET_PATH TOKENIZER_PATH PRETRAINED_CHECKPOINT ROUTE_PATH ROOT
# All required model/data artifacts are already cached in the Pod.  Prevent
# the LeRobot metadata resolver from making a network request during a long run.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
printf 'phase=calibration\nroute=%s\n' "$ROUTE_PATH" > "$ROOT/status.env"
TARGET_STEP="$CALIBRATION_TARGET_STEP" RUN_ROOT="$ROOT" bash "$CALIBRATION_RUNNER"
[[ -s "$ROUTE_PATH" ]] || { echo "ERROR route map was not produced" >&2; exit 1; }
python3 "$ROOT/staging/summarize_route.py" "$ROUTE_PATH" \
  --csv "$ROOT/route/per_tensor_route.csv" > "$ROOT/route/summary.json"
printf 'phase=formal_comparison\nroute=%s\n' "$ROUTE_PATH" > "$ROOT/status.env"
TARGET_STEP="$FORMAL_TARGET_STEP" RUN_ROOT="$ROOT" bash "$FORMAL_RUNNER"
printf 'phase=completed\nroute=%s\n' "$ROUTE_PATH" > "$ROOT/status.env"
