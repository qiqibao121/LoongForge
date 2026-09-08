#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:?ROOT required}"
SOURCE="${SOURCE:-/workspace/LoongForge/loongforge/embodied/distributed/ddp_utils/ddp_comm_hook.py}"
ARGS_SOURCE="${ARGS_SOURCE:-/workspace/LoongForge/loongforge/embodied/train/training_args.py}"
STAGING="$ROOT/staging"
PATCH_BASE="$STAGING/ddp_comm_hook.base.py"
PREPARE_FIXED="$STAGING/prepare_fixed_fp8_patch.py"
PREPARE_PER_TENSOR="$STAGING/prepare_per_tensor_fp8_patch.py"
PREPARE_COSMOS="$STAGING/prepare_cosmos_per_tensor_patch.py"
PREPARE_ARGS="$STAGING/prepare_training_args_patch.py"
BASE_CHECKPOINT="${BASE_CHECKPOINT:?BASE_CHECKPOINT required}"
mkdir -p "$ROOT/route" "$ROOT/staging" "$ROOT/calibration"
BACKUP="$STAGING/ddp_comm_hook.original.$(date +%Y%m%d_%H%M%S).py"
ARGS_BACKUP="$STAGING/training_args.original.$(date +%Y%m%d_%H%M%S).py"
cp -a "$SOURCE" "$BACKUP"
cp -a "$ARGS_SOURCE" "$ARGS_BACKUP"
ORIGINAL_SHA="$(sha256sum "$SOURCE" | awk '{print $1}')"
ORIGINAL_ARGS_SHA="$(sha256sum "$ARGS_SOURCE" | awk '{print $1}')"
RESTORED=0
restore_shared() {
  [[ "$RESTORED" == 1 ]] && return
  cp -a "$BACKUP" "$SOURCE"
  cp -a "$ARGS_BACKUP" "$ARGS_SOURCE"
  RESTORED=1
  [[ "$(sha256sum "$SOURCE" | awk '{print $1}')" == "$ORIGINAL_SHA" ]]
  [[ "$(sha256sum "$ARGS_SOURCE" | awk '{print $1}')" == "$ORIGINAL_ARGS_SHA" ]]
}
trap restore_shared EXIT INT TERM
# The shared hook evolves with the main codebase.  Refresh the staged base
# when the checked-in reference predates symbols imported by ddp_utils.__init__;
# this keeps the experiment self-contained without overwriting the shared file.
if ! grep -q "class ACPPowerSGDPlusState" "$PATCH_BASE" && grep -q "class ACPPowerSGDPlusState" "$SOURCE"; then
  cp -a "$SOURCE" "$PATCH_BASE"
fi
cp -a "$PATCH_BASE" "$SOURCE"
python3 "$PREPARE_FIXED" --path "$SOURCE"
python3 "$PREPARE_PER_TENSOR" --path "$SOURCE"
python3 "$PREPARE_COSMOS" --path "$SOURCE"
python3 -m py_compile "$SOURCE"
python3 "$PREPARE_ARGS" --path "$ARGS_SOURCE"
python3 -m py_compile "$ARGS_SOURCE"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
printf 'phase=calibration\nvariant=per_tensor\ndetail=steps_603_620_finalize_621\n' > "$ROOT/status.env"
RUN_ROOT="$ROOT" ROUTE_PATH="$ROOT/route/per_tensor_route.json" BASE_CHECKPOINT="$BASE_CHECKPOINT" \
  bash "$STAGING/run_calibration.sh"
[[ -s "$ROOT/route/per_tensor_route.json" ]] || { echo 'ERROR route missing' >&2; exit 1; }
python3 "$STAGING/summarize_route.py" "$ROOT/route/per_tensor_route.json" \
  --csv "$ROOT/route/per_tensor_route.csv" > "$ROOT/route/summary.json"
printf 'phase=formal\nvariant=all\ndetail=step_600_to_1000_serial\n' > "$ROOT/status.env"
RUN_ROOT="$ROOT" BASE_CHECKPOINT="$BASE_CHECKPOINT" bash "$STAGING/run_formal_comparison.sh"
if tail -n +2 "$ROOT/variant_results.tsv" | grep -q $'\tfailed\t'; then
  printf 'phase=completed_with_failures\nvariant=all\ndetail=serial_five_variant_comparison\n' > "$ROOT/status.env"
else
  printf 'phase=completed\nvariant=all\ndetail=serial_five_variant_comparison\n' > "$ROOT/status.env"
fi
