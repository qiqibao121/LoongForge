#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/raid0/fastwam/experiments}"
CODE_ROOT="${CODE_ROOT:-/workspace/LoongForge}"
BACKUP_ROOT="${BACKUP_ROOT:-/raid0/fastwam/results/pi05_low_rank_experiments_20260821}"
BOS_DEST="${BOS_DEST:-bos://aihc-ai-datasets-bj/cce-ai-datasets.bj.bcebos.com/ljh/pi05_low_rank_experiments_20260821/}"

EXPERIMENTS=(
  pi05_layerwise_powersgd_a800_0_300_v1
  pi05_powersgd_plus_acp_a800_0_1000_v1
  pi05_powersgd_plus_acp_a800_smoke_20260821_v5
  pi05_powersgd_plus_acp_resume_a800_100_1000_v1
  pi05_powersgd_plus_acp_resume_a800_100_1000_v2
  pi05_powersgd_plus_reproduce_old_100_300_v1
  pi05_lr_threshold_sweep_a800_100_1000_v1
  pi05_lr_threshold_sweep_a800_100_1000_v2
  pi05_loss_window_powersgd_a800_100_1000_v1
  pi05_lr_schedule_powersgd_a800_0_1000_v1
  pi05_powersgd_plus_ablation_start300_a800_100_1000_v1
  pi05_ef21_powersgd_start300_a800_100_1000_v1
)

mkdir -p "${BACKUP_ROOT}/experiments" "${BACKUP_ROOT}/code" "${BACKUP_ROOT}/metadata"

copy_experiment_metadata() {
  local name="$1"
  local src="${SOURCE_ROOT}/${name}"
  local dst="${BACKUP_ROOT}/experiments/${name}"
  [[ -d "${src}" ]] || return 0
  mkdir -p "${dst}"
  (
    cd "${src}"
    find . -type f \
      ! -path './checkpoints/*' \
      ! -path '*/checkpoints/*' \
      ! -path './tensorboard/*' \
      ! -path '*/tensorboard/*' \
      ! -path './code/*' \
      ! -path '*/.git/*' \
      ! -path '*/__pycache__/*' \
      ! -name 'core.*' \
      ! -name '*.pyc' \
      -size -128M -print0
  ) | while IFS= read -r -d '' rel; do
    mkdir -p "${dst}/$(dirname "${rel}")"
    cp -p "${src}/${rel#./}" "${dst}/${rel#./}"
  done
}

for experiment in "${EXPERIMENTS[@]}"; do
  copy_experiment_metadata "${experiment}"
done

# Preserve standalone smoke-test logs as well.
find "${SOURCE_ROOT}" -maxdepth 1 -type f -name 'pi05_*' -size -128M -print0 |
  while IFS= read -r -d '' file; do
    cp -p "${file}" "${BACKUP_ROOT}/experiments/"
  done

# Preserve the exact experiment launchers/analyzers and communication-hook code.
if [[ -d "${CODE_ROOT}/examples/embodied/pi05" ]]; then
  mkdir -p "${BACKUP_ROOT}/code/examples/embodied/pi05"
  find "${CODE_ROOT}/examples/embodied/pi05" -maxdepth 1 -type f \
    \( -name '*.sh' -o -name '*.py' -o -name '*.md' -o -name '*.json' -o -name '*.env' \) \
    -exec cp -p {} "${BACKUP_ROOT}/code/examples/embodied/pi05/" \;
fi

for rel in \
  loongforge/embodied/distributed/ddp_utils/ddp_comm_hook.py \
  loongforge/embodied/distributed/parallel.py \
  loongforge/embodied/train/training_args.py; do
  if [[ -f "${CODE_ROOT}/${rel}" ]]; then
    mkdir -p "${BACKUP_ROOT}/code/$(dirname "${rel}")"
    cp -p "${CODE_ROOT}/${rel}" "${BACKUP_ROOT}/code/${rel}"
  fi
done

cp -p "$0" "${BACKUP_ROOT}/metadata/backup_pi05_a800_results.sh"

{
  echo "snapshot_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "snapshot_local=$(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "hostname=$(hostname)"
  echo "source_root=${SOURCE_ROOT}"
  echo "code_root=${CODE_ROOT}"
  echo "bos_destination=${BOS_DEST}"
  echo "excluded=checkpoints,tensorboard,embedded_code_snapshots,core_dumps,git,pycache"
  echo "python=$(python --version 2>&1 || true)"
  python - <<'PY'
try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"cuda={torch.version.cuda}")
except Exception as exc:
    print(f"torch_probe_error={exc}")
PY
} > "${BACKUP_ROOT}/metadata/snapshot.env"

if git -C "${CODE_ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  git -C "${CODE_ROOT}" rev-parse HEAD > "${BACKUP_ROOT}/metadata/git_commit.txt"
  git -C "${CODE_ROOT}" status --short > "${BACKUP_ROOT}/metadata/git_status.txt"
  git -C "${CODE_ROOT}" diff -- \
    examples/embodied/pi05 \
    loongforge/embodied/distributed/ddp_utils/ddp_comm_hook.py \
    loongforge/embodied/distributed/parallel.py \
    loongforge/embodied/train/training_args.py \
    > "${BACKUP_ROOT}/metadata/relevant_code.diff"
fi

find "${BACKUP_ROOT}" -type f ! -name SHA256SUMS -printf '%P\n' | sort \
  > "${BACKUP_ROOT}/metadata/file_list.txt"
(
  cd "${BACKUP_ROOT}"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum
) > "${BACKUP_ROOT}/metadata/SHA256SUMS"
du -ah "${BACKUP_ROOT}" | sort -h > "${BACKUP_ROOT}/metadata/sizes.txt"

bcecmd bos sync "${BACKUP_ROOT}/" "${BOS_DEST}" \
  --yes --disable-bar --sync-type time-size-crc32

echo "backup_complete local=${BACKUP_ROOT} bos=${BOS_DEST}"
