#!/usr/bin/env bash
set -euo pipefail
ROOT=/raid0/fastwam/experiments/cosmos3_fp8_granularity_a800_600_1000_v1
STAGING="$ROOT/staging"
env ROOT="$ROOT" BASE_CHECKPOINT=/raid0/fastwam/experiments/cosmos3_start600_v_ablation_a800_100_1000_v2/dense_prefix_600/checkpoints/steps_600 \
  bash "$STAGING/launch_cosmos_fp8_granularity_a800.sh" > "$ROOT/controller.log" 2>&1
