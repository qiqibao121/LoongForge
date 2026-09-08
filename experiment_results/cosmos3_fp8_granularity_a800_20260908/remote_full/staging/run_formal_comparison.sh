#!/usr/bin/env bash
set -uo pipefail
RUN_ROOT="${RUN_ROOT:?RUN_ROOT required}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:?BASE_CHECKPOINT required}"
CODE_ROOT="${CODE_ROOT:-/workspace/LoongForge}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$CODE_ROOT/examples/embodied/cosmos3/run_cosmos3_nano_droid_ddp_finetune.sh}"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-/raid0/fastwam/vla_artifacts}"
DATA_PATH="${DATA_PATH:-$ARTIFACTS_ROOT/cosmos3/datasets/Cosmos3-DROID-subset}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$ARTIFACTS_ROOT/cosmos3/tokenizers/Qwen3-VL-8B-Instruct}"
VAE_PATH="${VAE_PATH:-$ARTIFACTS_ROOT/cosmos3/models/Wan2.2_VAE/Wan2.2_VAE.pth}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ARTIFACTS_ROOT/cosmos3/models/Cosmos3-Nano}"
TARGET_STEP="${TARGET_STEP:-1000}"
RESULTS="$RUN_ROOT/variant_results.tsv"

wait_idle() {
  local stable=0 busy m u
  while ((stable < 3)); do
    busy=0
    while IFS=, read -r m u; do
      m=${m// /}; u=${u// /}
      if ((m >= 1024 || u >= 10)); then busy=1; break; fi
    done < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
    if ((busy == 0)); then stable=$((stable + 1)); else stable=0; fi
    ((stable == 3)) || sleep 10
  done
}

last_step() { [[ -f "$1" ]] && sed -nE 's/.*"step": ([0-9]+).*/\1/p' "$1" | tail -1 || echo 0; }
classify() {
  local log=$1
  grep -Eqi 'CUDA out of memory|OutOfMemoryError|torch\.OutOfMemoryError' "$log" && { echo cuda_oom; return; }
  grep -Eqi 'NCCL watchdog|NCCL.*error|illegal memory access' "$log" && { echo cuda_or_nccl_error; return; }
  grep -Eqi '(^|[^[:alpha:]])(nan|inf|infinity)([^[:alpha:]]|$)|non-finite' "$log" && { echo numerical_nan_or_inf; return; }
  grep -q Traceback "$log" && { echo code_error; return; }
  echo unknown_runtime
}

run_one() {
  local name=$1 kind=$2 port=$3 out="$RUN_ROOT/$1" log="$RUN_ROOT/$1/train.log"
  mkdir -p "$out/checkpoints"
  ln -sfn "$BASE_CHECKPOINT" "$out/checkpoints/steps_600"
  wait_idle
  printf 'phase=running\nvariant=%s\ndetail=target_step_%s\n' "$name" "$TARGET_STEP" > "$RUN_ROOT/status.env"
  local hook_env=() hook_args=()
  if [[ "$kind" == dense ]]; then
    hook_args=(--ddp-comm-hook allreduce_hook)
  elif [[ "$kind" == per_tensor ]]; then
    hook_env=(FP8_PER_TENSOR_MODE=route FP8_PER_TENSOR_ROUTE_PATH="$RUN_ROOT/route/per_tensor_route.json" FP8_PER_TENSOR_CALIBRATION_END_STEP=620 FP8_GRANULARITY_MIN_BYTES=65536 FP8_GRANULARITY_BLOCK=128 FP8_METRICS_PATH="$out/metrics.jsonl" FP8_GLOBAL_STEP_OFFSET=600 FP8_QUANT_START_STEP=603 FP8_PER_TENSOR_LOG="$out/per_tensor_stats.jsonl")
    hook_args=(--ddp-comm-hook fp8_per_tensor_hook)
  else
    hook_env=(FP8_GRANULARITY_MODE="$kind" FP8_GRANULARITY_MIN_BYTES=65536 FP8_GRANULARITY_BLOCK=128 FP8_METRICS_PATH="$out/metrics.jsonl" FP8_GLOBAL_STEP_OFFSET=600 FP8_QUANT_START_STEP=603 FP8_GRANULARITY_LOG="$out/fp8_tensor_stats.jsonl")
    hook_args=(--ddp-comm-hook fp8_granularity_hook)
  fi
  set +e
  env PYTHONPATH="$CODE_ROOT:/workspace/Loong-Megatron" LOCAL_VLA_ARTIFACTS_ROOT="$ARTIFACTS_ROOT" \
    RUN_ROOT="$RUN_ROOT" OUTPUT_DIR="$out" TENSORBOARD_DIR="$out/tensorboard" \
    TRAIN_ITERS="$TARGET_STEP" GPUS_PER_NODE=8 PER_DEVICE_BATCH_SIZE=1 NUM_WORKERS=4 \
    SEED=42 SAVE_INTERVAL=0 MASTER_PORT="$port" TOKENIZER_PATH="$TOKENIZER_PATH" \
    VAE_PATH="$VAE_PATH" CHECKPOINT_PATH="$CHECKPOINT_PATH" DATA_PATH="$DATA_PATH" \
    VIDEO_BACKEND=pyav PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128" \
    "${hook_env[@]}" bash "$TRAIN_SCRIPT" "${hook_args[@]}" \
    --ddp-powersgd-matrix-approximation-rank 1 --ddp-powersgd-start-iter 2 \
    --ddp-powersgd-min-compression-rate 4.0 --ddp-powersgd-scheduled-start-step 603 \
    --save-format dcp --lr-decay-iters 1000 --zero-optimizer \
    --zero-parameters-as-bucket-view --resume > "$log" 2>&1
  local rc=$?
  set -e
  local s="$(last_step "$out/metrics.jsonl")"
  if [[ "$rc" == 0 && "$s" == "$TARGET_STEP" ]]; then
    printf '%s\tsuccess\t1\t%s\tcompleted\n' "$name" "$s" >> "$RESULTS"
    return 0
  fi
  printf '%s\tfailed\t1\t%s\tclass=%s,rc=%s\n' "$name" "$s" "$(classify "$log")" "$rc" >> "$RESULTS"
  return 1
}

printf 'variant\tresult\tattempt\tlast_step\tdetail\n' > "$RESULTS"
run_one dense dense 33201 || true
run_one gradient_row row 33202 || true
run_one gradient_column column 33203 || true
run_one gradient_block block 33204 || true
run_one per_tensor per_tensor 33205 || true
