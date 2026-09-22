#!/usr/bin/env bash
set -euo pipefail

# Qwen MATH-500: initial, RPO-5/RPO-10, and only validation-improving refiners.
# One vLLM model stays loaded for five separately seeded final-test runs.
# Local launch:
#   nohup bash codes/run_math500_final_test_qwen.sh \
#     > codes/nohup_outs/math500_final_test_qwen.log 2>&1 &
# Validation only:
#   FINAL_TEST_DRY_RUN=1 bash codes/run_math500_final_test_qwen.sh

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"
export CUDA_VISIBLE_DEVICES="${RUN_GPU:-1}"

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
  printf 'RUN_TAG must contain only letters, numbers, underscores, or hyphens.\n' >&2
  exit 2
fi
EXTRA_ARGS=()
if [[ "${FINAL_TEST_DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

"${PYTHON_BIN:-python}" -u codes/run_math500_final_test_evaluation.py \
  --code "math500_reasoning_qwen_selected_refiners_5seeds_${RUN_TAG}" \
  --manifest experiment_tracking/final_test/math500_qwen_selected_prompts.tsv \
  --dataset data/processed/math500/test.jsonl \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --gpu-memory-utilization "${FINAL_TEST_GPU_RATIO:-0.9}" \
  --vllm-max-model-len 32768 \
  --max-new-tokens 4096 \
  --seeds 42 1 100 1000 10000 \
  --output-root outputs/math_final_test/reasoning/selected_prompts \
  "${EXTRA_ARGS[@]}"
