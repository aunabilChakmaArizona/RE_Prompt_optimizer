#!/usr/bin/env bash
set -euo pipefail

# Qwen ONLY: five separate latest-RPO batches, with ONE model load.
# Initial + two first-stage sources + twelve refiner rows = 15 rows/seed.
# The six refiners are evaluated for both RPO-5 and RPO-10; identical prompts
# share predictions automatically.
# GPU 3 by default; do not run Gemma concurrently on the SAME GPU at ratio 0.8.
# mkdir -p codes/nohup_outs
# nohup bash codes/run_openbookqa_final_test_qwen_5runs.sh > codes/nohup_outs/openbookqa_final_test_qwen_5runs.log 2>&1 &
# FINAL_TEST_DRY_RUN=1 bash codes/run_openbookqa_final_test_qwen_5runs.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${RUN_GPU:-1}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "RUN_TAG must contain only letters, numbers, underscores, or hyphens." >&2
  exit 1
}
EXTRA_ARGS=()
if [[ "${FINAL_TEST_DRY_RUN:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

"${PYTHON_BIN:-python}" -u codes/run_openbookqa_final_test_evaluation.py \
  --family qwen \
  --code "openbookqa_non_reasoning_qwen_latest_rpo_six_refiners_5seeds_${RUN_TAG}" \
  --manifest experiment_tracking/final_test/openbookqa_qwen_latest_rpo_final_prompts.tsv \
  --source-slots rpo5 rpo10 \
  --test-path data/processed/openbookqa/test.jsonl \
  --backend vllm \
  --device cuda:0 \
  --gpu-memory-utilization "${FINAL_TEST_GPU_RATIO:-0.9}" \
  --vllm-max-model-len 40960 \
  --max-new-tokens 16 \
  --seeds 42 1 100 1000 10000 \
  --output-root outputs/qa_final_test \
  "${EXTRA_ARGS[@]}"

# Each seed's results are kept separately; mean/std will be reported later.
# Timestamped CODE and no --overwrite preserve all earlier tests.
