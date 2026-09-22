#!/usr/bin/env bash
set -euo pipefail

# Test all selected OpenBookQA prompts together, loading each target model ONCE.
# Actual manifest count: 36 rows/model (not 32); identical prompts share predictions.
# GPU 3 by default; Qwen then Gemma run sequentially, not concurrently.
# All evaluations use ONE decoding run, non-reasoning, and max 10 output tokens.
# Both models:
# mkdir -p codes/nohup_outs
# nohup bash codes/run_openbookqa_final_test_batched.sh > codes/nohup_outs/openbookqa_final_test_batched.log 2>&1 &
# One model only:
# bash codes/run_openbookqa_final_test_batched.sh qwen
# bash codes/run_openbookqa_final_test_batched.sh gemma
# CPU-only checks (no inference or outputs):
# FINAL_TEST_DRY_RUN=1 bash codes/run_openbookqa_final_test_batched.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${RUN_GPU:-3}"

case "${1:-both}" in
  both) FAMILIES=(qwen gemma) ;;
  qwen|gemma) FAMILIES=("$1") ;;
  *) echo "Usage: bash $0 [both|qwen|gemma]" >&2; exit 1 ;;
esac

run_model() {
  # Submit all unique prompt-question pairs to one vLLM instance for this model.
  local family="$1"
  local extra_args=()
  if [[ "${FINAL_TEST_DRY_RUN:-0}" == "1" ]]; then
    extra_args+=(--dry-run)
  fi
  echo "[final-test:batched] model=${family} | GPU=${CUDA_VISIBLE_DEVICES} | decoding runs=1"
  "${PYTHON_BIN:-python}" -u codes/run_openbookqa_final_test_evaluation.py \
    --family "${family}" \
    --code "openbookqa_non_reasoning_${family}_selected_prompts_test_once" \
    --manifest experiment_tracking/final_test/openbookqa_selected_prompts.tsv \
    --test-path data/processed/openbookqa/test.jsonl \
    --backend vllm \
    --device cuda:0 \
    --gpu-memory-utilization "${FINAL_TEST_GPU_RATIO:-0.8}" \
    --vllm-max-model-len 16384 \
    --max-new-tokens 10 \
    --seed 42 \
    --output-root outputs/qa_final_test \
    --overwrite \
    "${extra_args[@]}"
}

for family in "${FAMILIES[@]}"; do
  run_model "${family}"
done

if [[ "${#FAMILIES[@]}" -eq 2 && "${FINAL_TEST_DRY_RUN:-0}" != "1" ]]; then
  # Combine both completed batch summaries into the central 72-row TXT table.
  "${PYTHON_BIN:-python}" -u codes/report_openbookqa_final_test_results.py
fi
