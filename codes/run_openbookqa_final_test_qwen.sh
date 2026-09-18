#!/usr/bin/env bash
set -euo pipefail

# Official OpenBookQA test only; non-reasoning; one decoding run per prompt.
# Use the EXISTING single-prompt evaluator; each invocation reloads its model.
# Both model scripts default to physical GPU 3. Run them SEQUENTIALLY.
# mkdir -p codes/nohup_outs
# nohup bash codes/run_openbookqa_final_test_qwen.sh > codes/nohup_outs/openbookqa_final_test_qwen.log 2>&1 &
# CPU-only file/command checks:
# FINAL_TEST_DRY_RUN=1 bash codes/run_openbookqa_final_test_qwen.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${RUN_GPU:-3}"
FAMILY="qwen"
MODEL="Qwen/Qwen3-4B"
MANIFEST="experiment_tracking/final_test/openbookqa_selected_prompts.tsv"

# Check all selected files and hashes before starting any test inference.
selected_count=0
while IFS=$'\t' read -r family row_id stage source actual_iteration method optimization_code prompt_file prompt_sha256 parent_row_id; do
  [[ "${family}" == "${FAMILY}" ]] || continue
  [[ -f "${prompt_file}" ]] || {
    echo "Missing selected prompt: ${prompt_file}" >&2
    exit 1
  }
  actual_hash="$(sha256sum "${prompt_file}")"
  [[ "${actual_hash%% *}" == "${prompt_sha256}" ]] || {
    echo "Selected prompt changed: ${prompt_file}" >&2
    exit 1
  }
  selected_count=$((selected_count + 1))
done < "${MANIFEST}"

[[ "${selected_count}" -eq 36 ]] || {
  echo "Expected 36 selected rows for ${FAMILY}; found ${selected_count}." >&2
  exit 1
}
echo "[final-test:${FAMILY}] selected rows=${selected_count} | decoding runs=1 | CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

run_selected_prompt() {
  # Evaluate one saved instruction using the existing runner and its own CODE.
  local row_id="$1"
  local prompt_file="$2"
  local code="openbookqa_non_reasoning_${FAMILY}_${row_id}_test_once"
  local command=(
    "${PYTHON_BIN:-python}" -u codes/run_qa_final_test_evaluation.py
    --code "${code}"
    --qa-mode non_reasoning
    --model "${MODEL}"
    --device cuda:0
    --backend vllm
    --gpu-memory-utilization "${FINAL_TEST_GPU_RATIO:-0.8}"
    --test-path data/processed/openbookqa/test.jsonl
    --prompt-file "${prompt_file}"
    --runs 1
    --max-new-tokens 10
    --seed 42
    --output-root outputs/qa_final_test
    --overwrite
  )
  if [[ "${FINAL_TEST_DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

completed=0
while IFS=$'\t' read -r family row_id stage source actual_iteration method optimization_code prompt_file prompt_sha256 parent_row_id; do
  [[ "${family}" == "${FAMILY}" ]] || continue
  echo "[final-test:${FAMILY}] prompt $((completed + 1))/${selected_count} | ${row_id} | source CODE=${optimization_code}"
  run_selected_prompt "${row_id}" "${prompt_file}"
  completed=$((completed + 1))
done < "${MANIFEST}"

if [[ "${FINAL_TEST_DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run passed: 36 commands use the existing evaluator with --runs 1; no inference or outputs created."
else
  echo "Completed ${completed} prompts. Results: outputs/qa_final_test/non_reasoning/final_test/openbookqa_non_reasoning_${FAMILY}_*_test_once/"
fi
