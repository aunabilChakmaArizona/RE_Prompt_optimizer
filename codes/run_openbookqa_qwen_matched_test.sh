#!/usr/bin/env bash
set -euo pipefail

# Fresh Qwen-only tests: solo initial, then all 36 selected rows (27 unique).
# Like the Gemma retest: 16 output tokens, native context, memory ratio 0.8,
# seed 42, unchanged prompts/grading, one decoding run, fresh timestamped CODEs.
# Qwen's cached model config specifies native context 40960, not Gemma's 131072.
# mkdir -p codes/nohup_outs
# nohup bash codes/run_openbookqa_qwen_matched_test.sh > codes/nohup_outs/openbookqa_qwen_matched_test.log 2>&1 &
# FINAL_TEST_DRY_RUN=1 bash codes/run_openbookqa_qwen_matched_test.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${RUN_GPU:-3}"

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "RUN_TAG must contain only letters, numbers, underscores, or hyphens." >&2
  exit 1
}
SOLO_CODE="openbookqa_non_reasoning_qwen_initial_matched16_${RUN_TAG}"
BATCH_CODE="openbookqa_non_reasoning_qwen_selected_matched16_${RUN_TAG}"
MANIFEST="experiment_tracking/final_test/openbookqa_selected_prompts.tsv"
INITIAL_PROMPT="outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/initial_prompt.txt"
PYTHON_BIN="${PYTHON_BIN:-python}"

# CPU preflight: validate every selected prompt/hash/parent and the test split.
"${PYTHON_BIN}" -u codes/run_openbookqa_final_test_evaluation.py \
  --family qwen \
  --code "${BATCH_CODE}" \
  --manifest "${MANIFEST}" \
  --test-path data/processed/openbookqa/test.jsonl \
  --vllm-max-model-len 40960 \
  --max-new-tokens 16 \
  --dry-run

if [[ "${FINAL_TEST_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

echo "[qwen-matched:1/2] solo initial | CODE=${SOLO_CODE} | GPU=${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" -u codes/run_qa_final_test_evaluation.py \
  --code "${SOLO_CODE}" \
  --qa-mode non_reasoning \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization "${FINAL_TEST_GPU_RATIO:-0.8}" \
  --test-path data/processed/openbookqa/test.jsonl \
  --prompt-file "${INITIAL_PROMPT}" \
  --runs 1 \
  --max-new-tokens 16 \
  --seed 42 \
  --output-root outputs/qa_final_test

echo "[qwen-matched:2/2] all selected prompts | CODE=${BATCH_CODE} | GPU=${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" -u codes/run_openbookqa_final_test_evaluation.py \
  --family qwen \
  --code "${BATCH_CODE}" \
  --manifest "${MANIFEST}" \
  --test-path data/processed/openbookqa/test.jsonl \
  --backend vllm \
  --device cuda:0 \
  --gpu-memory-utilization "${FINAL_TEST_GPU_RATIO:-0.8}" \
  --vllm-max-model-len 40960 \
  --max-new-tokens 16 \
  --seed 42 \
  --output-root outputs/qa_final_test

# Compare ALL rows by their row IDs and exact prompt identities, not position.
# No historical tables are overwritten and no best prompt is selected by test.
"${PYTHON_BIN}" -u codes/report_openbookqa_test_rerun.py \
  --previous-summary outputs/qa_final_test/non_reasoning/selected_prompts/openbookqa_non_reasoning_qwen_selected_prompts_test_once/summary.json \
  --new-summary "outputs/qa_final_test/non_reasoning/selected_prompts/${BATCH_CODE}/summary.json" \
  --solo-summary "outputs/qa_final_test/non_reasoning/final_test/${SOLO_CODE}/summary.json" \
  --output "experiment_tracking/final_test/openbookqa_qwen_matched16_comparison_${RUN_TAG}.txt"

echo "[qwen-matched] comparison: experiment_tracking/final_test/openbookqa_qwen_matched16_comparison_${RUN_TAG}.txt"
