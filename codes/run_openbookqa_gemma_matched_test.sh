#!/usr/bin/env bash
set -euo pipefail

# Fresh Gemma checks: solo initial, then all 36 selected rows (30 unique prompts).
# Both use 16 output tokens, model-native 131072 context, images allowed,
# model-default sampling, seed 42, and GPU memory utilization 0.8.
# Images stay ALLOWED intentionally to match the healthy standalone engine path.
# Each command starts a fresh engine. Existing test runs/reports are not replaced.
# mkdir -p codes/nohup_outs
# nohup bash codes/run_openbookqa_gemma_matched_test.sh > codes/nohup_outs/openbookqa_gemma_matched_test.log 2>&1 &
# CPU-only manifest/test checks:
# FINAL_TEST_DRY_RUN=1 bash codes/run_openbookqa_gemma_matched_test.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${RUN_GPU:-3}"

RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "RUN_TAG must contain only letters, numbers, underscores, or hyphens." >&2
  exit 1
}
SOLO_CODE="openbookqa_non_reasoning_gemma_initial_matched16_${RUN_TAG}"
BATCH_CODE="openbookqa_non_reasoning_gemma_selected_matched16_${RUN_TAG}"
MANIFEST="experiment_tracking/final_test/openbookqa_selected_prompts.tsv"
INITIAL_PROMPT="outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_gemma_rpo_gemma12opt_lambda1/initial_prompt.txt"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Validate all 36 files/hashes/parent links and the official 500-question test
# BEFORE starting either engine. This path never writes evaluation outputs.
"${PYTHON_BIN}" -u codes/run_openbookqa_final_test_evaluation.py \
  --family gemma \
  --code "${BATCH_CODE}" \
  --manifest "${MANIFEST}" \
  --test-path data/processed/openbookqa/test.jsonl \
  --vllm-max-model-len 131072 \
  --max-new-tokens 16 \
  --dry-run

if [[ "${FINAL_TEST_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

echo "[gemma-matched:1/2] solo initial | CODE=${SOLO_CODE} | GPU=${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" -u codes/run_qa_final_test_evaluation.py \
  --code "${SOLO_CODE}" \
  --qa-mode non_reasoning \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization "${FINAL_TEST_GPU_RATIO:-0.8}" \
  --test-path data/processed/openbookqa/test.jsonl \
  --prompt-file "${INITIAL_PROMPT}" \
  --runs 1 \
  --max-new-tokens 16 \
  --seed 42 \
  --output-root outputs/qa_final_test

echo "[gemma-matched:2/2] all selected prompts | CODE=${BATCH_CODE} | GPU=${CUDA_VISIBLE_DEVICES}"
"${PYTHON_BIN}" -u codes/run_openbookqa_final_test_evaluation.py \
  --family gemma \
  --code "${BATCH_CODE}" \
  --manifest "${MANIFEST}" \
  --test-path data/processed/openbookqa/test.jsonl \
  --backend vllm \
  --device cuda:0 \
  --gpu-memory-utilization "${FINAL_TEST_GPU_RATIO:-0.8}" \
  --vllm-max-model-len 131072 \
  --max-new-tokens 16 \
  --seed 42 \
  --output-root outputs/qa_final_test

echo "[gemma-matched] solo: outputs/qa_final_test/non_reasoning/final_test/${SOLO_CODE}/"
echo "[gemma-matched] batch: outputs/qa_final_test/non_reasoning/selected_prompts/${BATCH_CODE}/"
# Do not invoke the historical combined report: it reads the OLD fixed CODEs.
