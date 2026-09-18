#!/usr/bin/env bash
set -euo pipefail

# Fresh standalone check of ONLY Gemma's original OpenBookQA instruction.
# One test decoding run; non-reasoning; GPU 3; vLLM memory ratio 0.8; output limit 16.
# The separate retry CODE preserves the anomalous batched baseline result.
# mkdir -p codes/nohup_outs
# nohup bash codes/run_openbookqa_initial_test_gemma.sh > codes/nohup_outs/openbookqa_initial_test_gemma_retry.log 2>&1 &

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${RUN_GPU:-3}"

PROMPT_FILE="outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_gemma_rpo_gemma12opt_lambda1/initial_prompt.txt"
[[ -f "${PROMPT_FILE}" ]] || {
  echo "Missing initial prompt: ${PROMPT_FILE}" >&2
  exit 1
}

"${PYTHON_BIN:-python}" -u codes/run_qa_final_test_evaluation.py \
  --code openbookqa_non_reasoning_gemma_initial_test_once_retry \
  --qa-mode non_reasoning \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization "${FINAL_TEST_GPU_RATIO:-0.8}" \
  --test-path data/processed/openbookqa/test.jsonl \
  --prompt-file "${PROMPT_FILE}" \
  --runs 1 \
  --max-new-tokens 16 \
  --seed 42 \
  --output-root outputs/qa_final_test \
  --overwrite
