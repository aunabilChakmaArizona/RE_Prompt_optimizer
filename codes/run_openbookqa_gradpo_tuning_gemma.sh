#!/usr/bin/env bash
set -euo pipefail

# OpenBookQA non-reasoning GradPO tuning for Gemma.
# Active section: ten cases for the distinct lower Gemma RPO checkpoint.
# Experimental RPO-5 uses actual RPO iteration 3; experimental RPO-10 uses
# actual iteration 4, which was retained unchanged at iterations 5 and 10.
# nohup bash codes/run_openbookqa_gradpo_tuning_gemma.sh > codes/nohup_outs/openbookqa_gradpo_tuning_gemma_itr3_as_rpo5.log 2>&1 &

SOURCE_PROMPT="outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_gemma_rpo_gemma12opt_lambda1/prompt_iteration_3.txt"
HISTORICAL_RPO5_SOURCE_PROMPT="outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_gemma_rpo_gemma12opt_lambda1/prompt_iteration_5.txt"
RPO10_SOURCE_PROMPT="outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_gemma_rpo_gemma12opt_lambda1/prompt_iteration_10.txt"

[[ -f "${SOURCE_PROMPT}" ]] || {
  echo "Missing first-stage prompt: ${SOURCE_PROMPT}"
  exit 1
}

[[ -f "${RPO10_SOURCE_PROMPT}" ]] || {
  echo "Missing first-stage prompt: ${RPO10_SOURCE_PROMPT}"
  exit 1
}

# Completed fluency-tuning runs; retained as history and not rerun.
if false; then
CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_gemma_rpo5_gradpo_tune_s5_t5_c7_g300_b10_f025 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file "${HISTORICAL_RPO5_SOURCE_PROMPT}" \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --hf-device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.6 \
  --vllm-max-model-len 16384 \
  --vllm-disable-images \
  --variant gen \
  --train-sample-size 600 \
  --gradient-sample-size 300 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 5 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 7 \
  --beam-width 10 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.25 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --seed 42
fi

# -----------------------------------------------------------------------------
# Active Gemma RPO-5 and RPO-10 tuning matrices (fluency lambda = 0.5)
#
# Each source has ten cases:
#   1-6. spans {3,5} x maximum tokens/span {1,3,5}
#   7.   gradient subset 200
#   8.   candidates/span 5
#   9.   beam width 5
#   10.  repeat the strongest setting with seed 43
# RPO-5 S5/T5/C7/G300/B10/seed42 already completed and is retained below,
# so it is omitted from the active calls.

TUNING_GPU="${TUNING_GPU:-1}"

run_gemma_case() {
  # Run one Gemma GradPO configuration from the compact matrices below.
  local source_prompt="$1"
  local code="$2"
  local spans="$3"
  local tokens_per_span="$4"
  local candidates="$5"
  local gradient_subset="$6"
  local beam_width="$7"
  local seed="$8"
  local threshold="${9:-0.6}"

  echo "Running ${code} on CUDA_VISIBLE_DEVICES=${TUNING_GPU}"
  CUDA_VISIBLE_DEVICES="${TUNING_GPU}" python -u codes/run_qa_promptopt_gradpo.py \
    --code "${code}" \
    --qa-task openbookqa \
    --qa-mode non_reasoning \
    --train-path data/processed/openbookqa/train.jsonl \
    --validation-path data/processed/openbookqa/validation.jsonl \
    --initial-prompt-file "${source_prompt}" \
    --model google/gemma-3-4b-it \
    --device cuda:0 \
    --hf-device cuda:0 \
    --target-max-new-tokens 10 \
    --validation-std-penalty 1.0 \
    --output-root outputs/qa_prompt_optimization \
    --overwrite \
    --backend dual \
    --final-evaluation-backend vllm \
    --dual-vllm-gpu-memory-utilization 0.6 \
    --vllm-max-model-len 16384 \
    --vllm-disable-images \
    --variant gen \
    --train-sample-size 600 \
    --gradient-sample-size "${gradient_subset}" \
    --gradient-batch-size 2 \
    --selection-batch-size 4 \
    --num-edit-regions "${spans}" \
    --max-region-tokens "${tokens_per_span}" \
    --region-expansion-threshold "${threshold}" \
    --num-region-candidates "${candidates}" \
    --beam-width "${beam_width}" \
    --beam-replacement-mode llm_synthesis \
    --fluency-lambda 0.5 \
    --candidate-max-new-tokens 10000 \
    --synthesis-max-new-tokens 10000 \
    --synthesis-batch-size 4 \
    --seed "${seed}"
}

# Ten active cases for actual iteration 3, used as the lower/RPO-5 source.
# Completed successfully; retained as commented run history.
# run_gemma_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_from_itr3_gradpo_tune_s5_t3_h060_c7_g200_b10_f050 5 3 7 200 10 42 0.60
# run_gemma_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_from_itr3_gradpo_tune_s3_t3_h060_c7_g200_b10_f050 3 3 7 200 10 42 0.60
# run_gemma_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_from_itr3_gradpo_tune_s4_t3_h060_c7_g200_b10_f050 4 3 7 200 10 42 0.60
# run_gemma_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_from_itr3_gradpo_tune_s5_t1_h060_c7_g200_b10_f050 5 1 7 200 10 42 0.60
# run_gemma_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_from_itr3_gradpo_tune_s5_t5_h060_c7_g200_b10_f050 5 5 7 200 10 42 0.60

# Resume here: the first case below previously failed while starting vLLM.
run_gemma_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_from_itr3_gradpo_tune_s5_t3_h045_c7_g200_b10_f050 5 3 7 200 10 42 0.45
run_gemma_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_from_itr3_gradpo_tune_s4_t3_h045_c7_g200_b10_f050 4 3 7 200 10 42 0.45
run_gemma_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_from_itr3_gradpo_tune_s5_t5_h030_c7_g200_b10_f050 5 5 7 200 10 42 0.30
run_gemma_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_from_itr3_gradpo_tune_s5_t3_h060_c10_g200_b10_f050 5 3 10 200 10 42 0.60
run_gemma_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_from_itr3_gradpo_tune_s5_t3_h060_c7_g300_b10_f050 5 3 7 300 10 42 0.60

# Completed duplicate-source matrices. Both sections below used the actual
# iteration-4 prompt because saved iterations 5 and 10 were byte-identical.
if false; then
# Historical Gemma RPO-5 label (actual source was iteration 4).
run_gemma_case "${HISTORICAL_RPO5_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_gradpo_tune_s3_t1_c7_g300_b10_f050 3 1 7 300 10 42
run_gemma_case "${HISTORICAL_RPO5_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_gradpo_tune_s3_t3_c7_g300_b10_f050 3 3 7 300 10 42
run_gemma_case "${HISTORICAL_RPO5_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_gradpo_tune_s3_t5_c7_g300_b10_f050 3 5 7 300 10 42
run_gemma_case "${HISTORICAL_RPO5_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_gradpo_tune_s5_t1_c7_g300_b10_f050 5 1 7 300 10 42
run_gemma_case "${HISTORICAL_RPO5_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_gradpo_tune_s5_t3_c7_g300_b10_f050 5 3 7 300 10 42
run_gemma_case "${HISTORICAL_RPO5_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_gradpo_tune_s5_t3_c7_g200_b10_f050 5 3 7 200 10 42
run_gemma_case "${HISTORICAL_RPO5_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_gradpo_tune_s5_t3_c5_g300_b10_f050 5 3 5 300 10 42
run_gemma_case "${HISTORICAL_RPO5_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_gradpo_tune_s5_t3_c7_g300_b5_f050 5 3 7 300 5 42
run_gemma_case "${HISTORICAL_RPO5_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo5_gradpo_tune_s5_t3_c7_g300_b10_f050_seed43 5 3 7 300 10 43

# Gemma RPO-10: all ten cases.
run_gemma_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo10_gradpo_tune_s3_t1_c7_g300_b10_f050 3 1 7 300 10 42
run_gemma_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo10_gradpo_tune_s3_t3_c7_g300_b10_f050 3 3 7 300 10 42
run_gemma_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo10_gradpo_tune_s3_t5_c7_g300_b10_f050 3 5 7 300 10 42
run_gemma_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo10_gradpo_tune_s5_t1_c7_g300_b10_f050 5 1 7 300 10 42
run_gemma_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo10_gradpo_tune_s5_t3_c7_g300_b10_f050 5 3 7 300 10 42
run_gemma_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo10_gradpo_tune_s5_t5_c7_g300_b10_f050 5 5 7 300 10 42
run_gemma_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo10_gradpo_tune_s5_t3_c7_g200_b10_f050 5 3 7 200 10 42
run_gemma_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo10_gradpo_tune_s5_t3_c5_g300_b10_f050 5 3 5 300 10 42
run_gemma_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo10_gradpo_tune_s5_t3_c7_g300_b5_f050 5 3 7 300 5 42
run_gemma_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_gemma_rpo10_gradpo_tune_s5_t3_c7_g300_b10_f050_seed43 5 3 7 300 10 43
fi

# Completed RPO-5 lambda-0.5 reference case; retained as history and not rerun.
if false; then
CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_gemma_rpo5_gradpo_tune_s5_t5_c7_g300_b10_f050 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file "${HISTORICAL_RPO5_SOURCE_PROMPT}" \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --hf-device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.6 \
  --vllm-max-model-len 16384 \
  --vllm-disable-images \
  --variant gen \
  --train-sample-size 600 \
  --gradient-sample-size 300 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 5 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 7 \
  --beam-width 10 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --seed 42
fi
