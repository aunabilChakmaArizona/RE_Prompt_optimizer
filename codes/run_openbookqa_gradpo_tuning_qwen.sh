#!/usr/bin/env bash
set -euo pipefail

# OpenBookQA GradPO threshold/T/candidate tuning for Qwen RPO-5 and RPO-10.
# Active section: ten configurations applied to both RPO-5 and RPO-10.
# Five configurations use G=300 and the same five use G=200.
# nohup bash codes/run_openbookqa_gradpo_tuning_qwen.sh > codes/nohup_outs/openbookqa_qwen_next_tuning.log 2>&1 &

SOURCE_PROMPT="outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_5.txt"
RPO10_SOURCE_PROMPT="outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_10.txt"

[[ -f "${SOURCE_PROMPT}" ]] || {
  echo "Missing first-stage prompt: ${SOURCE_PROMPT}"
  exit 1
}

[[ -f "${RPO10_SOURCE_PROMPT}" ]] || {
  echo "Missing first-stage prompt: ${RPO10_SOURCE_PROMPT}"
  exit 1
}

# Completed RPO-10 reference case; retained as history and not rerun.
if false; then
CUDA_VISIBLE_DEVICES=1 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_rpo10_gradpo_tuned_s5_t3_c7_g300_b10_f050 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file "${RPO10_SOURCE_PROMPT}" \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --hf-device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen \
  --train-sample-size 800 \
  --gradient-sample-size 300 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 3 \
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

# Completed RPO-5 tuning commands below are retained only as run history.

# CUDA_VISIBLE_DEVICES=1 python -u codes/run_qa_promptopt_gradpo.py \
#   --code openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t5_c7_g300_b10_f025 \
#   --qa-task openbookqa \
#   --qa-mode non_reasoning \
#   --train-path data/processed/openbookqa/train.jsonl \
#   --validation-path data/processed/openbookqa/validation.jsonl \
#   --initial-prompt-file "${SOURCE_PROMPT}" \
#   --model Qwen/Qwen3-4B \
#   --device cuda:0 \
#   --hf-device cuda:0 \
#   --target-max-new-tokens 10 \
#   --validation-std-penalty 1.0 \
#   --output-root outputs/qa_prompt_optimization \
#   --overwrite \
#   --backend dual \
#   --final-evaluation-backend vllm \
#   --dual-vllm-gpu-memory-utilization 0.5 \
#   --vllm-max-model-len 16384 \
#   --variant gen \
#   --train-sample-size 800 \
#   --gradient-sample-size 300 \
#   --gradient-batch-size 2 \
#   --selection-batch-size 4 \
#   --num-edit-regions 5 \
#   --max-region-tokens 5 \
#   --region-expansion-threshold 0.6 \
#   --num-region-candidates 7 \
#   --beam-width 10 \
#   --beam-replacement-mode llm_synthesis \
#   --fluency-lambda 0.25 \
#   --candidate-max-new-tokens 10000 \
#   --synthesis-max-new-tokens 10000 \
#   --synthesis-batch-size 4 \
#   --seed 42

# CUDA_VISIBLE_DEVICES=1 python -u codes/run_qa_promptopt_gradpo.py \
#   --code openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t5_c7_g300_b10_f050 \
#   --qa-task openbookqa \
#   --qa-mode non_reasoning \
#   --train-path data/processed/openbookqa/train.jsonl \
#   --validation-path data/processed/openbookqa/validation.jsonl \
#   --initial-prompt-file "${SOURCE_PROMPT}" \
#   --model Qwen/Qwen3-4B \
#   --device cuda:0 \
#   --hf-device cuda:0 \
#   --target-max-new-tokens 10 \
#   --validation-std-penalty 1.0 \
#   --output-root outputs/qa_prompt_optimization \
#   --overwrite \
#   --backend dual \
#   --final-evaluation-backend vllm \
#   --dual-vllm-gpu-memory-utilization 0.5 \
#   --vllm-max-model-len 16384 \
#   --variant gen \
#   --train-sample-size 800 \
#   --gradient-sample-size 300 \
#   --gradient-batch-size 2 \
#   --selection-batch-size 4 \
#   --num-edit-regions 5 \
#   --max-region-tokens 5 \
#   --region-expansion-threshold 0.6 \
#   --num-region-candidates 7 \
#   --beam-width 10 \
#   --beam-replacement-mode llm_synthesis \
#   --fluency-lambda 0.5 \
#   --candidate-max-new-tokens 10000 \
#   --synthesis-max-new-tokens 10000 \
#   --synthesis-batch-size 4 \
#   --seed 42

# Stage 2: all remaining span-count x token-per-span combinations.
# Fixed settings: fluency=0.5, gradient subset=300, candidates=7, beam=10.
# Completed and disabled; retained as experiment history.

# if false; then

# CUDA_VISIBLE_DEVICES=1 python -u codes/run_qa_promptopt_gradpo.py \
#   --code openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s3_t1_c7_g300_b10_f050 \
#   --qa-task openbookqa \
#   --qa-mode non_reasoning \
#   --train-path data/processed/openbookqa/train.jsonl \
#   --validation-path data/processed/openbookqa/validation.jsonl \
#   --initial-prompt-file "${SOURCE_PROMPT}" \
#   --model Qwen/Qwen3-4B \
#   --device cuda:0 \
#   --hf-device cuda:0 \
#   --target-max-new-tokens 10 \
#   --validation-std-penalty 1.0 \
#   --output-root outputs/qa_prompt_optimization \
#   --overwrite \
#   --backend dual \
#   --final-evaluation-backend vllm \
#   --dual-vllm-gpu-memory-utilization 0.5 \
#   --vllm-max-model-len 16384 \
#   --variant gen \
#   --train-sample-size 800 \
#   --gradient-sample-size 300 \
#   --gradient-batch-size 2 \
#   --selection-batch-size 4 \
#   --num-edit-regions 3 \
#   --max-region-tokens 1 \
#   --region-expansion-threshold 0.6 \
#   --num-region-candidates 7 \
#   --beam-width 10 \
#   --beam-replacement-mode llm_synthesis \
#   --fluency-lambda 0.5 \
#   --candidate-max-new-tokens 10000 \
#   --synthesis-max-new-tokens 10000 \
#   --synthesis-batch-size 4 \
#   --seed 42
# fi

# -----------------------------------------------------------------------------
# Next paired Qwen matrix. Previous calls remain below as disabled history.

TUNING_GPU="${TUNING_GPU:-1}"

run_qwen_case() {
  # Run one Qwen GradPO configuration.
  local source_prompt="$1"
  local code="$2"
  local spans="$3"
  local tokens_per_span="$4"
  local threshold="$5"
  local candidates="$6"
  local gradient_subset="$7"

  echo "Running ${code} on CUDA_VISIBLE_DEVICES=${TUNING_GPU}"
  CUDA_VISIBLE_DEVICES="${TUNING_GPU}" python -u codes/run_qa_promptopt_gradpo.py \
    --code "${code}" \
    --qa-task openbookqa \
    --qa-mode non_reasoning \
    --train-path data/processed/openbookqa/train.jsonl \
    --validation-path data/processed/openbookqa/validation.jsonl \
    --initial-prompt-file "${source_prompt}" \
    --model Qwen/Qwen3-4B \
    --device cuda:0 \
    --hf-device cuda:0 \
    --target-max-new-tokens 10 \
    --validation-std-penalty 1.0 \
    --output-root outputs/qa_prompt_optimization \
    --overwrite \
    --backend dual \
    --final-evaluation-backend vllm \
    --dual-vllm-gpu-memory-utilization 0.5 \
    --vllm-max-model-len 16384 \
    --variant gen \
    --train-sample-size 800 \
    --gradient-sample-size "${gradient_subset}" \
    --gradient-batch-size 2 \
    --selection-batch-size 4 \
    --num-edit-regions "${spans}" \
    --max-region-tokens "${tokens_per_span}" \
    --region-expansion-threshold "${threshold}" \
    --num-region-candidates "${candidates}" \
    --beam-width 10 \
    --beam-replacement-mode llm_synthesis \
    --fluency-lambda 0.5 \
    --candidate-max-new-tokens 10000 \
    --synthesis-max-new-tokens 10000 \
    --synthesis-batch-size 4 \
    --seed 42
}

# Five matched shapes at G=300.
run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t3_h045_c10_g300_b10_f050 5 3 0.45 10 300
run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s5_t3_h045_c10_g300_b10_f050 5 3 0.45 10 300

run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s4_t3_h045_c10_g300_b10_f050 4 3 0.45 10 300
run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s4_t3_h045_c10_g300_b10_f050 4 3 0.45 10 300

run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s3_t3_h045_c10_g300_b10_f050 3 3 0.45 10 300
run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s3_t3_h045_c10_g300_b10_f050 3 3 0.45 10 300

run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s4_t4_h045_c10_g300_b10_f050 4 4 0.45 10 300
run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s4_t4_h045_c10_g300_b10_f050 4 4 0.45 10 300

run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s4_t3_h050_c10_g300_b10_f050 4 3 0.50 10 300
run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s4_t3_h050_c10_g300_b10_f050 4 3 0.50 10 300

# The same five shapes at G=200 for a matched-configuration comparison.
run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t3_h045_c10_g200_b10_f050 5 3 0.45 10 200
run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s5_t3_h045_c10_g200_b10_f050 5 3 0.45 10 200

run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s4_t3_h045_c10_g200_b10_f050 4 3 0.45 10 200
run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s4_t3_h045_c10_g200_b10_f050 4 3 0.45 10 200

run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s3_t3_h045_c10_g200_b10_f050 3 3 0.45 10 200
run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s3_t3_h045_c10_g200_b10_f050 3 3 0.45 10 200

run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s4_t4_h045_c10_g200_b10_f050 4 4 0.45 10 200
run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s4_t4_h045_c10_g200_b10_f050 4 4 0.45 10 200

run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s4_t3_h050_c10_g200_b10_f050 4 3 0.50 10 200
run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s4_t3_h050_c10_g200_b10_f050 4 3 0.50 10 200

# Completed H={0.30,0.45} matrix; retained without rerunning.
if false; then
run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t3_h045_c7_g300_b10_f050 5 3 0.45 7 300
run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s5_t3_h045_c7_g300_b10_f050 5 3 0.45 7 300
run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t5_h045_c7_g300_b10_f050 5 5 0.45 7 300
run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s5_t5_h045_c7_g300_b10_f050 5 5 0.45 7 300
run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t5_h030_c7_g300_b10_f050 5 5 0.30 7 300
run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s5_t5_h030_c7_g300_b10_f050 5 5 0.30 7 300
run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t5_h045_c10_g300_b10_f050 5 5 0.45 10 300
run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s5_t5_h045_c10_g300_b10_f050 5 5 0.45 10 300
run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t5_h030_c10_g300_b10_f050 5 5 0.30 10 300
run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s5_t5_h030_c10_g300_b10_f050 5 5 0.30 10 300
fi

# Completed previous RPO-10 matrix; commented out and retained as history.
# run_qwen_rpo10_case openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s3_t1_c7_g300_b10_f050 3 1 7 300 10 42
# run_qwen_rpo10_case openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s3_t3_c7_g300_b10_f050 3 3 7 300 10 42
# run_qwen_rpo10_case openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s3_t5_c7_g300_b10_f050 3 5 7 300 10 42
# run_qwen_rpo10_case openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s5_t1_c7_g300_b10_f050 5 1 7 300 10 42
# run_qwen_rpo10_case openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s5_t5_c7_g300_b10_f050 5 5 7 300 10 42
# run_qwen_rpo10_case openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s5_t3_c7_g200_b10_f050 5 3 7 200 10 42
# run_qwen_rpo10_case openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s5_t3_c5_g300_b10_f050 5 3 5 300 10 42
# run_qwen_rpo10_case openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s5_t3_c7_g300_b5_f050 5 3 7 300 5 42
# run_qwen_rpo10_case openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s5_t3_c7_g300_b10_f050_seed43 5 3 7 300 10 43

# # Stage 3A: compare gradient subset 200 against the completed subset-300 winner.

# CUDA_VISIBLE_DEVICES=1 python -u codes/run_qa_promptopt_gradpo.py \
#   --code openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t3_c7_g200_b10_f050 \
#   --qa-task openbookqa \
#   --qa-mode non_reasoning \
#   --train-path data/processed/openbookqa/train.jsonl \
#   --validation-path data/processed/openbookqa/validation.jsonl \
#   --initial-prompt-file "${SOURCE_PROMPT}" \
#   --model Qwen/Qwen3-4B \
#   --device cuda:0 \
#   --hf-device cuda:0 \
#   --target-max-new-tokens 10 \
#   --validation-std-penalty 1.0 \
#   --output-root outputs/qa_prompt_optimization \
#   --overwrite \
#   --backend dual \
#   --final-evaluation-backend vllm \
#   --dual-vllm-gpu-memory-utilization 0.5 \
#   --vllm-max-model-len 16384 \
#   --variant gen \
#   --train-sample-size 800 \
#   --gradient-sample-size 200 \
#   --gradient-batch-size 2 \
#   --selection-batch-size 4 \
#   --num-edit-regions 5 \
#   --max-region-tokens 3 \
#   --region-expansion-threshold 0.6 \
#   --num-region-candidates 7 \
#   --beam-width 10 \
#   --beam-replacement-mode llm_synthesis \
#   --fluency-lambda 0.5 \
#   --candidate-max-new-tokens 10000 \
#   --synthesis-max-new-tokens 10000 \
#   --synthesis-batch-size 4 \
#   --seed 42

# # Stage 3B: compare five candidates/span against the completed seven-candidate winner.

# CUDA_VISIBLE_DEVICES=1 python -u codes/run_qa_promptopt_gradpo.py \
#   --code openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t3_c5_g300_b10_f050 \
#   --qa-task openbookqa \
#   --qa-mode non_reasoning \
#   --train-path data/processed/openbookqa/train.jsonl \
#   --validation-path data/processed/openbookqa/validation.jsonl \
#   --initial-prompt-file "${SOURCE_PROMPT}" \
#   --model Qwen/Qwen3-4B \
#   --device cuda:0 \
#   --hf-device cuda:0 \
#   --target-max-new-tokens 10 \
#   --validation-std-penalty 1.0 \
#   --output-root outputs/qa_prompt_optimization \
#   --overwrite \
#   --backend dual \
#   --final-evaluation-backend vllm \
#   --dual-vllm-gpu-memory-utilization 0.5 \
#   --vllm-max-model-len 16384 \
#   --variant gen \
#   --train-sample-size 800 \
#   --gradient-sample-size 300 \
#   --gradient-batch-size 2 \
#   --selection-batch-size 4 \
#   --num-edit-regions 5 \
#   --max-region-tokens 3 \
#   --region-expansion-threshold 0.6 \
#   --num-region-candidates 5 \
#   --beam-width 10 \
#   --beam-replacement-mode llm_synthesis \
#   --fluency-lambda 0.5 \
#   --candidate-max-new-tokens 10000 \
#   --synthesis-max-new-tokens 10000 \
#   --synthesis-batch-size 4 \
#   --seed 42

# # Stage 3C: compare beam width five against the completed beam-width-10 winner.

# CUDA_VISIBLE_DEVICES=1 python -u codes/run_qa_promptopt_gradpo.py \
#   --code openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t3_c7_g300_b5_f050 \
#   --qa-task openbookqa \
#   --qa-mode non_reasoning \
#   --train-path data/processed/openbookqa/train.jsonl \
#   --validation-path data/processed/openbookqa/validation.jsonl \
#   --initial-prompt-file "${SOURCE_PROMPT}" \
#   --model Qwen/Qwen3-4B \
#   --device cuda:0 \
#   --hf-device cuda:0 \
#   --target-max-new-tokens 10 \
#   --validation-std-penalty 1.0 \
#   --output-root outputs/qa_prompt_optimization \
#   --overwrite \
#   --backend dual \
#   --final-evaluation-backend vllm \
#   --dual-vllm-gpu-memory-utilization 0.5 \
#   --vllm-max-model-len 16384 \
#   --variant gen \
#   --train-sample-size 800 \
#   --gradient-sample-size 300 \
#   --gradient-batch-size 2 \
#   --selection-batch-size 4 \
#   --num-edit-regions 5 \
#   --max-region-tokens 3 \
#   --region-expansion-threshold 0.6 \
#   --num-region-candidates 7 \
#   --beam-width 5 \
#   --beam-replacement-mode llm_synthesis \
#   --fluency-lambda 0.5 \
#   --candidate-max-new-tokens 10000 \
#   --synthesis-max-new-tokens 10000 \
#   --synthesis-batch-size 4 \
#   --seed 42

# # Remaining completed Stage 2 commands; disabled and retained as history.
# if false; then

# CUDA_VISIBLE_DEVICES=1 python -u codes/run_qa_promptopt_gradpo.py \
#   --code openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s3_t3_c7_g300_b10_f050 \
#   --qa-task openbookqa \
#   --qa-mode non_reasoning \
#   --train-path data/processed/openbookqa/train.jsonl \
#   --validation-path data/processed/openbookqa/validation.jsonl \
#   --initial-prompt-file "${SOURCE_PROMPT}" \
#   --model Qwen/Qwen3-4B \
#   --device cuda:0 \
#   --hf-device cuda:0 \
#   --target-max-new-tokens 10 \
#   --validation-std-penalty 1.0 \
#   --output-root outputs/qa_prompt_optimization \
#   --overwrite \
#   --backend dual \
#   --final-evaluation-backend vllm \
#   --dual-vllm-gpu-memory-utilization 0.5 \
#   --vllm-max-model-len 16384 \
#   --variant gen \
#   --train-sample-size 800 \
#   --gradient-sample-size 300 \
#   --gradient-batch-size 2 \
#   --selection-batch-size 4 \
#   --num-edit-regions 3 \
#   --max-region-tokens 3 \
#   --region-expansion-threshold 0.6 \
#   --num-region-candidates 7 \
#   --beam-width 10 \
#   --beam-replacement-mode llm_synthesis \
#   --fluency-lambda 0.5 \
#   --candidate-max-new-tokens 10000 \
#   --synthesis-max-new-tokens 10000 \
#   --synthesis-batch-size 4 \
#   --seed 42

# CUDA_VISIBLE_DEVICES=1 python -u codes/run_qa_promptopt_gradpo.py \
#   --code openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s3_t5_c7_g300_b10_f050 \
#   --qa-task openbookqa \
#   --qa-mode non_reasoning \
#   --train-path data/processed/openbookqa/train.jsonl \
#   --validation-path data/processed/openbookqa/validation.jsonl \
#   --initial-prompt-file "${SOURCE_PROMPT}" \
#   --model Qwen/Qwen3-4B \
#   --device cuda:0 \
#   --hf-device cuda:0 \
#   --target-max-new-tokens 10 \
#   --validation-std-penalty 1.0 \
#   --output-root outputs/qa_prompt_optimization \
#   --overwrite \
#   --backend dual \
#   --final-evaluation-backend vllm \
#   --dual-vllm-gpu-memory-utilization 0.5 \
#   --vllm-max-model-len 16384 \
#   --variant gen \
#   --train-sample-size 800 \
#   --gradient-sample-size 300 \
#   --gradient-batch-size 2 \
#   --selection-batch-size 4 \
#   --num-edit-regions 3 \
#   --max-region-tokens 5 \
#   --region-expansion-threshold 0.6 \
#   --num-region-candidates 7 \
#   --beam-width 10 \
#   --beam-replacement-mode llm_synthesis \
#   --fluency-lambda 0.5 \
#   --candidate-max-new-tokens 10000 \
#   --synthesis-max-new-tokens 10000 \
#   --synthesis-batch-size 4 \
#   --seed 42

# CUDA_VISIBLE_DEVICES=1 python -u codes/run_qa_promptopt_gradpo.py \
#   --code openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t1_c7_g300_b10_f050 \
#   --qa-task openbookqa \
#   --qa-mode non_reasoning \
#   --train-path data/processed/openbookqa/train.jsonl \
#   --validation-path data/processed/openbookqa/validation.jsonl \
#   --initial-prompt-file "${SOURCE_PROMPT}" \
#   --model Qwen/Qwen3-4B \
#   --device cuda:0 \
#   --hf-device cuda:0 \
#   --target-max-new-tokens 10 \
#   --validation-std-penalty 1.0 \
#   --output-root outputs/qa_prompt_optimization \
#   --overwrite \
#   --backend dual \
#   --final-evaluation-backend vllm \
#   --dual-vllm-gpu-memory-utilization 0.5 \
#   --vllm-max-model-len 16384 \
#   --variant gen \
#   --train-sample-size 800 \
#   --gradient-sample-size 300 \
#   --gradient-batch-size 2 \
#   --selection-batch-size 4 \
#   --num-edit-regions 5 \
#   --max-region-tokens 1 \
#   --region-expansion-threshold 0.6 \
#   --num-region-candidates 7 \
#   --beam-width 10 \
#   --beam-replacement-mode llm_synthesis \
#   --fluency-lambda 0.5 \
#   --candidate-max-new-tokens 10000 \
#   --synthesis-max-new-tokens 10000 \
#   --synthesis-batch-size 4 \
#   --seed 42

# CUDA_VISIBLE_DEVICES=1 python -u codes/run_qa_promptopt_gradpo.py \
#   --code openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t3_c7_g300_b10_f050 \
#   --qa-task openbookqa \
#   --qa-mode non_reasoning \
#   --train-path data/processed/openbookqa/train.jsonl \
#   --validation-path data/processed/openbookqa/validation.jsonl \
#   --initial-prompt-file "${SOURCE_PROMPT}" \
#   --model Qwen/Qwen3-4B \
#   --device cuda:0 \
#   --hf-device cuda:0 \
#   --target-max-new-tokens 10 \
#   --validation-std-penalty 1.0 \
#   --output-root outputs/qa_prompt_optimization \
#   --overwrite \
#   --backend dual \
#   --final-evaluation-backend vllm \
#   --dual-vllm-gpu-memory-utilization 0.5 \
#   --vllm-max-model-len 16384 \
#   --variant gen \
#   --train-sample-size 800 \
#   --gradient-sample-size 300 \
#   --gradient-batch-size 2 \
#   --selection-batch-size 4 \
#   --num-edit-regions 5 \
#   --max-region-tokens 3 \
#   --region-expansion-threshold 0.6 \
#   --num-region-candidates 7 \
#   --beam-width 10 \
#   --beam-replacement-mode llm_synthesis \
#   --fluency-lambda 0.5 \
#   --candidate-max-new-tokens 10000 \
#   --synthesis-max-new-tokens 10000 \
#   --synthesis-batch-size 4 \
#   --seed 42
# fi
