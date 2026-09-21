#!/usr/bin/env bash
set -euo pipefail

# OpenBookQA GradPO-Gen tuning for the final Qwen RPO-5 and RPO-10 sources.
# The default phase tests H=0.30 with S={2,3}, C={5,7}, and G=300 on both
# latest source prompts: 8 runs. If needed, PHASE=h030_g200 runs the same
# 8-case matrix with G=200. T=3, B=5, and F=0.5 are fixed.
# Completed phases s3 and s2 remain selectable only for reproducibility.
# RUN_GPU=1 PHASE=h030_g300 nohup bash codes/run_openbookqa_gradpo_tuning_qwen.sh > codes/nohup_outs/openbookqa_qwen_h030_g300.log 2>&1 &

LATEST_RPO_DIR="outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1_vs1500_generalmeta_retry_gpu045"
# Experimental RPO-5 retains the prompt generated at actual iteration 1.
SOURCE_PROMPT="${LATEST_RPO_DIR}/prompt_iteration_5.txt"
# Experimental RPO-10 retains the prompt generated at actual iteration 9.
RPO10_SOURCE_PROMPT="${LATEST_RPO_DIR}/prompt_iteration_10.txt"

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

RUN_GPU="${RUN_GPU:-1}"
PHASE="${PHASE:-h030_g300}"

run_qwen_case() {
  # Run one Qwen GradPO configuration.
  local source_prompt="$1"
  local code="$2"
  local spans="$3"
  local tokens_per_span="$4"
  local threshold="$5"
  local candidates="$6"
  local gradient_subset="$7"
  local beam_width="${8:-10}"

  echo "Running ${code} on CUDA_VISIBLE_DEVICES=${RUN_GPU}"
  CUDA_VISIBLE_DEVICES="${RUN_GPU}" python -u codes/run_qa_promptopt_gradpo.py \
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
    --beam-width "${beam_width}" \
    --beam-replacement-mode llm_synthesis \
    --fluency-lambda 0.5 \
    --candidate-max-new-tokens 10000 \
    --synthesis-max-new-tokens 10000 \
    --synthesis-batch-size 4 \
    --seed 42
}

# Completed paired Qwen matrix; commented out and retained as experiment history.
# Five matched shapes at G=300.
# run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t3_h045_c10_g300_b10_f050 5 3 0.45 10 300
# run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s5_t3_h045_c10_g300_b10_f050 5 3 0.45 10 300

# run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s4_t3_h045_c10_g300_b10_f050 4 3 0.45 10 300
# run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s4_t3_h045_c10_g300_b10_f050 4 3 0.45 10 300

# run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s3_t3_h045_c10_g300_b10_f050 3 3 0.45 10 300
# run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s3_t3_h045_c10_g300_b10_f050 3 3 0.45 10 300

# run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s4_t4_h045_c10_g300_b10_f050 4 4 0.45 10 300
# run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s4_t4_h045_c10_g300_b10_f050 4 4 0.45 10 300

# run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s4_t3_h050_c10_g300_b10_f050 4 3 0.50 10 300
# run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s4_t3_h050_c10_g300_b10_f050 4 3 0.50 10 300

# The same five shapes at G=200 for a matched-configuration comparison.
# run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s5_t3_h045_c10_g200_b10_f050 5 3 0.45 10 200
# run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s5_t3_h045_c10_g200_b10_f050 5 3 0.45 10 200

# run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s4_t3_h045_c10_g200_b10_f050 4 3 0.45 10 200
# run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s4_t3_h045_c10_g200_b10_f050 4 3 0.45 10 200

# run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s3_t3_h045_c10_g200_b10_f050 3 3 0.45 10 200
# run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s3_t3_h045_c10_g200_b10_f050 3 3 0.45 10 200

# run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s4_t4_h045_c10_g200_b10_f050 4 4 0.45 10 200
# run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s4_t4_h045_c10_g200_b10_f050 4 4 0.45 10 200

# run_qwen_case "${SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo5_gradpo_tune_s4_t3_h050_c10_g200_b10_f050 4 3 0.50 10 200
# run_qwen_case "${RPO10_SOURCE_PROMPT}" openbookqa_non_reasoning_qwen_rpo10_gradpo_tune_s4_t3_h050_c10_g200_b10_f050 4 3 0.50 10 200

# Completed latest-source S={3,5}, G=200 matrix; retained as disabled history.
if false; then
  for source_spec in \
    "rpo5_latest|${SOURCE_PROMPT}" \
    "rpo10_latest|${RPO10_SOURCE_PROMPT}"
  do
    IFS='|' read -r source_name source_prompt <<< "${source_spec}"
    for spans in 3 5; do
      for threshold in 0.45 0.60; do
        threshold_code="${threshold/./}"
        for candidates in 7 10; do
          code="openbookqa_non_reasoning_qwen_${source_name}_gradpo_gen_s${spans}_t3_h${threshold_code}_c${candidates}_g200_b5_f050"
          run_qwen_case "${source_prompt}" "${code}" "${spans}" 3 \
            "${threshold}" "${candidates}" 200 5
        done
      done
    done
  done
fi

# Active threshold-0.30 tuning. Earlier s3/s2 phases remain reproducible.
case "${PHASE}" in
  h030_g300)
    spans_values=(2 3)
    gradient_subsets=(300)
    thresholds=(0.30)
    ;;
  h030_g200)
    spans_values=(2 3)
    gradient_subsets=(200)
    thresholds=(0.30)
    ;;
  s3)
    spans_values=(3)
    gradient_subsets=(300)
    thresholds=(0.45 0.60)
    ;;
  s2)
    spans_values=(2)
    gradient_subsets=(200 300)
    thresholds=(0.45 0.60)
    ;;
  *)
    echo "Unknown PHASE=${PHASE}. Use h030_g300, h030_g200, s3, or s2." >&2
    exit 1
    ;;
esac

echo "Running Qwen tuning phase ${PHASE} on GPU ${RUN_GPU}"
for source_spec in \
  "rpo5_latest|${SOURCE_PROMPT}" \
  "rpo10_latest|${RPO10_SOURCE_PROMPT}"
do
  IFS='|' read -r source_name source_prompt <<< "${source_spec}"
  for spans in "${spans_values[@]}"; do
    for gradient_subset in "${gradient_subsets[@]}"; do
      for threshold in "${thresholds[@]}"; do
        threshold_code="${threshold/./}"
        for candidates in 5 7; do
          code="openbookqa_non_reasoning_qwen_${source_name}_gradpo_gen_s${spans}_t3_h${threshold_code}_c${candidates}_g${gradient_subset}_b5_f050"
          run_qwen_case \
            "${source_prompt}" \
            "${code}" \
            "${spans}" \
            3 \
            "${threshold}" \
            "${candidates}" \
            "${gradient_subset}" \
            5
        done
      done
    done
  done
done

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
