#!/usr/bin/env bash
set -euo pipefail

# The original 20-run OpenBookQA RPO second-stage matrix is complete.
# The only active commands below rerun GreaTer and GreaTer-TG after replacing
# our overly strict exact-ID retokenization filter with GreaTer's original
# same-total-token-count filter. This gives eight active runs:
#   2 models x 2 RPO sources x 2 GreaTer variants.
# Test evaluation remains disabled; selection uses 3 x 500 validation examples.
#
# Background command:
#   nohup bash codes/run_openbookqa_remaining_second_stage.sh \
#     > codes/nohup_outs/openbookqa_remaining_second_stage.log 2>&1 &
#
# The models run sequentially and default to physical GPU 1. Override it with:
#   RUN_GPU=3 bash codes/run_openbookqa_remaining_second_stage.sh

RUN_GPU="${RUN_GPU:-1}"

QWEN_SOURCES=(
  "rpo5|outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_5.txt"
  "rpo10|outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_10.txt"
)

GEMMA_SOURCES=(
  "rpo5|outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_gemma_rpo_gemma12opt_lambda1/prompt_iteration_3.txt"
  "rpo10|outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_gemma_rpo_gemma12opt_lambda1/prompt_iteration_10.txt"
)

for source_spec in "${QWEN_SOURCES[@]}" "${GEMMA_SOURCES[@]}"; do
  IFS='|' read -r _ source_prompt <<< "${source_spec}"
  [[ -f "${source_prompt}" ]] || {
    echo "Missing first-stage prompt: ${source_prompt}"
    exit 1
  }
done

set_model_family() {
  # Set model-specific runtime and tuned GradPO parameters.
  local family="$1"
  if [[ "${family}" == "qwen" ]]; then
    TARGET_MODEL="Qwen/Qwen3-4B"
    OPTIMIZER_MODEL="Qwen/Qwen3-14B"
    TRAIN_POOL=800
    GRADPO_SPANS=3
    GRADPO_TOKENS=3
    GRADPO_THRESHOLD=0.45
    GRADPO_CANDIDATES=10
    VLLM_MEMORY=0.5
    VLLM_MAX_LEN=16384
    VLLM_IMAGE_ARGS=()
  elif [[ "${family}" == "gemma" ]]; then
    TARGET_MODEL="google/gemma-3-4b-it"
    OPTIMIZER_MODEL="google/gemma-3-12b-it"
    TRAIN_POOL=600
    GRADPO_SPANS=5
    GRADPO_TOKENS=3
    GRADPO_THRESHOLD=0.60
    GRADPO_CANDIDATES=7
    VLLM_MEMORY=0.6
    VLLM_MAX_LEN=16384
    VLLM_IMAGE_ARGS=(--vllm-disable-images)
  else
    echo "Unsupported model family: ${family}" >&2
    exit 1
  fi
}

run_gradpo() {
  # Run GradPO-Prob or GradPO-Gen-Random with the selected shared settings.
  local family="$1"
  local source_name="$2"
  local source_prompt="$3"
  local variant="$4"
  set_model_family "${family}"
  local code="openbookqa_non_reasoning_${family}_${source_name}_gradpo_${variant}_bestcfg_lambda1_vs1500"

  echo "Running ${code} on CUDA_VISIBLE_DEVICES=${RUN_GPU}"
  CUDA_VISIBLE_DEVICES="${RUN_GPU}" python -u codes/run_qa_promptopt_gradpo.py \
    --code "${code}" \
    --qa-task openbookqa \
    --qa-mode non_reasoning \
    --train-path data/processed/openbookqa/train.jsonl \
    --validation-path data/processed/openbookqa/validation.jsonl \
    --initial-prompt-file "${source_prompt}" \
    --model "${TARGET_MODEL}" \
    --device cuda:0 \
    --hf-device cuda:0 \
    --target-max-new-tokens 10 \
    --validation-std-penalty 1.0 \
    --output-root outputs/qa_prompt_optimization \
    --overwrite \
    --backend dual \
    --final-evaluation-backend vllm \
    --dual-vllm-gpu-memory-utilization "${VLLM_MEMORY}" \
    --vllm-max-model-len "${VLLM_MAX_LEN}" \
    "${VLLM_IMAGE_ARGS[@]}" \
    --variant "${variant}" \
    --train-sample-size "${TRAIN_POOL}" \
    --gradient-sample-size 200 \
    --gradient-batch-size 2 \
    --selection-batch-size 4 \
    --num-edit-regions "${GRADPO_SPANS}" \
    --max-region-tokens "${GRADPO_TOKENS}" \
    --region-expansion-threshold "${GRADPO_THRESHOLD}" \
    --num-region-candidates "${GRADPO_CANDIDATES}" \
    --beam-width 10 \
    --beam-replacement-mode llm_synthesis \
    --fluency-lambda 0.5 \
    --candidate-max-new-tokens 10000 \
    --synthesis-max-new-tokens 10000 \
    --synthesis-batch-size 4 \
    --seed 42
}

run_lpo() {
  # Generate one batch of ten LPO rewrites and validate every distinct result.
  local family="$1"
  local source_name="$2"
  local source_prompt="$3"
  set_model_family "${family}"
  local code="openbookqa_non_reasoning_${family}_${source_name}_lpo_c10_lambda1_vs1500"

  echo "Running ${code} on CUDA_VISIBLE_DEVICES=${RUN_GPU}"
  CUDA_VISIBLE_DEVICES="${RUN_GPU}" python -u codes/run_qa_promptopt_lpo.py \
    --code "${code}" \
    --qa-task openbookqa \
    --qa-mode non_reasoning \
    --train-path data/processed/openbookqa/train.jsonl \
    --validation-path data/processed/openbookqa/validation.jsonl \
    --initial-prompt-file "${source_prompt}" \
    --model "${TARGET_MODEL}" \
    --device cuda:0 \
    --target-max-new-tokens 10 \
    --validation-std-penalty 1.0 \
    --output-root outputs/qa_prompt_optimization \
    --overwrite \
    --optimizer-model "${OPTIMIZER_MODEL}" \
    --optimizer-device cuda:0 \
    --backend vllm \
    --gpu-memory-utilization 0.90 \
    --vllm-max-model-len 16384 \
    "${VLLM_IMAGE_ARGS[@]}" \
    --optimizer-max-new-tokens 10000 \
    --train-sample-size 512 \
    --feedback-examples 3 \
    --max-locations 5 \
    --max-words-per-location 3 \
    --num-candidates 10 \
    --seed 42
}

run_greater() {
  # Run sequential GreaTer or top-gradient GreaTer-TG with top-u ten.
  local family="$1"
  local source_name="$2"
  local source_prompt="$3"
  local variant="$4"
  set_model_family "${family}"
  local code="openbookqa_non_reasoning_${family}_${source_name}_${variant}_g200_topu10_same_token_count_lambda1_vs1500"

  echo "Running ${code} on CUDA_VISIBLE_DEVICES=${RUN_GPU}"
  CUDA_VISIBLE_DEVICES="${RUN_GPU}" python -u codes/run_qa_promptopt_greater.py \
    --code "${code}" \
    --qa-task openbookqa \
    --qa-mode non_reasoning \
    --train-path data/processed/openbookqa/train.jsonl \
    --validation-path data/processed/openbookqa/validation.jsonl \
    --initial-prompt-file "${source_prompt}" \
    --model "${TARGET_MODEL}" \
    --device cuda:0 \
    --hf-device cuda:0 \
    --target-max-new-tokens 10 \
    --validation-std-penalty 1.0 \
    --output-root outputs/qa_prompt_optimization \
    --overwrite \
    --backend dual \
    --final-evaluation-backend vllm \
    --dual-vllm-gpu-memory-utilization "${VLLM_MEMORY}" \
    --vllm-max-model-len "${VLLM_MAX_LEN}" \
    "${VLLM_IMAGE_ARGS[@]}" \
    --variant "${variant}" \
    --train-sample-size "${TRAIN_POOL}" \
    --gradient-sample-size 200 \
    --gradient-batch-size 4 \
    --selection-batch-size 8 \
    --proposal-top-k 25 \
    --proposal-example-size 50 \
    --proposal-min-candidates 10 \
    --selection-top-mu 10 \
    --top-u 10 \
    --fluency-lambda 0.2 \
    --region-expansion-threshold 0.6 \
    --seed 42
}

# Completed and disabled: GradPO-Prob, GradPO-Gen-Random, and LPO.
# Their function definitions remain above so the exact configurations stay
# documented in this experiment script.

# Active: rerun both corrected GreaTer variants.
for family in qwen gemma; do
  if [[ "${family}" == "qwen" ]]; then
    sources=("${QWEN_SOURCES[@]}")
  else
    sources=("${GEMMA_SOURCES[@]}")
  fi
  for source_spec in "${sources[@]}"; do
    IFS='|' read -r source_name source_prompt <<< "${source_spec}"
    run_greater "${family}" "${source_name}" "${source_prompt}" greater
    run_greater "${family}" "${source_name}" "${source_prompt}" greater_tg
  done
done
