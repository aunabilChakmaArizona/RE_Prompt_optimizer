#!/usr/bin/env bash
set -euo pipefail

# Final selected Gemma baseline settings on the two latest RPO source prompts.
# LPO uses five incorrect feedback examples, at most three spans, at most three
# words per span, and five candidates. GreaTer/GreaTer-TG use G=300 and top-u=5.
# Every older experiment block remains below as disabled history.
#
# Background command:
#   nohup bash codes/run_openbookqa_second_stage_gemma_latest_rpo.sh \
#     > codes/nohup_outs/openbookqa_gemma_final_selected_baselines.log 2>&1 &
#
# Override the default physical GPU with, for example:
#   RUN_GPU=1 bash codes/run_openbookqa_second_stage_gemma_latest_rpo.sh

RUN_GPU="${RUN_GPU:-1}"

LATEST_RPO_DIR="outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500_generalmeta"
SOURCES=(
  "rpo5_latest|${LATEST_RPO_DIR}/prompt_experimental_iteration_5_from_actual_iteration_1.txt"
  "rpo10_latest|${LATEST_RPO_DIR}/prompt_experimental_iteration_10_from_actual_iteration_2.txt"
)

for source_spec in "${SOURCES[@]}"; do
  IFS='|' read -r _ source_prompt <<< "${source_spec}"
  [[ -f "${source_prompt}" ]] || {
    echo "Missing first-stage source prompt: ${source_prompt}" >&2
    exit 1
  }
done

run_gradpo() {
  # Run one GradPO variant with one requested span count, threshold, and C.
  local source_name="$1"
  local source_prompt="$2"
  local variant="$3"
  local span_count="$4"
  local threshold="$5"
  local candidates="$6"
  local threshold_code="${threshold/./}"
  local code="openbookqa_non_reasoning_gemma_${source_name}_gradpo_${variant}_s${span_count}_t3_h${threshold_code}_c${candidates}_g200_b5_f050"

  echo "[Gemma second stage] ${code} | GPU=${RUN_GPU}"
  CUDA_VISIBLE_DEVICES="${RUN_GPU}" python -u codes/run_qa_promptopt_gradpo.py \
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
    --variant "${variant}" \
    --train-sample-size 600 \
    --gradient-sample-size 200 \
    --gradient-batch-size 2 \
    --selection-batch-size 4 \
    --num-edit-regions "${span_count}" \
    --max-region-tokens 3 \
    --region-expansion-threshold "${threshold}" \
    --num-region-candidates "${candidates}" \
    --beam-width 5 \
    --beam-replacement-mode llm_synthesis \
    --fluency-lambda 0.5 \
    --candidate-max-new-tokens 10000 \
    --synthesis-max-new-tokens 10000 \
    --synthesis-batch-size 4 \
    --seed 42
}

run_lpo() {
  # Run one LPO configuration with at most five validated rewrites.
  local source_name="$1"
  local source_prompt="$2"
  local feedback_examples="${3:-3}"
  local max_locations="${4:-5}"
  local max_words_per_location="${5:-3}"
  local run_label="${6:-}"
  local code_suffix=""
  if [[ -n "${run_label}" ]]; then
    code_suffix="_${run_label}"
  fi
  local code="openbookqa_non_reasoning_gemma_${source_name}_lpo_c5${code_suffix}_lambda1_vs1500"

  echo "[Gemma second stage] ${code} | GPU=${RUN_GPU}"
  CUDA_VISIBLE_DEVICES="${RUN_GPU}" python -u codes/run_qa_promptopt_lpo.py \
    --code "${code}" \
    --qa-task openbookqa \
    --qa-mode non_reasoning \
    --train-path data/processed/openbookqa/train.jsonl \
    --validation-path data/processed/openbookqa/validation.jsonl \
    --initial-prompt-file "${source_prompt}" \
    --model google/gemma-3-4b-it \
    --optimizer-model google/gemma-3-12b-it \
    --device cuda:0 \
    --optimizer-device cuda:0 \
    --target-max-new-tokens 10 \
    --optimizer-max-new-tokens 10000 \
    --validation-std-penalty 1.0 \
    --output-root outputs/qa_prompt_optimization \
    --overwrite \
    --backend vllm \
    --gpu-memory-utilization 0.6 \
    --vllm-max-model-len 16384 \
    --train-sample-size 600 \
    --feedback-examples "${feedback_examples}" \
    --max-locations "${max_locations}" \
    --max-words-per-location "${max_words_per_location}" \
    --num-candidates 5 \
    --seed 42
}

run_greater() {
  # Run GreaTer or GreaTer-TG with five final validation candidates.
  local source_name="$1"
  local source_prompt="$2"
  local variant="$3"
  local gradient_batch_size="${4:-4}"
  local vllm_gpu_utilization="${5:-0.6}"
  local run_label="${6:-}"
  local gradient_sample_size="${7:-200}"
  local code_suffix=""
  if [[ -n "${run_label}" ]]; then
    code_suffix="_${run_label}"
  fi
  local code="openbookqa_non_reasoning_gemma_${source_name}_${variant}_g${gradient_sample_size}_topu5_same_token_count${code_suffix}_lambda1_vs1500"

  echo "[Gemma second stage] ${code} | GPU=${RUN_GPU}"
  CUDA_VISIBLE_DEVICES="${RUN_GPU}" python -u codes/run_qa_promptopt_greater.py \
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
    --dual-vllm-gpu-memory-utilization "${vllm_gpu_utilization}" \
    --vllm-max-model-len 16384 \
    --variant "${variant}" \
    --train-sample-size 600 \
    --gradient-sample-size "${gradient_sample_size}" \
    --gradient-batch-size "${gradient_batch_size}" \
    --selection-batch-size 8 \
    --proposal-top-k 25 \
    --proposal-example-size 50 \
    --proposal-min-candidates 10 \
    --selection-top-mu 10 \
    --top-u 5 \
    --fluency-lambda 0.2 \
    --region-expansion-threshold 0.6 \
    --seed 42
}

# Completed 18-run block; retained as commented experiment history.
# for source_spec in "${SOURCES[@]}"; do
#   IFS='|' read -r source_name source_prompt <<< "${source_spec}"
#   run_gradpo "${source_name}" "${source_prompt}" gen 3 0.60 7
#   run_gradpo "${source_name}" "${source_prompt}" gen 5 0.60 7
#   for variant in prob gen_random; do
#     run_gradpo "${source_name}" "${source_prompt}" "${variant}" 3 0.60 7
#     run_gradpo "${source_name}" "${source_prompt}" "${variant}" 5 0.60 7
#   done
#   run_lpo "${source_name}" "${source_prompt}"
#   run_greater "${source_name}" "${source_prompt}" greater
#   run_greater "${source_name}" "${source_prompt}" greater_tg
# done

# Completed S=5 H/C follow-up; retained as disabled history.
if false; then
for source_spec in "${SOURCES[@]}"; do
  IFS='|' read -r source_name source_prompt <<< "${source_spec}"
  for tuning_spec in "0.45|7" "0.60|10" "0.45|10"; do
    IFS='|' read -r threshold candidates <<< "${tuning_spec}"
    run_gradpo \
      "${source_name}" \
      "${source_prompt}" \
      gen \
      5 \
      "${threshold}" \
      "${candidates}"
  done
done
fi

# Completed S=3 matrix; retained as disabled history.
if false; then
for source_spec in "${SOURCES[@]}"; do
  IFS='|' read -r source_name source_prompt <<< "${source_spec}"
  for tuning_spec in "0.45|5" "0.45|7" "0.45|10" "0.60|5" "0.60|10"; do
    IFS='|' read -r threshold candidates <<< "${tuning_spec}"
    run_gradpo \
      "${source_name}" \
      "${source_prompt}" \
      gen \
      3 \
      "${threshold}" \
      "${candidates}"
  done
done
fi

# Completed final GreaTer matrix with G=200; retained as disabled history.
# for source_spec in "${SOURCES[@]}"; do
#   IFS='|' read -r source_name source_prompt <<< "${source_spec}"
#   run_greater "${source_name}" "${source_prompt}" greater 2 0.5 gb2_vllm050
#   run_greater "${source_name}" "${source_prompt}" greater_tg 2 0.5 gb2_vllm050
# done

# Active final selected matrix: three methods x two RPO sources = 6 runs.
for source_spec in "${SOURCES[@]}"; do
  IFS='|' read -r source_name source_prompt <<< "${source_spec}"

  run_greater \
    "${source_name}" \
    "${source_prompt}" \
    greater \
    2 \
    0.5 \
    final_sensitivity \
    300

  run_greater \
    "${source_name}" \
    "${source_prompt}" \
    greater_tg \
    2 \
    0.5 \
    final_sensitivity \
    300

  run_lpo \
    "${source_name}" \
    "${source_prompt}" \
    5 \
    3 \
    3 \
    feedback5_s3_t3
done

# Completed last LPO sensitivity check; retained as disabled history.
# for source_spec in "${SOURCES[@]}"; do
#   IFS='|' read -r source_name source_prompt <<< "${source_spec}"
#   run_lpo \
#     "${source_name}" \
#     "${source_prompt}" \
#     3 \
#     3 \
#     3 \
#     feedback3_s3_t3_final_sensitivity
# done
