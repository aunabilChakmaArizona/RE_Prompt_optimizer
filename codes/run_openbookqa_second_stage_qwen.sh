#!/usr/bin/env bash
set -euo pipefail

# ACTIVE (2026-09-15): remaining EvoPrompt and ETGPO source prompts only.
# The completed RPO command archive remains below the explicit `exit 0`.
# Qwen EvoPrompt-5 is actual iteration 1; EvoPrompt-10 is actual iteration 4.
# This script runs 3 sources x 6 refiners = 18 experiments.
RUN_GPU="${RUN_GPU:-2}"

ACTIVE_SOURCES=(
  "evoprompt5|outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_experimental_iteration_5_from_actual_iteration_1.txt"
  "evoprompt10|outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_experimental_iteration_10_from_actual_iteration_4.txt"
  "etgpo1|outputs/qa_prompt_optimization/non_reasoning/etgpo/openbookqa_non_reasoning_qwen_etgpo_qwen14opt_lambda1_fixed_vs1500/final_prompt.txt"
)

for source_spec in "${ACTIVE_SOURCES[@]}"; do
  IFS='|' read -r _ source_prompt <<< "${source_spec}"
  [[ -f "${source_prompt}" ]] || {
    echo "Missing first-stage prompt: ${source_prompt}"
    exit 1
  }
done

run_active_lpo() {
  # Run LPO with the final ten-candidate OpenBookQA configuration.
  local source_name="$1"
  local source_prompt="$2"
  CUDA_VISIBLE_DEVICES="${RUN_GPU}" python -u codes/run_qa_promptopt_lpo.py \
    --code "openbookqa_non_reasoning_qwen_${source_name}_lpo_c10_lambda1_vs1500" \
    --qa-task openbookqa \
    --qa-mode non_reasoning \
    --train-path data/processed/openbookqa/train.jsonl \
    --validation-path data/processed/openbookqa/validation.jsonl \
    --initial-prompt-file "${source_prompt}" \
    --model Qwen/Qwen3-4B \
    --device cuda:0 \
    --target-max-new-tokens 10 \
    --validation-std-penalty 1.0 \
    --output-root outputs/qa_prompt_optimization \
    --overwrite \
    --optimizer-model Qwen/Qwen3-14B \
    --optimizer-device cuda:0 \
    --backend vllm \
    --gpu-memory-utilization 0.90 \
    --vllm-max-model-len 16384 \
    --optimizer-max-new-tokens 10000 \
    --train-sample-size 512 \
    --feedback-examples 3 \
    --max-locations 5 \
    --max-words-per-location 3 \
    --num-candidates 10 \
    --seed 42
}

run_active_greater() {
  # Run one corrected GreaTer variant with top-u ten.
  local source_name="$1"
  local source_prompt="$2"
  local variant="$3"
  CUDA_VISIBLE_DEVICES="${RUN_GPU}" python -u codes/run_qa_promptopt_greater.py \
    --code "openbookqa_non_reasoning_qwen_${source_name}_${variant}_g200_topu10_same_token_count_lambda1_vs1500" \
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
    --variant "${variant}" \
    --train-sample-size 800 \
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

run_active_gradpo() {
  # Run one GradPO variant with the selected shared Qwen configuration.
  local source_name="$1"
  local source_prompt="$2"
  local variant="$3"
  CUDA_VISIBLE_DEVICES="${RUN_GPU}" python -u codes/run_qa_promptopt_gradpo.py \
    --code "openbookqa_non_reasoning_qwen_${source_name}_gradpo_${variant}_bestcfg_lambda1_vs1500" \
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
    --variant "${variant}" \
    --train-sample-size 800 \
    --gradient-sample-size 200 \
    --gradient-batch-size 2 \
    --selection-batch-size 4 \
    --num-edit-regions 3 \
    --max-region-tokens 3 \
    --region-expansion-threshold 0.45 \
    --num-region-candidates 10 \
    --beam-width 10 \
    --beam-replacement-mode llm_synthesis \
    --fluency-lambda 0.5 \
    --candidate-max-new-tokens 10000 \
    --synthesis-max-new-tokens 10000 \
    --synthesis-batch-size 4 \
    --seed 42
}

for source_spec in "${ACTIVE_SOURCES[@]}"; do
  IFS='|' read -r source_name source_prompt <<< "${source_spec}"
  run_active_gradpo "${source_name}" "${source_prompt}" gen
  run_active_gradpo "${source_name}" "${source_prompt}" prob
  run_active_gradpo "${source_name}" "${source_prompt}" gen_random
  run_active_lpo "${source_name}" "${source_prompt}"
  run_active_greater "${source_name}" "${source_prompt}" greater
  run_active_greater "${source_name}" "${source_prompt}" greater_tg
done

exit 0

# ARCHIVE ONLY: older RPO/EvoPrompt/ETGPO commands below are disabled by the
# explicit exit above. They are retained to preserve the experiment history.

# OpenBookQA non-reasoning: five source prompts x six refiners.
# Test evaluation remains disabled; selection uses 3 x 500 validation.
# nohup bash codes/run_openbookqa_second_stage_qwen.sh > codes/nohup_outs/openbookqa_second_stage_qwen.log 2>&1 &

[[ -f outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_5.txt ]] || { echo 'Missing first-stage prompt: outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_5.txt'; exit 1; }
[[ -f outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_10.txt ]] || { echo 'Missing first-stage prompt: outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_10.txt'; exit 1; }
[[ -f outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_5.txt ]] || { echo 'Missing first-stage prompt: outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_5.txt'; exit 1; }
[[ -f outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_10.txt ]] || { echo 'Missing first-stage prompt: outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_10.txt'; exit 1; }
[[ -f outputs/qa_prompt_optimization/non_reasoning/etgpo/openbookqa_non_reasoning_qwen_etgpo_qwen14opt_lambda1_fixed_vs1500/final_prompt.txt ]] || { echo 'Missing first-stage prompt: outputs/qa_prompt_optimization/non_reasoning/etgpo/openbookqa_non_reasoning_qwen_etgpo_qwen14opt_lambda1_fixed_vs1500/final_prompt.txt'; exit 1; }

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_lpo.py \
  --code openbookqa_non_reasoning_qwen_rpo5_lpo_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --optimizer-model Qwen/Qwen3-14B \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --train-sample-size 512 \
  --feedback-examples 3 \
  --max-locations 5 \
  --max-words-per-location 3 \
  --num-candidates 5

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code openbookqa_non_reasoning_qwen_rpo5_greater_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code openbookqa_non_reasoning_qwen_rpo5_greater_tg_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater_tg \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_rpo5_gradpo_gen_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
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
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_rpo5_gradpo_prob_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant prob \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_rpo5_gradpo_gen_random_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen_random \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_lpo.py \
  --code openbookqa_non_reasoning_qwen_rpo10_lpo_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --optimizer-model Qwen/Qwen3-14B \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --train-sample-size 512 \
  --feedback-examples 3 \
  --max-locations 5 \
  --max-words-per-location 3 \
  --num-candidates 5

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code openbookqa_non_reasoning_qwen_rpo10_greater_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code openbookqa_non_reasoning_qwen_rpo10_greater_tg_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater_tg \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_rpo10_gradpo_gen_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
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
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_rpo10_gradpo_prob_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant prob \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_rpo10_gradpo_gen_random_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen_random \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_lpo.py \
  --code openbookqa_non_reasoning_qwen_evoprompt5_lpo_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --optimizer-model Qwen/Qwen3-14B \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --train-sample-size 512 \
  --feedback-examples 3 \
  --max-locations 5 \
  --max-words-per-location 3 \
  --num-candidates 5

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code openbookqa_non_reasoning_qwen_evoprompt5_greater_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code openbookqa_non_reasoning_qwen_evoprompt5_greater_tg_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater_tg \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_evoprompt5_gradpo_gen_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
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
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_evoprompt5_gradpo_prob_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant prob \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_evoprompt5_gradpo_gen_random_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen_random \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_lpo.py \
  --code openbookqa_non_reasoning_qwen_evoprompt10_lpo_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --optimizer-model Qwen/Qwen3-14B \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --train-sample-size 512 \
  --feedback-examples 3 \
  --max-locations 5 \
  --max-words-per-location 3 \
  --num-candidates 5

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code openbookqa_non_reasoning_qwen_evoprompt10_greater_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code openbookqa_non_reasoning_qwen_evoprompt10_greater_tg_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater_tg \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_evoprompt10_gradpo_gen_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
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
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_evoprompt10_gradpo_prob_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant prob \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_evoprompt10_gradpo_gen_random_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen_random \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_lpo.py \
  --code openbookqa_non_reasoning_qwen_etgpo1_lpo_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/etgpo/openbookqa_non_reasoning_qwen_etgpo_qwen14opt_lambda1_fixed_vs1500/final_prompt.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --optimizer-model Qwen/Qwen3-14B \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --train-sample-size 512 \
  --feedback-examples 3 \
  --max-locations 5 \
  --max-words-per-location 3 \
  --num-candidates 5

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code openbookqa_non_reasoning_qwen_etgpo1_greater_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/etgpo/openbookqa_non_reasoning_qwen_etgpo_qwen14opt_lambda1_fixed_vs1500/final_prompt.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code openbookqa_non_reasoning_qwen_etgpo1_greater_tg_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/etgpo/openbookqa_non_reasoning_qwen_etgpo_qwen14opt_lambda1_fixed_vs1500/final_prompt.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater_tg \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_etgpo1_gradpo_gen_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/etgpo/openbookqa_non_reasoning_qwen_etgpo_qwen14opt_lambda1_fixed_vs1500/final_prompt.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
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
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_etgpo1_gradpo_prob_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/etgpo/openbookqa_non_reasoning_qwen_etgpo_qwen14opt_lambda1_fixed_vs1500/final_prompt.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant prob \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_non_reasoning_qwen_etgpo1_gradpo_gen_random_lambda1_vs1500 \
  --qa-task openbookqa \
  --qa-mode non_reasoning \
  --train-path data/processed/openbookqa/train.jsonl \
  --validation-path data/processed/openbookqa/validation.jsonl \
  --initial-prompt-file outputs/qa_prompt_optimization/non_reasoning/etgpo/openbookqa_non_reasoning_qwen_etgpo_qwen14opt_lambda1_fixed_vs1500/final_prompt.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 10 \
  --validation-std-penalty 1.0 \
  --output-root outputs/qa_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen_random \
  --train-sample-size 800 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0
