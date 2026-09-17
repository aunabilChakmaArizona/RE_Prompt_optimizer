#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. No final test evaluation is enabled.
# Default: attempts 1-4 only. Later phases require selected values explicitly.
# nohup bash codes/run_math_gradpo_tuning_qwen.sh > codes/nohup_outs/math_gradpo_tuning_qwen_width.log 2>&1 &
# MATH_TUNING_STAGE=expansion MATH_TUNING_T=SELECTED_T nohup bash codes/run_math_gradpo_tuning_qwen.sh > codes/nohup_outs/math_gradpo_tuning_qwen_expansion.log 2>&1 &
# MATH_TUNING_STAGE=count MATH_TUNING_T=SELECTED_T MATH_TUNING_H=SELECTED_H nohup bash codes/run_math_gradpo_tuning_qwen.sh > codes/nohup_outs/math_gradpo_tuning_qwen_count.log 2>&1 &
# Optional GPU override: MATH_TUNING_GPU=1 (default physical GPU 3).
# Both HF and vLLM use logical cuda:0; vLLM reserves 40%.
# Cache sharing uses greedy source responses. Validation remains sampled.
# RPO5/RPO10 are assigned source slots; actual iterations are 2/5.

TUNING_GPU="${MATH_TUNING_GPU:-3}"
TUNING_STAGE="${MATH_TUNING_STAGE:-width}"

case "$TUNING_STAGE" in
  width|expansion|count) ;;
  *) printf 'Invalid MATH_TUNING_STAGE: %s\n' "$TUNING_STAGE" >&2; exit 2 ;;
esac

if [[ "$TUNING_STAGE" != "width" ]]; then
  : "${MATH_TUNING_T:?Set MATH_TUNING_T to 3 or 5 after inspecting both source results}"
  case "$MATH_TUNING_T" in
    3|5) ;;
    *) printf 'MATH_TUNING_T must be 3 or 5.\n' >&2; exit 2 ;;
  esac
fi

if [[ "$TUNING_STAGE" == "count" ]]; then
  : "${MATH_TUNING_H:?Set MATH_TUNING_H to the selected 0.30 or 0.45}"
  case "$MATH_TUNING_H" in
    0.3|0.30) TUNING_H=0.30; H_TAG=030 ;;
    0.45) TUNING_H=0.45; H_TAG=045 ;;
    *) printf 'MATH_TUNING_H must be 0.30 or 0.45.\n' >&2; exit 2 ;;
  esac
fi

if [[ "$TUNING_STAGE" == "width" ]]; then
# Attempts 1-2: exact selected OpenBookQA Qwen search settings.
CUDA_VISIBLE_DEVICES="$TUNING_GPU" python -u codes/run_qa_promptopt_gradpo.py \
  --code "math500_reasoning_qwen_rpo5_gradpo_tune_s3_t3_h045_c10_g200_b10_f050_pool1600_vs900" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_5_from_actual_iteration_2.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.40" \
  --dual-vllm-gpu-memory-utilization "0.40" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "8192" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "1" \
  --num-edit-regions "3" \
  --max-region-tokens "3" \
  --region-expansion-threshold "0.45" \
  --num-region-candidates "10" \
  --beam-width "10" \
  --fluency-lambda "0.5" \
  --beam-replacement-mode "llm_synthesis" \
  --candidate-max-new-tokens "10000" \
  --synthesis-max-new-tokens "10000" \
  --synthesis-batch-size "4" \
  --gradient-cache-root "outputs/shared_gradient_cache" \
  --seed "42" \
  --output-root "outputs/math_prompt_optimization" \
  --overwrite

CUDA_VISIBLE_DEVICES="$TUNING_GPU" python -u codes/run_qa_promptopt_gradpo.py \
  --code "math500_reasoning_qwen_rpo10_gradpo_tune_s3_t3_h045_c10_g200_b10_f050_pool1600_vs900" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_10_from_actual_iteration_5.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.40" \
  --dual-vllm-gpu-memory-utilization "0.40" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "8192" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "1" \
  --num-edit-regions "3" \
  --max-region-tokens "3" \
  --region-expansion-threshold "0.45" \
  --num-region-candidates "10" \
  --beam-width "10" \
  --fluency-lambda "0.5" \
  --beam-replacement-mode "llm_synthesis" \
  --candidate-max-new-tokens "10000" \
  --synthesis-max-new-tokens "10000" \
  --synthesis-batch-size "4" \
  --gradient-cache-root "outputs/shared_gradient_cache" \
  --seed "42" \
  --output-root "outputs/math_prompt_optimization" \
  --overwrite

# Attempts 3-4: change only the maximum source-region width to five.
CUDA_VISIBLE_DEVICES="$TUNING_GPU" python -u codes/run_qa_promptopt_gradpo.py \
  --code "math500_reasoning_qwen_rpo5_gradpo_tune_s3_t5_h045_c10_g200_b10_f050_pool1600_vs900" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_5_from_actual_iteration_2.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.40" \
  --dual-vllm-gpu-memory-utilization "0.40" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "8192" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "1" \
  --num-edit-regions "3" \
  --max-region-tokens "5" \
  --region-expansion-threshold "0.45" \
  --num-region-candidates "10" \
  --beam-width "10" \
  --fluency-lambda "0.5" \
  --beam-replacement-mode "llm_synthesis" \
  --candidate-max-new-tokens "10000" \
  --synthesis-max-new-tokens "10000" \
  --synthesis-batch-size "4" \
  --gradient-cache-root "outputs/shared_gradient_cache" \
  --seed "42" \
  --output-root "outputs/math_prompt_optimization" \
  --overwrite

CUDA_VISIBLE_DEVICES="$TUNING_GPU" python -u codes/run_qa_promptopt_gradpo.py \
  --code "math500_reasoning_qwen_rpo10_gradpo_tune_s3_t5_h045_c10_g200_b10_f050_pool1600_vs900" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_10_from_actual_iteration_5.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.40" \
  --dual-vllm-gpu-memory-utilization "0.40" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "8192" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "1" \
  --num-edit-regions "3" \
  --max-region-tokens "5" \
  --region-expansion-threshold "0.45" \
  --num-region-candidates "10" \
  --beam-width "10" \
  --fluency-lambda "0.5" \
  --beam-replacement-mode "llm_synthesis" \
  --candidate-max-new-tokens "10000" \
  --synthesis-max-new-tokens "10000" \
  --synthesis-batch-size "4" \
  --gradient-cache-root "outputs/shared_gradient_cache" \
  --seed "42" \
  --output-root "outputs/math_prompt_optimization" \
  --overwrite
fi

if [[ "$TUNING_STAGE" == "expansion" ]]; then
# Attempts 5-6: chosen shared T, lower neighboring-gradient threshold.
CUDA_VISIBLE_DEVICES="$TUNING_GPU" python -u codes/run_qa_promptopt_gradpo.py \
  --code "math500_reasoning_qwen_rpo5_gradpo_tune_s3_t${MATH_TUNING_T}_h030_c10_g200_b10_f050_pool1600_vs900" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_5_from_actual_iteration_2.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.40" \
  --dual-vllm-gpu-memory-utilization "0.40" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "8192" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "1" \
  --num-edit-regions "3" \
  --max-region-tokens "$MATH_TUNING_T" \
  --region-expansion-threshold "0.30" \
  --num-region-candidates "10" \
  --beam-width "10" \
  --fluency-lambda "0.5" \
  --beam-replacement-mode "llm_synthesis" \
  --candidate-max-new-tokens "10000" \
  --synthesis-max-new-tokens "10000" \
  --synthesis-batch-size "4" \
  --gradient-cache-root "outputs/shared_gradient_cache" \
  --seed "42" \
  --output-root "outputs/math_prompt_optimization" \
  --overwrite

CUDA_VISIBLE_DEVICES="$TUNING_GPU" python -u codes/run_qa_promptopt_gradpo.py \
  --code "math500_reasoning_qwen_rpo10_gradpo_tune_s3_t${MATH_TUNING_T}_h030_c10_g200_b10_f050_pool1600_vs900" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_10_from_actual_iteration_5.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.40" \
  --dual-vllm-gpu-memory-utilization "0.40" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "8192" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "1" \
  --num-edit-regions "3" \
  --max-region-tokens "$MATH_TUNING_T" \
  --region-expansion-threshold "0.30" \
  --num-region-candidates "10" \
  --beam-width "10" \
  --fluency-lambda "0.5" \
  --beam-replacement-mode "llm_synthesis" \
  --candidate-max-new-tokens "10000" \
  --synthesis-max-new-tokens "10000" \
  --synthesis-batch-size "4" \
  --gradient-cache-root "outputs/shared_gradient_cache" \
  --seed "42" \
  --output-root "outputs/math_prompt_optimization" \
  --overwrite
fi

if [[ "$TUNING_STAGE" == "count" ]]; then
# Attempts 7-8: five regions using the shared selected T and H.
CUDA_VISIBLE_DEVICES="$TUNING_GPU" python -u codes/run_qa_promptopt_gradpo.py \
  --code "math500_reasoning_qwen_rpo5_gradpo_tune_s5_t${MATH_TUNING_T}_h${H_TAG}_c10_g200_b10_f050_pool1600_vs900" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_5_from_actual_iteration_2.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.40" \
  --dual-vllm-gpu-memory-utilization "0.40" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "8192" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "1" \
  --num-edit-regions "5" \
  --max-region-tokens "$MATH_TUNING_T" \
  --region-expansion-threshold "$TUNING_H" \
  --num-region-candidates "10" \
  --beam-width "10" \
  --fluency-lambda "0.5" \
  --beam-replacement-mode "llm_synthesis" \
  --candidate-max-new-tokens "10000" \
  --synthesis-max-new-tokens "10000" \
  --synthesis-batch-size "4" \
  --gradient-cache-root "outputs/shared_gradient_cache" \
  --seed "42" \
  --output-root "outputs/math_prompt_optimization" \
  --overwrite

CUDA_VISIBLE_DEVICES="$TUNING_GPU" python -u codes/run_qa_promptopt_gradpo.py \
  --code "math500_reasoning_qwen_rpo10_gradpo_tune_s5_t${MATH_TUNING_T}_h${H_TAG}_c10_g200_b10_f050_pool1600_vs900" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_10_from_actual_iteration_5.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.40" \
  --dual-vllm-gpu-memory-utilization "0.40" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "8192" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "1" \
  --num-edit-regions "5" \
  --max-region-tokens "$MATH_TUNING_T" \
  --region-expansion-threshold "$TUNING_H" \
  --num-region-candidates "10" \
  --beam-width "10" \
  --fluency-lambda "0.5" \
  --beam-replacement-mode "llm_synthesis" \
  --candidate-max-new-tokens "10000" \
  --synthesis-max-new-tokens "10000" \
  --synthesis-batch-size "4" \
  --gradient-cache-root "outputs/shared_gradient_cache" \
  --seed "42" \
  --output-root "outputs/math_prompt_optimization" \
  --overwrite
fi
