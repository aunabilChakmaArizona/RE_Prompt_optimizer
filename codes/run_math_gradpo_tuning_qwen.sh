#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. No final test evaluation is enabled.
# Activate re_prompt_optimization_vllm_v2 before running this script.
# vLLM scores candidate loss/fluency; HF still computes all gradients.
# Default: all eight fast attempts, sequentially (four configurations x RPO5/RPO10).
# C=7 new replacements (+ unchanged), B=5, target output=4096, pool=1600, G=200.
# HF gradient batch=1; HF selection batch=4; vLLM objective submissions=128.
# Exact prompt scores are reused within each beam search; no length sorting.
# nohup bash codes/run_math_gradpo_tuning_qwen.sh > codes/nohup_outs/math_gradpo_tuning_qwen_fast4096.log 2>&1 &
# Optional: MATH_TUNING_STAGE=width runs just the first four attempts.
# MATH_TUNING_STAGE=expansion MATH_TUNING_T=SELECTED_T nohup bash codes/run_math_gradpo_tuning_qwen.sh > codes/nohup_outs/math_gradpo_tuning_qwen_expansion.log 2>&1 &
# MATH_TUNING_STAGE=count MATH_TUNING_T=SELECTED_T MATH_TUNING_H=SELECTED_H nohup bash codes/run_math_gradpo_tuning_qwen.sh > codes/nohup_outs/math_gradpo_tuning_qwen_count.log 2>&1 &
# Optional GPU override: MATH_TUNING_GPU=1 (default physical GPU 3).
# Both HF and vLLM use logical cuda:0; vLLM memory budget is 50%.
# Cache sharing uses greedy source responses. Validation remains sampled.
# RPO5/RPO10 are assigned source slots; actual iterations are 2/5.

TUNING_GPU="${MATH_TUNING_GPU:-3}"
TUNING_STAGE="${MATH_TUNING_STAGE:-all}"

case "$TUNING_STAGE" in
  all|width|expansion|count) ;;
  *) printf 'Invalid MATH_TUNING_STAGE: %s\n' "$TUNING_STAGE" >&2; exit 2 ;;
esac

# In the all-attempts plan, later configurations use T=5 and H=0.30.
# Explicit phase overrides remain available for follow-up tuning.
if [[ "$TUNING_STAGE" == "all" ]]; then
  MATH_TUNING_T="${MATH_TUNING_T:-5}"
  MATH_TUNING_H="${MATH_TUNING_H:-0.30}"
fi

if [[ "$TUNING_STAGE" != "width" ]]; then
  : "${MATH_TUNING_T:?Set MATH_TUNING_T to 3 or 5 after inspecting both source results}"
  case "$MATH_TUNING_T" in
    3|5) ;;
    *) printf 'MATH_TUNING_T must be 3 or 5.\n' >&2; exit 2 ;;
  esac
fi

if [[ "$TUNING_STAGE" == "count" || "$TUNING_STAGE" == "all" ]]; then
  : "${MATH_TUNING_H:?Set MATH_TUNING_H to the selected 0.30 or 0.45}"
  case "$MATH_TUNING_H" in
    0.3|0.30) TUNING_H=0.30; H_TAG=030 ;;
    0.45) TUNING_H=0.45; H_TAG=045 ;;
    *) printf 'MATH_TUNING_H must be 0.30 or 0.45.\n' >&2; exit 2 ;;
  esac
fi

if [[ "$TUNING_STAGE" == "width" || "$TUNING_STAGE" == "all" ]]; then
# Attempts 1-2: transferred S/T/H with the reduced Math C/B and output budget.
CUDA_VISIBLE_DEVICES="$TUNING_GPU" python -u codes/run_qa_promptopt_gradpo.py \
  --code "math500_reasoning_qwen_rpo5_gradpo_tune_s3_t3_h045_c7_g200_b5_f050_pool1600_vs900_tok4096_v2" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_5_from_actual_iteration_2.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --objective-scoring-backend "vllm" \
  --objective-scoring-batch-size "128" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.50" \
  --dual-vllm-gpu-memory-utilization "0.50" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "4096" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "4" \
  --num-edit-regions "3" \
  --max-region-tokens "3" \
  --region-expansion-threshold "0.45" \
  --num-region-candidates "7" \
  --beam-width "5" \
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
  --code "math500_reasoning_qwen_rpo10_gradpo_tune_s3_t3_h045_c7_g200_b5_f050_pool1600_vs900_tok4096_v2" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_10_from_actual_iteration_5.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --objective-scoring-backend "vllm" \
  --objective-scoring-batch-size "128" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.50" \
  --dual-vllm-gpu-memory-utilization "0.50" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "4096" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "4" \
  --num-edit-regions "3" \
  --max-region-tokens "3" \
  --region-expansion-threshold "0.45" \
  --num-region-candidates "7" \
  --beam-width "5" \
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
  --code "math500_reasoning_qwen_rpo5_gradpo_tune_s3_t5_h045_c7_g200_b5_f050_pool1600_vs900_tok4096_v2" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_5_from_actual_iteration_2.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --objective-scoring-backend "vllm" \
  --objective-scoring-batch-size "128" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.50" \
  --dual-vllm-gpu-memory-utilization "0.50" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "4096" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "4" \
  --num-edit-regions "3" \
  --max-region-tokens "5" \
  --region-expansion-threshold "0.45" \
  --num-region-candidates "7" \
  --beam-width "5" \
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
  --code "math500_reasoning_qwen_rpo10_gradpo_tune_s3_t5_h045_c7_g200_b5_f050_pool1600_vs900_tok4096_v2" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_10_from_actual_iteration_5.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --objective-scoring-backend "vllm" \
  --objective-scoring-batch-size "128" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.50" \
  --dual-vllm-gpu-memory-utilization "0.50" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "4096" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "4" \
  --num-edit-regions "3" \
  --max-region-tokens "5" \
  --region-expansion-threshold "0.45" \
  --num-region-candidates "7" \
  --beam-width "5" \
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

if [[ "$TUNING_STAGE" == "expansion" || "$TUNING_STAGE" == "all" ]]; then
# Attempts 5-6: T=5 in the default all-attempts plan, with lower H=0.30.
CUDA_VISIBLE_DEVICES="$TUNING_GPU" python -u codes/run_qa_promptopt_gradpo.py \
  --code "math500_reasoning_qwen_rpo5_gradpo_tune_s3_t${MATH_TUNING_T}_h030_c7_g200_b5_f050_pool1600_vs900_tok4096_v2" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_5_from_actual_iteration_2.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --objective-scoring-backend "vllm" \
  --objective-scoring-batch-size "128" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.50" \
  --dual-vllm-gpu-memory-utilization "0.50" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "4096" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "4" \
  --num-edit-regions "3" \
  --max-region-tokens "$MATH_TUNING_T" \
  --region-expansion-threshold "0.30" \
  --num-region-candidates "7" \
  --beam-width "5" \
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
  --code "math500_reasoning_qwen_rpo10_gradpo_tune_s3_t${MATH_TUNING_T}_h030_c7_g200_b5_f050_pool1600_vs900_tok4096_v2" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_10_from_actual_iteration_5.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --objective-scoring-backend "vllm" \
  --objective-scoring-batch-size "128" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.50" \
  --dual-vllm-gpu-memory-utilization "0.50" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "4096" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "4" \
  --num-edit-regions "3" \
  --max-region-tokens "$MATH_TUNING_T" \
  --region-expansion-threshold "0.30" \
  --num-region-candidates "7" \
  --beam-width "5" \
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

if [[ "$TUNING_STAGE" == "count" || "$TUNING_STAGE" == "all" ]]; then
# Attempts 7-8: five regions, T=5 and H=0.30 by default.
CUDA_VISIBLE_DEVICES="$TUNING_GPU" python -u codes/run_qa_promptopt_gradpo.py \
  --code "math500_reasoning_qwen_rpo5_gradpo_tune_s5_t${MATH_TUNING_T}_h${H_TAG}_c7_g200_b5_f050_pool1600_vs900_tok4096_v2" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_5_from_actual_iteration_2.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --objective-scoring-backend "vllm" \
  --objective-scoring-batch-size "128" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.50" \
  --dual-vllm-gpu-memory-utilization "0.50" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "4096" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "4" \
  --num-edit-regions "5" \
  --max-region-tokens "$MATH_TUNING_T" \
  --region-expansion-threshold "$TUNING_H" \
  --num-region-candidates "7" \
  --beam-width "5" \
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
  --code "math500_reasoning_qwen_rpo10_gradpo_tune_s5_t${MATH_TUNING_T}_h${H_TAG}_c7_g200_b5_f050_pool1600_vs900_tok4096_v2" \
  --qa-task "math500" \
  --qa-mode "reasoning" \
  --train-path "data/processed/math500/train.jsonl" \
  --validation-path "data/processed/math500/validation.jsonl" \
  --initial-prompt-file "outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit/prompt_experimental_iteration_10_from_actual_iteration_5.txt" \
  --model "Qwen/Qwen3-4B" \
  --device "cuda:0" \
  --hf-device "cuda:0" \
  --backend "dual" \
  --objective-scoring-backend "vllm" \
  --objective-scoring-batch-size "128" \
  --final-evaluation-backend "vllm" \
  --gpu-memory-utilization "0.50" \
  --dual-vllm-gpu-memory-utilization "0.50" \
  --vllm-max-model-len "32768" \
  --target-max-new-tokens "4096" \
  --validation-std-penalty "1.0" \
  --validation-fold-size "300" \
  --variant "gen" \
  --train-sample-size "1600" \
  --gradient-sample-size "200" \
  --gradient-batch-size "1" \
  --selection-batch-size "4" \
  --num-edit-regions "5" \
  --max-region-tokens "$MATH_TUNING_T" \
  --region-expansion-threshold "$TUNING_H" \
  --num-region-candidates "7" \
  --beam-width "5" \
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
