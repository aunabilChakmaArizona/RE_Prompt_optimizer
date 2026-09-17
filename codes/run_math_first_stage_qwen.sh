#!/usr/bin/env bash

# Run from the repository root. Only the revised Math Qwen RPO trial is active.
# nohup bash codes/run_math_first_stage_qwen.sh > codes/nohup_outs/math_rpo_qwen_error3_tokenlimit.log 2>&1 &

# Previous first-stage runs are complete and preserved below for tracking.
# CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_rpo.py \
#   --code math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900 \
#   --qa-task math500 \
#   --qa-mode reasoning \
#   --train-path data/processed/math500/train.jsonl \
#   --validation-path data/processed/math500/validation.jsonl \
#   --model Qwen/Qwen3-4B \
#   --optimizer-model Qwen/Qwen3-14B \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --target-max-new-tokens 8192 \
#   --optimizer-max-new-tokens 10000 \
#   --optimizer-feedback-max-tokens 2000 \
#   --validation-std-penalty 1.0 \
#   --validation-fold-size 300 \
#   --output-root outputs/math_prompt_optimization \
#   --overwrite

# CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_evoprompt.py \
#   --code math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900 \
#   --qa-task math500 \
#   --qa-mode reasoning \
#   --train-path data/processed/math500/train.jsonl \
#   --validation-path data/processed/math500/validation.jsonl \
#   --model Qwen/Qwen3-4B \
#   --optimizer-model Qwen/Qwen3-14B \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --target-max-new-tokens 8192 \
#   --optimizer-max-new-tokens 10000 \
#   --optimizer-feedback-max-tokens 2000 \
#   --train-sample-size 500 \
#   --duplicate-retries 1 \
#   --validation-std-penalty 1.0 \
#   --validation-fold-size 300 \
#   --output-root outputs/math_prompt_optimization \
#   --overwrite

# CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_etgpo.py \
#   --code math500_reasoning_qwen_etgpo_qwen14opt_lambda1_vs900 \
#   --qa-task math500 \
#   --qa-mode reasoning \
#   --train-path data/processed/math500/train.jsonl \
#   --validation-path data/processed/math500/validation.jsonl \
#   --model Qwen/Qwen3-4B \
#   --optimizer-model Qwen/Qwen3-14B \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --target-max-new-tokens 8192 \
#   --optimizer-max-new-tokens 10000 \
#   --optimizer-feedback-max-tokens 2000 \
#   --feedback-max-new-tokens 10000 \
#   --taxonomy-retries 2 \
#   --validation-std-penalty 1.0 \
#   --validation-fold-size 300 \
#   --output-root outputs/math_prompt_optimization \
#   --overwrite

# Revised RPO trial: three incorrect examples; explicit token-limit feedback.
# New CODE preserves the historical RPO run. Test evaluation remains disabled.
CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_rpo.py \
  --code math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --model Qwen/Qwen3-4B \
  --optimizer-model Qwen/Qwen3-14B \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --iterations 10 \
  --feedback-sample-size 100 \
  --feedback-examples 3 \
  --target-max-new-tokens 8192 \
  --optimizer-max-new-tokens 10000 \
  --optimizer-feedback-max-tokens 2000 \
  --validation-std-penalty 1.0 \
  --validation-fold-size 300 \
  --seed 42 \
  --output-root outputs/math_prompt_optimization \
  --overwrite
