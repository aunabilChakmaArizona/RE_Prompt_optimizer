#!/usr/bin/env bash

# Run from the repository root. Each active command is a standalone experiment.
# Launch this entire file in the background from the repository root:
# nohup bash codes/run_qa_first_stage_qwen.sh > codes/nohup_outs/run_qa_first_stage_qwen.log 2>&1 &

# Previous lambda-1 runs are preserved below but commented out.

# PAUSED FOR NOW: completed, but reasoning RPO retained the initial prompt.
# CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_rpo.py \
#   --code openbookqa_reasoning_qwen_rpo_qwen14opt_lambda1 \
#   --qa-mode reasoning \
#   --model Qwen/Qwen3-4B \
#   --optimizer-model Qwen/Qwen3-14B \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --optimizer-max-new-tokens 10000 \
#   --validation-std-penalty 1.0 \
#   --overwrite

# DONE FOR NOW: reasoning EvoPrompt-DE improved over the initial prompt.
# CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_evoprompt.py \
#   --code openbookqa_reasoning_qwen_evoprompt_qwen14opt_lambda1 \
#   --qa-mode reasoning \
#   --model Qwen/Qwen3-4B \
#   --optimizer-model Qwen/Qwen3-14B \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --optimizer-max-new-tokens 10000 \
#   --duplicate-retries 3 \
#   --validation-std-penalty 1.0 \
#   --overwrite

# SUPERSEDED: reasoning ETGPO retained the initial prompt; rerun below.
# CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_etgpo.py \
#   --code openbookqa_reasoning_qwen_etgpo_qwen14opt_lambda1 \
#   --qa-mode reasoning \
#   --model Qwen/Qwen3-4B \
#   --optimizer-model Qwen/Qwen3-14B \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --optimizer-max-new-tokens 10000 \
#   --validation-std-penalty 1.0 \
#   --overwrite

# DONE FOR NOW: non-reasoning RPO improved over the initial prompt.
# CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_rpo.py \
#   --code openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1 \
#   --qa-mode non_reasoning \
#   --model Qwen/Qwen3-4B \
#   --optimizer-model Qwen/Qwen3-14B \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --optimizer-max-new-tokens 10000 \
#   --validation-std-penalty 1.0 \
#   --overwrite

# DONE FOR NOW: non-reasoning EvoPrompt-DE improved over the initial prompt.
# CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_evoprompt.py \
#   --code openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1 \
#   --qa-mode non_reasoning \
#   --model Qwen/Qwen3-4B \
#   --optimizer-model Qwen/Qwen3-14B \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --optimizer-max-new-tokens 10000 \
#   --duplicate-retries 3 \
#   --validation-std-penalty 1.0 \
#   --overwrite

# SUPERSEDED: non-reasoning ETGPO predates feedback-first taxonomy; rerun below.
# CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_etgpo.py \
#   --code openbookqa_non_reasoning_qwen_etgpo_qwen14opt_lambda1 \
#   --qa-mode non_reasoning \
#   --model Qwen/Qwen3-4B \
#   --optimizer-model Qwen/Qwen3-14B \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --optimizer-max-new-tokens 10000 \
#   --validation-std-penalty 1.0 \
#   --overwrite

# Archived ETGPO reruns using lambda=1 and validation=1,500. These predate
# optimizer-response sanitization and must be rerun before use as current results.

# ARCHIVED: Qwen reasoning ETGPO improved by +0.20 raw and +0.68 stable points.
# CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_etgpo.py \
#   --code openbookqa_reasoning_qwen_etgpo_qwen14opt_lambda1_fixed_vs1500 \
#   --qa-mode reasoning \
#   --model Qwen/Qwen3-4B \
#   --optimizer-model Qwen/Qwen3-14B \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --optimizer-max-new-tokens 10000 \
#   --feedback-max-new-tokens 10000 \
#   --validation-std-penalty 1.0 \
#   --overwrite

# ARCHIVED: Qwen non-reasoning ETGPO improved by +0.53 raw and +0.68 stable points.
# CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_etgpo.py \
#   --code openbookqa_non_reasoning_qwen_etgpo_qwen14opt_lambda1_fixed_vs1500 \
#   --qa-mode non_reasoning \
#   --model Qwen/Qwen3-4B \
#   --optimizer-model Qwen/Qwen3-14B \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --optimizer-max-new-tokens 10000 \
#   --feedback-max-new-tokens 10000 \
#   --validation-std-penalty 1.0 \
#   --overwrite

# ACTIVE: Qwen reasoning RPO previously retained the initial prompt; rerun with
# lambda=1 on the expanded 1,500-question validation set.
CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_rpo.py \
  --code openbookqa_reasoning_qwen_rpo_qwen14opt_lambda1_vs1500 \
  --qa-mode reasoning \
  --model Qwen/Qwen3-4B \
  --optimizer-model Qwen/Qwen3-14B \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 1.0 \
  --overwrite


  
