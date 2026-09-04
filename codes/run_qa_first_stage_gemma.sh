#!/usr/bin/env bash

# Run from the repository root. Each active command is a standalone experiment.
# Launch this entire file in the background from the repository root:
# nohup bash codes/run_qa_first_stage_gemma.sh > codes/nohup_outs/qa_etgpo_fixed_gemma.log 2>&1 &

# Previous lambda-1 runs are preserved below but commented out.

# PAUSED FOR NOW: completed, but reasoning RPO retained the initial prompt.
# CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_rpo.py \
#   --code openbookqa_reasoning_gemma_rpo_gemma12opt_lambda1 \
#   --qa-mode reasoning \
#   --model google/gemma-3-4b-it \
#   --optimizer-model google/gemma-3-12b-it \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --vllm-max-model-len 32768 \
#   --vllm-disable-images \
#   --optimizer-max-new-tokens 10000 \
#   --validation-std-penalty 1.0 \
#   --overwrite

# DONE FOR NOW: reasoning EvoPrompt-DE improved over the initial prompt.
# CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_evoprompt.py \
#   --code openbookqa_reasoning_gemma_evoprompt_gemma12opt_lambda1 \
#   --qa-mode reasoning \
#   --model google/gemma-3-4b-it \
#   --optimizer-model google/gemma-3-12b-it \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --vllm-max-model-len 32768 \
#   --vllm-disable-images \
#   --optimizer-max-new-tokens 10000 \
#   --duplicate-retries 3 \
#   --validation-std-penalty 1.0 \
#   --overwrite

# SUPERSEDED FOR CONSISTENT RERUN: reasoning ETGPO previously improved; rerun below.
# CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_etgpo.py \
#   --code openbookqa_reasoning_gemma_etgpo_gemma12opt_lambda1 \
#   --qa-mode reasoning \
#   --model google/gemma-3-4b-it \
#   --optimizer-model google/gemma-3-12b-it \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --vllm-max-model-len 32768 \
#   --vllm-disable-images \
#   --optimizer-max-new-tokens 10000 \
#   --validation-std-penalty 1.0 \
#   --overwrite

# DONE FOR NOW: non-reasoning RPO improved over the initial prompt.
# CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_rpo.py \
#   --code openbookqa_non_reasoning_gemma_rpo_gemma12opt_lambda1 \
#   --qa-mode non_reasoning \
#   --model google/gemma-3-4b-it \
#   --optimizer-model google/gemma-3-12b-it \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --vllm-max-model-len 32768 \
#   --vllm-disable-images \
#   --optimizer-max-new-tokens 10000 \
#   --validation-std-penalty 1.0 \
#   --overwrite

# DONE FOR NOW: non-reasoning EvoPrompt-DE improved over the initial prompt.
# CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_evoprompt.py \
#   --code openbookqa_non_reasoning_gemma_evoprompt_gemma12opt_lambda1 \
#   --qa-mode non_reasoning \
#   --model google/gemma-3-4b-it \
#   --optimizer-model google/gemma-3-12b-it \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --vllm-max-model-len 32768 \
#   --vllm-disable-images \
#   --optimizer-max-new-tokens 10000 \
#   --duplicate-retries 3 \
#   --validation-std-penalty 1.0 \
#   --overwrite

# SUPERSEDED: non-reasoning ETGPO predates feedback-first taxonomy; rerun below.
# CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_etgpo.py \
#   --code openbookqa_non_reasoning_gemma_etgpo_gemma12opt_lambda1 \
#   --qa-mode non_reasoning \
#   --model google/gemma-3-4b-it \
#   --optimizer-model google/gemma-3-12b-it \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.90 \
#   --vllm-max-model-len 32768 \
#   --vllm-disable-images \
#   --optimizer-max-new-tokens 10000 \
#   --validation-std-penalty 1.0 \
#   --overwrite

# Active ETGPO reruns using the current implementation.

# Gemma reasoning ETGPO: rerun alongside the corrected ETGPO set.
CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_etgpo.py \
  --code openbookqa_reasoning_gemma_etgpo_gemma12opt_lambda1_fixed \
  --qa-mode reasoning \
  --model google/gemma-3-4b-it \
  --optimizer-model google/gemma-3-12b-it \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --vllm-max-model-len 32768 \
  --vllm-disable-images \
  --optimizer-max-new-tokens 10000 \
  --feedback-max-new-tokens 10000 \
  --validation-std-penalty 1.0 \
  --overwrite

# Gemma non-reasoning ETGPO: rerun with feedback before taxonomy construction.
CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_etgpo.py \
  --code openbookqa_non_reasoning_gemma_etgpo_gemma12opt_lambda1_fixed \
  --qa-mode non_reasoning \
  --model google/gemma-3-4b-it \
  --optimizer-model google/gemma-3-12b-it \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --vllm-max-model-len 32768 \
  --vllm-disable-images \
  --optimizer-max-new-tokens 10000 \
  --feedback-max-new-tokens 10000 \
  --validation-std-penalty 1.0 \
  --overwrite
