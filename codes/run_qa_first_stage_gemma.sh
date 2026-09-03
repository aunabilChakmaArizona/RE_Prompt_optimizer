#!/usr/bin/env bash

# Run from the repository root. Each command is a standalone experiment.

# Gemma reasoning.

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_rpo.py \
  --code openbookqa_reasoning_gemma_rpo_gemma12opt \
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
  --validation-std-penalty 2.0 \
  --overwrite

# Completed successfully; kept for reference.
# CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_evoprompt.py \
#   --code openbookqa_reasoning_gemma_evoprompt_gemma12opt \
#   --qa-mode reasoning \
#   --model google/gemma-3-4b-it \
#   --optimizer-model google/gemma-3-12b-it \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.65 \
#   --vllm-max-model-len 32768 \
#   --vllm-disable-images \
#   --optimizer-max-new-tokens 10000 \
#   --validation-std-penalty 2.0 \
#   --overwrite

# Completed successfully; kept for reference.
# CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_etgpo.py \
#   --code openbookqa_reasoning_gemma_etgpo_gemma12opt \
#   --qa-mode reasoning \
#   --model google/gemma-3-4b-it \
#   --optimizer-model google/gemma-3-12b-it \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.65 \
#   --vllm-max-model-len 32768 \
#   --vllm-disable-images \
#   --optimizer-max-new-tokens 10000 \
#   --validation-std-penalty 2.0 \
#   --overwrite

# Gemma non-reasoning.

# Completed successfully; kept for reference.
# CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_rpo.py \
#   --code openbookqa_non_reasoning_gemma_rpo_gemma12opt \
#   --qa-mode non_reasoning \
#   --model google/gemma-3-4b-it \
#   --optimizer-model google/gemma-3-12b-it \
#   --device cuda:0 \
#   --optimizer-device cuda:0 \
#   --backend vllm \
#   --gpu-memory-utilization 0.65 \
#   --vllm-max-model-len 32768 \
#   --vllm-disable-images \
#   --optimizer-max-new-tokens 10000 \
#   --validation-std-penalty 2.0 \
#   --overwrite

# CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_evoprompt.py \
#   --code openbookqa_non_reasoning_gemma_evoprompt_gemma12opt \
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
#   --validation-std-penalty 2.0 \
#   --overwrite

# CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_etgpo.py \
#   --code openbookqa_non_reasoning_gemma_etgpo_gemma12opt \
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
#   --validation-std-penalty 2.0 \
#   --overwrite
