#!/usr/bin/env bash

# Run from the repository root. Each command is a standalone experiment.

# Qwen reasoning.

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_rpo.py \
  --code openbookqa_reasoning_qwen_rpo_qwen14opt \
  --qa-mode reasoning \
  --model Qwen/Qwen3-4B \
  --optimizer-model Qwen/Qwen3-14B \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0 \
  --overwrite

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_evoprompt.py \
  --code openbookqa_reasoning_qwen_evoprompt_qwen14opt \
  --qa-mode reasoning \
  --model Qwen/Qwen3-4B \
  --optimizer-model Qwen/Qwen3-14B \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0 \
  --overwrite

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_etgpo.py \
  --code openbookqa_reasoning_qwen_etgpo_qwen14opt \
  --qa-mode reasoning \
  --model Qwen/Qwen3-4B \
  --optimizer-model Qwen/Qwen3-14B \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0 \
  --overwrite

# Qwen non-reasoning.

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_rpo.py \
  --code openbookqa_non_reasoning_qwen_rpo_qwen14opt \
  --qa-mode non_reasoning \
  --model Qwen/Qwen3-4B \
  --optimizer-model Qwen/Qwen3-14B \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0 \
  --overwrite

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_evoprompt.py \
  --code openbookqa_non_reasoning_qwen_evoprompt_qwen14opt \
  --qa-mode non_reasoning \
  --model Qwen/Qwen3-4B \
  --optimizer-model Qwen/Qwen3-14B \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0 \
  --overwrite

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_etgpo.py \
  --code openbookqa_non_reasoning_qwen_etgpo_qwen14opt \
  --qa-mode non_reasoning \
  --model Qwen/Qwen3-4B \
  --optimizer-model Qwen/Qwen3-14B \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0 \
  --overwrite
