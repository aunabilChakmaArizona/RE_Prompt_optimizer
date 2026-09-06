#!/usr/bin/env bash

# FOLIO v2 validation used as a labeled local test set.
CUDA_VISIBLE_DEVICES=3 python -u codes/run_folio_reasoning_test_inference.py \
  --backend vllm \
  --code folio_qwen3_4b_reasoning \
  --dataset data/processed/folio/test.jsonl \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --max_new_tokens 4096 \
  --gpu_memory_utilization 0.90 \
  --overwrite

CUDA_VISIBLE_DEVICES=3 python -u codes/run_folio_reasoning_test_inference.py \
  --backend vllm \
  --code folio_gemma3_4b_reasoning \
  --dataset data/processed/folio/test.jsonl \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --max_new_tokens 4096 \
  --gpu_memory_utilization 0.90 \
  --overwrite

# HotpotQA distractor validation used as a labeled local test set.
CUDA_VISIBLE_DEVICES=3 python -u codes/run_hotpotqa_reasoning_test_inference.py \
  --backend vllm \
  --code hotpotqa_qwen3_4b_reasoning \
  --dataset data/processed/hotpotqa/test.jsonl \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --max_new_tokens 4096 \
  --gpu_memory_utilization 0.90 \
  --overwrite

CUDA_VISIBLE_DEVICES=3 python -u codes/run_hotpotqa_reasoning_test_inference.py \
  --backend vllm \
  --code hotpotqa_gemma3_4b_reasoning \
  --dataset data/processed/hotpotqa/test.jsonl \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --max_new_tokens 4096 \
  --gpu_memory_utilization 0.90 \
  --overwrite
