python -u codes/run_qa_reasoning_test_inference.py \
  --code openbookqa_qwen_reasoning_base \
  --dataset data/processed/openbookqa/test.jsonl \
  --model Qwen/Qwen3-4B \
  --device cuda:3 \
  --batch_size 8 \
  --max_new_tokens 1024

python -u codes/run_qa_non_reasoning_test_inference.py \
  --code openbookqa_qwen_direct_base \
  --dataset data/processed/openbookqa/test.jsonl \
  --model Qwen/Qwen3-4B \
  --device cuda:3 \
  --batch_size 16 \
  --max_new_tokens 16


python -u codes/run_qa_reasoning_test_inference.py \
  --backend vllm \
  --code openbookqa_qwen3_4b_reasoning_vllm \
  --dataset data/processed/openbookqa/test.jsonl \
  --model Qwen/Qwen3-4B \
  --device cuda:3 \
  --max_new_tokens 2048 \
  --gpu_memory_utilization 0.90


# Gemma3-4B with Transformers: reasoning
python -u codes/run_qa_reasoning_test_inference.py \
  --code openbookqa_gemma3_4b_reasoning_base \
  --dataset data/processed/openbookqa/test.jsonl \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --batch_size 8 \
  --max_new_tokens 1024

# Gemma3-4B with Transformers: non-reasoning
python -u codes/run_qa_non_reasoning_test_inference.py \
  --code openbookqa_gemma3_4b_direct_base \
  --dataset data/processed/openbookqa/test.jsonl \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --batch_size 16 \
  --max_new_tokens 16

# Gemma3-4B with vLLM: reasoning
python -u codes/run_qa_reasoning_test_inference.py \
  --backend vllm \
  --code openbookqa_gemma3_4b_reasoning_vllm \
  --dataset data/processed/openbookqa/test.jsonl \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --max_new_tokens 2048 \
  --gpu_memory_utilization 0.90
