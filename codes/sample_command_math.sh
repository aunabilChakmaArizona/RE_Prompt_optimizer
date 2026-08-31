python -u codes/run_math_test_inference.py \
  --code math500_qwen3_4b_base \
  --dataset data/processed/math500/test.jsonl \
  --model Qwen/Qwen3-4B \
  --device cuda:1 \
  --max_new_tokens 4096 \
  --batch_size 8

python -u codes/run_math_test_inference.py \
  --backend vllm \
  --code math500_qwen3_4b_vllm_bs8192 \
  --dataset data/processed/math500/test.jsonl \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --max_new_tokens 8192 \
  --gpu_memory_utilization 0.90


# Gemma3-4B with Transformers
python -u codes/run_math_test_inference.py \
  --code math500_gemma3_4b_base \
  --dataset data/processed/math500/test.jsonl \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --max_new_tokens 4096 \
  --batch_size 8

# Gemma3-4B with vLLM
python -u codes/run_math_test_inference.py \
  --backend vllm \
  --code math500_gemma3_4b_vllm_bs8192 \
  --dataset data/processed/math500/test.jsonl \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --max_new_tokens 8192 \
  --gpu_memory_utilization 0.90
