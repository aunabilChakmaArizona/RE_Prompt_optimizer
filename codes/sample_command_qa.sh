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