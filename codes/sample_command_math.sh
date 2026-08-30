python -u codes/run_math_test_inference.py \
  --code math500_qwen3_4b_base \
  --dataset data/processed/math500/test.jsonl \
  --model Qwen/Qwen3-4B \
  --device cuda:1 \
  --max_new_tokens 4096 \
  --batch_size 8