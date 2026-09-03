CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_evoprompt.py \
  --code openbookqa_non_reasoning_gemma_evoprompt_gemma12opt \
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
  --duplicate-retries 3 \
  --validation-std-penalty 1.0 \
  --overwrite

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_etgpo.py \
  --code openbookqa_non_reasoning_gemma_etgpo_gemma12opt \
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
  --validation-std-penalty 1.0 \
  --overwrite
