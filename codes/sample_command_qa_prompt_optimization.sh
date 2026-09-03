#!/usr/bin/env bash

# Run from the repository root. Every Python command below is standalone.

# Base test inference: Qwen reasoning and non-reasoning.

python -u codes/run_qa_reasoning_test_inference.py \
  --backend vllm \
  --code openbookqa_qwen3_4b_reasoning_vllm \
  --dataset data/processed/openbookqa/test.jsonl \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --max_new_tokens 4096 \
  --gpu_memory_utilization 0.90

python -u codes/run_qa_non_reasoning_test_inference.py \
  --backend vllm \
  --code openbookqa_qwen3_4b_non_reasoning_vllm \
  --dataset data/processed/openbookqa/test.jsonl \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --max_new_tokens 16 \
  --gpu_memory_utilization 0.90

# Base test inference: Gemma reasoning and non-reasoning.

python -u codes/run_qa_reasoning_test_inference.py \
  --backend vllm \
  --code openbookqa_gemma3_4b_reasoning_vllm \
  --dataset data/processed/openbookqa/test.jsonl \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --max_new_tokens 4096 \
  --gpu_memory_utilization 0.90

python -u codes/run_qa_non_reasoning_test_inference.py \
  --backend vllm \
  --code openbookqa_gemma3_4b_non_reasoning_vllm \
  --dataset data/processed/openbookqa/test.jsonl \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --max_new_tokens 16 \
  --gpu_memory_utilization 0.90

# First stage: Qwen reasoning.

python -u codes/run_qa_promptopt_rpo.py \
  --code openbookqa_reasoning_qwen_rpo \
  --qa-mode reasoning \
  --model Qwen/Qwen3-4B \
  --optimizer-model Qwen/Qwen3-4B \
  --device cuda:3 \
  --optimizer-device cuda:3 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_evoprompt.py \
  --code openbookqa_reasoning_qwen_evoprompt \
  --qa-mode reasoning \
  --model Qwen/Qwen3-4B \
  --optimizer-model Qwen/Qwen3-14B \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_etgpo.py \
  --code openbookqa_reasoning_qwen_etgpo \
  --qa-mode reasoning \
  --model Qwen/Qwen3-4B \
  --optimizer-model Qwen/Qwen3-14B \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0

# First stage: Qwen non-reasoning.

python -u codes/run_qa_promptopt_rpo.py \
  --code openbookqa_non_reasoning_qwen_rpo \
  --qa-mode non_reasoning \
  --model Qwen/Qwen3-4B \
  --optimizer-model Qwen/Qwen3-14B \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_evoprompt.py \
  --code openbookqa_non_reasoning_qwen_evoprompt \
  --qa-mode non_reasoning \
  --model Qwen/Qwen3-4B \
  --optimizer-model Qwen/Qwen3-14B \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_etgpo.py \
  --code openbookqa_non_reasoning_qwen_etgpo \
  --qa-mode non_reasoning \
  --model Qwen/Qwen3-4B \
  --optimizer-model Qwen/Qwen3-14B \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0

# First stage: Gemma reasoning.

python -u codes/run_qa_promptopt_rpo.py \
  --code openbookqa_reasoning_gemma_rpo \
  --qa-mode reasoning \
  --model google/gemma-3-4b-it \
  --optimizer-model google/gemma-3-12b-it \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_evoprompt.py \
  --code openbookqa_reasoning_gemma_evoprompt \
  --qa-mode reasoning \
  --model google/gemma-3-4b-it \
  --optimizer-model google/gemma-3-12b-it \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_etgpo.py \
  --code openbookqa_reasoning_gemma_etgpo \
  --qa-mode reasoning \
  --model google/gemma-3-4b-it \
  --optimizer-model google/gemma-3-12b-it \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0

# First stage: Gemma non-reasoning.

python -u codes/run_qa_promptopt_rpo.py \
  --code openbookqa_non_reasoning_gemma_rpo \
  --qa-mode non_reasoning \
  --model google/gemma-3-4b-it \
  --optimizer-model google/gemma-3-12b-it \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_evoprompt.py \
  --code openbookqa_non_reasoning_gemma_evoprompt \
  --qa-mode non_reasoning \
  --model google/gemma-3-4b-it \
  --optimizer-model google/gemma-3-12b-it \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_etgpo.py \
  --code openbookqa_non_reasoning_gemma_etgpo \
  --qa-mode non_reasoning \
  --model google/gemma-3-4b-it \
  --optimizer-model google/gemma-3-12b-it \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --validation-std-penalty 2.0

# Second stage: Qwen reasoning examples starting from the saved RPO-5 prompt.

python -u codes/run_qa_promptopt_lpo.py \
  --code openbookqa_reasoning_qwen_rpo5_lpo \
  --qa-mode reasoning \
  --model Qwen/Qwen3-4B \
  --optimizer-model Qwen/Qwen3-14B \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --initial-prompt-file outputs/qa_prompt_optimization/reasoning/rpo/openbookqa_reasoning_qwen_rpo/prompt_iteration_5.txt \
  --feedback-examples 3 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_greater.py \
  --code openbookqa_reasoning_qwen_rpo5_greater \
  --qa-mode reasoning \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --backend transformers \
  --initial-prompt-file outputs/qa_prompt_optimization/reasoning/rpo/openbookqa_reasoning_qwen_rpo/prompt_iteration_5.txt \
  --variant greater \
  --train-sample-size 3000 \
  --gradient-batch-size 4 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_greater.py \
  --code openbookqa_reasoning_qwen_rpo5_greater_tg \
  --qa-mode reasoning \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --backend transformers \
  --initial-prompt-file outputs/qa_prompt_optimization/reasoning/rpo/openbookqa_reasoning_qwen_rpo/prompt_iteration_5.txt \
  --variant greater_tg \
  --train-sample-size 3000 \
  --gradient-batch-size 4 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_reasoning_qwen_rpo5_gradpo_gen \
  --qa-mode reasoning \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --backend transformers \
  --initial-prompt-file outputs/qa_prompt_optimization/reasoning/rpo/openbookqa_reasoning_qwen_rpo/prompt_iteration_5.txt \
  --variant gen \
  --train-sample-size 3000 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_reasoning_qwen_rpo5_gradpo_prob \
  --qa-mode reasoning \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --backend transformers \
  --initial-prompt-file outputs/qa_prompt_optimization/reasoning/rpo/openbookqa_reasoning_qwen_rpo/prompt_iteration_5.txt \
  --variant prob \
  --train-sample-size 3000 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_reasoning_qwen_rpo5_gradpo_gen_random \
  --qa-mode reasoning \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --backend transformers \
  --initial-prompt-file outputs/qa_prompt_optimization/reasoning/rpo/openbookqa_reasoning_qwen_rpo/prompt_iteration_5.txt \
  --variant gen_random \
  --train-sample-size 3000 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --validation-std-penalty 2.0

# Second stage: Gemma reasoning examples starting from the saved RPO-5 prompt.

python -u codes/run_qa_promptopt_lpo.py \
  --code openbookqa_reasoning_gemma_rpo5_lpo \
  --qa-mode reasoning \
  --model google/gemma-3-4b-it \
  --optimizer-model google/gemma-3-12b-it \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --initial-prompt-file outputs/qa_prompt_optimization/reasoning/rpo/openbookqa_reasoning_gemma_rpo/prompt_iteration_5.txt \
  --feedback-examples 3 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_greater.py \
  --code openbookqa_reasoning_gemma_rpo5_greater \
  --qa-mode reasoning \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --backend transformers \
  --initial-prompt-file outputs/qa_prompt_optimization/reasoning/rpo/openbookqa_reasoning_gemma_rpo/prompt_iteration_5.txt \
  --variant greater \
  --train-sample-size 3000 \
  --gradient-batch-size 4 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_greater.py \
  --code openbookqa_reasoning_gemma_rpo5_greater_tg \
  --qa-mode reasoning \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --backend transformers \
  --initial-prompt-file outputs/qa_prompt_optimization/reasoning/rpo/openbookqa_reasoning_gemma_rpo/prompt_iteration_5.txt \
  --variant greater_tg \
  --train-sample-size 3000 \
  --gradient-batch-size 4 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_reasoning_gemma_rpo5_gradpo_gen \
  --qa-mode reasoning \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --backend transformers \
  --initial-prompt-file outputs/qa_prompt_optimization/reasoning/rpo/openbookqa_reasoning_gemma_rpo/prompt_iteration_5.txt \
  --variant gen \
  --train-sample-size 3000 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_reasoning_gemma_rpo5_gradpo_prob \
  --qa-mode reasoning \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --backend transformers \
  --initial-prompt-file outputs/qa_prompt_optimization/reasoning/rpo/openbookqa_reasoning_gemma_rpo/prompt_iteration_5.txt \
  --variant prob \
  --train-sample-size 3000 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --validation-std-penalty 2.0

python -u codes/run_qa_promptopt_gradpo.py \
  --code openbookqa_reasoning_gemma_rpo5_gradpo_gen_random \
  --qa-mode reasoning \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --backend transformers \
  --initial-prompt-file outputs/qa_prompt_optimization/reasoning/rpo/openbookqa_reasoning_gemma_rpo/prompt_iteration_5.txt \
  --variant gen_random \
  --train-sample-size 3000 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --validation-std-penalty 2.0

# Generate the complete matrices, including both reasoning modes and all prompt sources.

python -u codes/generate_qa_promptopt_commands.py \
  --phase first_stage \
  --qa-mode all \
  --model-family all \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --output-file codes/qa_promptopt_first_stage_commands.sh

python -u codes/generate_qa_promptopt_commands.py \
  --phase second_stage \
  --qa-mode all \
  --model-family all \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --output-file codes/qa_promptopt_second_stage_commands.sh

# Summarize validation results.

python -u codes/report_qa_prompt_optimization.py \
  --output-root outputs/qa_prompt_optimization \
  --report-file qa_prompt_optimization_stats.txt

# Final five-run test evaluation examples after prompt selection is complete.

python -u codes/run_qa_final_test_evaluation.py \
  --code openbookqa_reasoning_qwen_rpo10_final \
  --qa-mode reasoning \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --runs 5 \
  --max-new-tokens 4096 \
  --prompt-file outputs/qa_prompt_optimization/reasoning/rpo/openbookqa_reasoning_qwen_rpo/prompt_iteration_10.txt

python -u codes/run_qa_final_test_evaluation.py \
  --code openbookqa_non_reasoning_qwen_rpo10_final \
  --qa-mode non_reasoning \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --runs 5 \
  --max-new-tokens 16 \
  --prompt-file outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_qwen_rpo/prompt_iteration_10.txt

python -u codes/run_qa_final_test_evaluation.py \
  --code openbookqa_reasoning_gemma_rpo10_final \
  --qa-mode reasoning \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --runs 5 \
  --max-new-tokens 4096 \
  --prompt-file outputs/qa_prompt_optimization/reasoning/rpo/openbookqa_reasoning_gemma_rpo/prompt_iteration_10.txt

python -u codes/run_qa_final_test_evaluation.py \
  --code openbookqa_non_reasoning_gemma_rpo10_final \
  --qa-mode non_reasoning \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --runs 5 \
  --max-new-tokens 16 \
  --prompt-file outputs/qa_prompt_optimization/non_reasoning/rpo/openbookqa_non_reasoning_gemma_rpo/prompt_iteration_10.txt
