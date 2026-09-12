#!/usr/bin/env bash
set -euo pipefail

# Lightweight preflight for the gradient path. Change GPU before running.
# This uses 3 validation examples and 2 training examples; it is not a result run.
SMOKE_GPU="${MATH_SMOKE_GPU:-2}"

CUDA_VISIBLE_DEVICES="$SMOKE_GPU" python -u codes/run_qa_promptopt_greater.py \
  --code math500_qwen_greater_tg_smoke \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --backend transformers \
  --final-evaluation-backend transformers \
  --target-max-new-tokens 256 \
  --validation-fold-size 1 \
  --validation-std-penalty 1.0 \
  --train-sample-size 2 \
  --gradient-batch-size 1 \
  --selection-batch-size 1 \
  --variant greater_tg \
  --proposal-top-k 2 \
  --proposal-example-size 2 \
  --proposal-min-candidates 2 \
  --selection-top-mu 2 \
  --top-u 1 \
  --output-root outputs/math_prompt_optimization_smoke \
  --overwrite

CUDA_VISIBLE_DEVICES="$SMOKE_GPU" python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_qwen_gradpo_gen_smoke \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --backend transformers \
  --final-evaluation-backend transformers \
  --target-max-new-tokens 256 \
  --validation-fold-size 1 \
  --validation-std-penalty 1.0 \
  --train-sample-size 2 \
  --gradient-batch-size 1 \
  --selection-batch-size 1 \
  --variant gen \
  --num-edit-regions 1 \
  --max-region-tokens 2 \
  --num-region-candidates 1 \
  --beam-width 1 \
  --beam-replacement-mode llm_synthesis \
  --candidate-max-new-tokens 256 \
  --synthesis-max-new-tokens 256 \
  --synthesis-batch-size 1 \
  --output-root outputs/math_prompt_optimization_smoke \
  --overwrite

test -f outputs/math_prompt_optimization_smoke/reasoning/greater_tg/math500_qwen_greater_tg_smoke/gradient_analysis.json
test -f outputs/math_prompt_optimization_smoke/reasoning/greater_tg/math500_qwen_greater_tg_smoke/selected_spans.json
test -f outputs/math_prompt_optimization_smoke/reasoning/gradpo_gen/math500_qwen_gradpo_gen_smoke/gradient_analysis.json
test -f outputs/math_prompt_optimization_smoke/reasoning/gradpo_gen/math500_qwen_gradpo_gen_smoke/beam_trace.json

echo "Math gradient smoke tests completed and required artifacts were saved."
