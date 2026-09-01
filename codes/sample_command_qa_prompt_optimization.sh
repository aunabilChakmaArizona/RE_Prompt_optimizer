#!/usr/bin/env bash

# Generate all 12 first-stage commands (2 modes x 2 models x 3 optimizers).
python -u codes/generate_qa_promptopt_commands.py \
  --phase first_stage \
  --qa-mode all \
  --model-family all \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --output-file codes/qa_promptopt_first_stage_commands.sh

# After first-stage runs finish, generate all 120 second-stage attempts
# (20 first-stage prompt snapshots x 6 refiners).
python -u codes/generate_qa_promptopt_commands.py \
  --phase second_stage \
  --qa-mode all \
  --model-family all \
  --device cuda:0 \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --include-test \
  --output-file codes/qa_promptopt_second_stage_commands.sh

# Summarize validation and test gains.
python -u codes/report_qa_prompt_optimization.py \
  --output-root outputs/qa_prompt_optimization \
  --report-file qa_prompt_optimization_stats.txt
