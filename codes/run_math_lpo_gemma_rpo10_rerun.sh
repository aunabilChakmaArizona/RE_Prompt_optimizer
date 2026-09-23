#!/usr/bin/env bash
set -euo pipefail

# Rerun the two malformed Gemma MATH-500 RPO10 LPO configurations after the
# exact-source parser fix. This script is intended for the VastAI machine.
#
# nohup bash codes/run_math_lpo_gemma_rpo10_rerun.sh \
#   > codes/nohup_outs/math_lpo_gemma_rpo10_parserfix.log 2>&1 &
#
# MATH_GEMMA_LPO_GPU=N changes the physical GPU (default: 0).
# MATH_GEMMA_LPO_DRY_RUN=1 prints both commands without running them.

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

RUN_GPU="${MATH_GEMMA_LPO_GPU:-0}"
DRY_RUN="${MATH_GEMMA_LPO_DRY_RUN:-0}"
OUTPUT_ROOT="outputs/math_prompt_optimization"
SOURCE_PROMPT="experiment_tracking/second_stage/math_gemma_sources/rpo10.txt"
GRADIENT_CACHE_ROOT="outputs/shared_gradient_cache"
SOURCE_CACHE_ROOT="outputs/shared_source_validation_cache"

if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  printf 'MATH_GEMMA_LPO_DRY_RUN must be 0 or 1.\n' >&2
  exit 2
fi

for required_file in \
  "$SOURCE_PROMPT" \
  data/processed/math500/train.jsonl \
  data/processed/math500/validation.jsonl; do
  if [[ ! -s "$required_file" ]]; then
    printf 'Missing required MATH-500 input: %s\n' "$required_file" >&2
    exit 2
  fi
done

run_lpo() {
  # Run one corrected RPO10 LPO configuration or skip its finished result.
  local max_locations="$1"
  local code output_dir
  code="math500_reasoning_gemma_rpo10_lpo_parserfix_f3_s${max_locations}_t3_c5_pool512_vs900_tok4096"
  output_dir="$OUTPUT_ROOT/reasoning/lpo/$code"
  local -a command=(
    python -u codes/run_qa_promptopt_lpo.py
    --code "$code"
    --qa-task math500
    --qa-mode reasoning
    --train-path data/processed/math500/train.jsonl
    --validation-path data/processed/math500/validation.jsonl
    --initial-prompt-file "$SOURCE_PROMPT"
    --model google/gemma-3-4b-it
    --optimizer-model google/gemma-3-12b-it
    --device cuda:0
    --optimizer-device cuda:0
    --target-max-new-tokens 4096
    --optimizer-max-new-tokens 10000
    --validation-std-penalty 1.0
    --validation-fold-size 300
    --gradient-cache-root "$GRADIENT_CACHE_ROOT"
    --source-validation-cache-root "$SOURCE_CACHE_ROOT"
    --seed 42
    --output-root "$OUTPUT_ROOT"
    --overwrite
    --backend vllm
    --gpu-memory-utilization 0.90
    --vllm-max-model-len 32768
    --train-sample-size 512
    --feedback-examples 3
    --max-locations "$max_locations"
    --max-words-per-location 3
    --num-candidates 5
  )

  printf '[math-gemma-lpo-parserfix] S=%s T=3 | GPU=%s | CODE=%s\n' \
    "$max_locations" "$RUN_GPU" "$code" >&2
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$RUN_GPU"
    printf '%q ' "${command[@]}"
    printf '\n'
  elif [[ -f "$output_dir/summary.json" ]]; then
    printf '[math-gemma-lpo-parserfix] already complete; skipping %s\n' "$code" >&2
  else
    CUDA_VISIBLE_DEVICES="$RUN_GPU" "${command[@]}"
  fi
}

run_lpo 5
run_lpo 7
