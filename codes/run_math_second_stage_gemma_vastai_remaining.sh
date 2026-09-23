#!/usr/bin/env bash
set -euo pipefail

# Run the 21 remaining Gemma MATH-500 second-stage validation experiments:
#   1 parser-fixed RPO5 LPO run
#   2 RPO5/RPO10 GreaTer-TG runs
#   18 runs over EvoPrompt-5, EvoPrompt-10, and ETGPO-1 (six refiners each)
#
# Activate re_prompt_optimization_vllm_v2 before launching.
# Example:
#   nohup bash codes/run_math_second_stage_gemma_vastai_remaining.sh \
#     > codes/nohup_outs/math_second_stage_gemma_vastai_remaining.log 2>&1 &
#
# Optional controls:
#   MATH_GEMMA_VASTAI_GPU=0       physical GPU visible to this script
#   MATH_GEMMA_VASTAI_START=1     first attempt to run (1-21)
#   MATH_GEMMA_VASTAI_END=21      last attempt to run (1-21)
#   MATH_GEMMA_VASTAI_DRY_RUN=1   print commands without running them

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

RUN_GPU="${MATH_GEMMA_VASTAI_GPU:-0}"
START="${MATH_GEMMA_VASTAI_START:-1}"
END="${MATH_GEMMA_VASTAI_END:-21}"
DRY_RUN="${MATH_GEMMA_VASTAI_DRY_RUN:-0}"
VLLM_RATIO="${MATH_GEMMA_VASTAI_VLLM_RATIO:-0.50}"
SOURCE_DIR="experiment_tracking/second_stage/math_gemma_sources"
OUTPUT_ROOT="outputs/math_prompt_optimization"
CACHE_ROOT="outputs/vastai_gemma_math_remaining_cache"

if [[ ! "$START" =~ ^([1-9]|1[0-9]|2[01])$ \
   || ! "$END" =~ ^([1-9]|1[0-9]|2[01])$ \
   || "$START" -gt "$END" ]]; then
  printf 'MATH_GEMMA_VASTAI_START/END must be 1-21 with START <= END.\n' >&2
  exit 2
fi
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  printf 'MATH_GEMMA_VASTAI_DRY_RUN must be 0 or 1.\n' >&2
  exit 2
fi

for required_file in \
  "$SOURCE_DIR/rpo5.txt" \
  "$SOURCE_DIR/rpo10.txt" \
  "$SOURCE_DIR/evoprompt5.txt" \
  "$SOURCE_DIR/evoprompt10.txt" \
  "$SOURCE_DIR/etgpo1.txt" \
  data/processed/math500/train.jsonl \
  data/processed/math500/validation.jsonl; do
  if [[ ! -s "$required_file" ]]; then
    printf 'Missing required Gemma MATH-500 input: %s\n' "$required_file" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES="$RUN_GPU"

run_attempt() {
  # Run one selected attempt and skip a completed output during restart.
  local attempt="$1" method="$2" code="$3"
  shift 3
  local output_dir="$OUTPUT_ROOT/reasoning/$method/$code"
  printf '[math-gemma-vastai] attempt %s/21 | method=%s | CODE=%s\n' \
    "$attempt" "$method" "$code" >&2
  if [[ "$attempt" -lt "$START" || "$attempt" -gt "$END" ]]; then
    printf '[math-gemma-vastai] outside selected range; skipping attempt %s\n' \
      "$attempt" >&2
  elif [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "$@"
    printf '\n'
  elif [[ -f "$output_dir/summary.json" ]]; then
    printf '[math-gemma-vastai] already complete; skipping %s\n' "$code" >&2
  else
    "$@"
  fi
}

common_args_for_source() {
  # Populate COMMON_ARGS for one portable first-stage source prompt.
  local source_prompt="$1"
  COMMON_ARGS=(
    --qa-task math500
    --qa-mode reasoning
    --train-path data/processed/math500/train.jsonl
    --validation-path data/processed/math500/validation.jsonl
    --initial-prompt-file "$source_prompt"
    --model google/gemma-3-4b-it
    --device cuda:0
    --target-max-new-tokens 4096
    --validation-std-penalty 1.0
    --validation-fold-size 300
    --gradient-cache-root "$CACHE_ROOT/gradient"
    --source-validation-cache-root "$CACHE_ROOT/source_validation"
    --seed 42
    --output-root "$OUTPUT_ROOT"
    --overwrite
  )
}

run_gradpo() {
  # Run one GradPO variant with the selected shared Gemma configuration.
  local attempt="$1" source_name="$2" source_prompt="$3" variant="$4"
  local method="gradpo_${variant}"
  local code="math500_reasoning_gemma_${source_name}_${method}_final_s7_t5_h045_c7_g200_b5_f050_pool600_vs900_tok4096_vastai"
  common_args_for_source "$source_prompt"
  run_attempt "$attempt" "$method" "$code" \
    python -u codes/run_qa_promptopt_gradpo.py \
    --code "$code" "${COMMON_ARGS[@]}" \
    --hf-device cuda:0 \
    --backend dual \
    --objective-scoring-backend vllm \
    --objective-scoring-batch-size 128 \
    --final-evaluation-backend vllm \
    --gpu-memory-utilization "$VLLM_RATIO" \
    --dual-vllm-gpu-memory-utilization "$VLLM_RATIO" \
    --vllm-max-model-len 32768 \
    --variant "$variant" \
    --train-sample-size 600 \
    --gradient-sample-size 200 \
    --gradient-batch-size 1 \
    --selection-batch-size 4 \
    --num-edit-regions 7 \
    --max-region-tokens 5 \
    --region-expansion-threshold 0.45 \
    --num-region-candidates 7 \
    --beam-width 5 \
    --beam-replacement-mode llm_synthesis \
    --fluency-lambda 0.5 \
    --candidate-max-new-tokens 10000 \
    --synthesis-max-new-tokens 10000 \
    --synthesis-batch-size 4
}

run_lpo() {
  # Run LPO with three mistakes, at most five spans, and three words per span.
  local attempt="$1" source_name="$2" source_prompt="$3"
  local suffix="${4:-final}"
  local method="lpo"
  local code="math500_reasoning_gemma_${source_name}_lpo_${suffix}_f3_s5_t3_c5_pool512_vs900_tok4096_vastai"
  common_args_for_source "$source_prompt"
  run_attempt "$attempt" "$method" "$code" \
    python -u codes/run_qa_promptopt_lpo.py \
    --code "$code" "${COMMON_ARGS[@]}" \
    --optimizer-model google/gemma-3-12b-it \
    --optimizer-device cuda:0 \
    --optimizer-max-new-tokens 10000 \
    --backend vllm \
    --gpu-memory-utilization 0.90 \
    --vllm-max-model-len 32768 \
    --train-sample-size 512 \
    --feedback-examples 3 \
    --max-locations 5 \
    --max-words-per-location 3 \
    --num-candidates 5
}

run_greater() {
  # Run GreaTer or GreaTer-TG with the selected 300-example gradient subset.
  local attempt="$1" source_name="$2" source_prompt="$3" variant="$4"
  local method="$variant"
  local code="math500_reasoning_gemma_${source_name}_${method}_final_g300_topu5_pool600_vs900_tok4096_vastai"
  common_args_for_source "$source_prompt"
  run_attempt "$attempt" "$method" "$code" \
    python -u codes/run_qa_promptopt_greater.py \
    --code "$code" "${COMMON_ARGS[@]}" \
    --hf-device cuda:0 \
    --backend dual \
    --objective-scoring-backend vllm \
    --objective-scoring-batch-size 128 \
    --final-evaluation-backend vllm \
    --gpu-memory-utilization "$VLLM_RATIO" \
    --dual-vllm-gpu-memory-utilization "$VLLM_RATIO" \
    --vllm-max-model-len 32768 \
    --variant "$variant" \
    --train-sample-size 600 \
    --gradient-sample-size 300 \
    --gradient-batch-size 1 \
    --selection-batch-size 4 \
    --proposal-top-k 25 \
    --proposal-example-size 50 \
    --proposal-min-candidates 10 \
    --selection-top-mu 10 \
    --top-u 5 \
    --fluency-lambda 0.2 \
    --region-expansion-threshold 0.6
}

# Outstanding RPO-source runs.
run_lpo 1 rpo5 "$SOURCE_DIR/rpo5.txt" parserfix
run_greater 2 rpo5 "$SOURCE_DIR/rpo5.txt" greater_tg
run_greater 3 rpo10 "$SOURCE_DIR/rpo10.txt" greater_tg

# Complete six-refiner matrices for the three remaining first-stage sources.
attempt=3
for source_name in evoprompt5 evoprompt10 etgpo1; do
  source_prompt="$SOURCE_DIR/${source_name}.txt"
  for variant in gen prob gen_random; do
    attempt=$((attempt + 1))
    run_gradpo "$attempt" "$source_name" "$source_prompt" "$variant"
  done
  attempt=$((attempt + 1))
  run_lpo "$attempt" "$source_name" "$source_prompt"
  attempt=$((attempt + 1))
  run_greater "$attempt" "$source_name" "$source_prompt" greater
  attempt=$((attempt + 1))
  run_greater "$attempt" "$source_name" "$source_prompt" greater_tg
done

printf '[math-gemma-vastai] selected attempts completed.\n' >&2
