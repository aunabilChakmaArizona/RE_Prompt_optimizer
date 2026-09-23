#!/usr/bin/env bash
set -euo pipefail

# Eight OpenBookQA GradPO-Gen larger-edit runs:
#   two models x two RPO sources x T={3,5}
# Fixed settings: S=7, H=0.45, C=7, B=5, F=0.50, seed=42.
# Qwen uses pool=800/G=300; Gemma uses pool=600/G=200.
#
# Activate re_prompt_optimization_vllm_v2 before launching.
# Example:
#   nohup bash codes/run_openbookqa_gradpo_s7_vastai.sh \
#     > codes/nohup_outs/openbookqa_gradpo_s7_vastai.log 2>&1 &
#
# Optional controls:
#   OPENBOOKQA_S7_GPU=0
#   OPENBOOKQA_S7_START=1
#   OPENBOOKQA_S7_END=8
#   OPENBOOKQA_S7_DRY_RUN=1

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

RUN_GPU="${OPENBOOKQA_S7_GPU:-0}"
START="${OPENBOOKQA_S7_START:-1}"
END="${OPENBOOKQA_S7_END:-8}"
DRY_RUN="${OPENBOOKQA_S7_DRY_RUN:-0}"
SOURCE_ROOT="experiment_tracking/second_stage/openbookqa_sources"
OUTPUT_ROOT="outputs/qa_prompt_optimization"
CACHE_ROOT="outputs/vastai_openbookqa_s7_cache"

if [[ ! "$START" =~ ^[1-8]$ || ! "$END" =~ ^[1-8]$ || "$START" -gt "$END" ]]; then
  printf 'OPENBOOKQA_S7_START/END must be 1-8 with START <= END.\n' >&2
  exit 2
fi
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  printf 'OPENBOOKQA_S7_DRY_RUN must be 0 or 1.\n' >&2
  exit 2
fi

for required_file in \
  "$SOURCE_ROOT/qwen/rpo5_latest.txt" \
  "$SOURCE_ROOT/qwen/rpo10_latest.txt" \
  "$SOURCE_ROOT/gemma/rpo5_latest.txt" \
  "$SOURCE_ROOT/gemma/rpo10_latest.txt" \
  data/processed/openbookqa/train.jsonl \
  data/processed/openbookqa/validation.jsonl; do
  if [[ ! -s "$required_file" ]]; then
    printf 'Missing required OpenBookQA input: %s\n' "$required_file" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES="$RUN_GPU"

run_attempt() {
  # Run one selected configuration or skip its completed output on restart.
  local attempt="$1" code="$2"
  shift 2
  local output_dir="$OUTPUT_ROOT/non_reasoning/gradpo_gen/$code"
  printf '[openbookqa-s7] attempt %s/8 | CODE=%s\n' "$attempt" "$code" >&2
  if [[ "$attempt" -lt "$START" || "$attempt" -gt "$END" ]]; then
    printf '[openbookqa-s7] outside selected range; skipping attempt %s\n' \
      "$attempt" >&2
  elif [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "$@"
    printf '\n'
  elif [[ -f "$output_dir/summary.json" ]]; then
    printf '[openbookqa-s7] already complete; skipping %s\n' "$code" >&2
  else
    "$@"
  fi
}

run_gradpo() {
  # Build one model-specific S=7 GradPO-Gen experiment.
  local attempt="$1" family="$2" source_name="$3" source_prompt="$4" max_tokens="$5"
  local model train_pool gradient_size code
  case "$family" in
    qwen)
      model="Qwen/Qwen3-4B"
      train_pool=800
      gradient_size=300
      ;;
    gemma)
      model="google/gemma-3-4b-it"
      train_pool=600
      gradient_size=200
      ;;
    *)
      printf 'Unknown model family: %s\n' "$family" >&2
      exit 2
      ;;
  esac

  code="openbookqa_non_reasoning_${family}_${source_name}_gradpo_gen_s7_t${max_tokens}_h045_c7_g${gradient_size}_b5_f050_vastai"
  run_attempt "$attempt" "$code" \
    python -u codes/run_qa_promptopt_gradpo.py \
    --code "$code" \
    --qa-task openbookqa \
    --qa-mode non_reasoning \
    --train-path data/processed/openbookqa/train.jsonl \
    --validation-path data/processed/openbookqa/validation.jsonl \
    --initial-prompt-file "$source_prompt" \
    --model "$model" \
    --device cuda:0 \
    --hf-device cuda:0 \
    --target-max-new-tokens 10 \
    --validation-std-penalty 1.0 \
    --gradient-cache-root "$CACHE_ROOT/gradient" \
    --source-validation-cache-root "$CACHE_ROOT/source_validation" \
    --output-root "$OUTPUT_ROOT" \
    --overwrite \
    --backend dual \
    --objective-scoring-backend transformers \
    --final-evaluation-backend vllm \
    --dual-vllm-gpu-memory-utilization 0.50 \
    --vllm-max-model-len 16384 \
    --variant gen \
    --train-sample-size "$train_pool" \
    --gradient-sample-size "$gradient_size" \
    --gradient-batch-size 2 \
    --selection-batch-size 4 \
    --num-edit-regions 7 \
    --max-region-tokens "$max_tokens" \
    --region-expansion-threshold 0.45 \
    --num-region-candidates 7 \
    --beam-width 5 \
    --beam-replacement-mode llm_synthesis \
    --fluency-lambda 0.5 \
    --candidate-max-new-tokens 10000 \
    --synthesis-max-new-tokens 10000 \
    --synthesis-batch-size 4 \
    --seed 42
}

attempt=0
for family in qwen gemma; do
  for source_name in rpo5 rpo10; do
    source_prompt="$SOURCE_ROOT/$family/${source_name}_latest.txt"
    for max_tokens in 3 5; do
      attempt=$((attempt + 1))
      run_gradpo "$attempt" "$family" "$source_name" "$source_prompt" "$max_tokens"
    done
  done
done

printf '[openbookqa-s7] selected attempts completed.\n' >&2
