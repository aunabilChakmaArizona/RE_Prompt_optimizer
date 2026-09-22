#!/usr/bin/env bash
set -euo pipefail

# Ten sequential GradPO-Gen attempts: five S/T/H settings x RPO5/RPO10.
# Activate re_prompt_optimization_vllm_v2 first. No test evaluation is enabled.
# On Slurm, preserve the scheduler's CUDA_VISIBLE_DEVICES.
# Preview: MATH_TUNING_DRY_RUN=1 bash codes/run_math_gradpo_tuning_gemma_s3_t3.sh
# Resume: MATH_TUNING_START=3 bash codes/run_math_gradpo_tuning_gemma_s3_t3.sh
# Gemma 3 with vLLM 0.11.0 must keep image support enabled.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/.."

START="${MATH_TUNING_START:-1}"
END="${MATH_TUNING_END:-10}"
DRY_RUN="${MATH_TUNING_DRY_RUN:-0}"
VLLM_RATIO="${MATH_TUNING_VLLM_RATIO:-0.50}"
SOURCE_DIR="experiment_tracking/second_stage/math_gemma_sources"
CACHE_ROOT="outputs/deltaai_gemma_vllm011_images_enabled_cache"

if [[ ! "$START" =~ ^([1-9]|10)$ || ! "$END" =~ ^([1-9]|10)$ || "$START" -gt "$END" ]]; then
  printf 'MATH_TUNING_START/END must be 1-10, with START <= END.\n' >&2
  exit 2
fi
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  printf 'MATH_TUNING_DRY_RUN must be 0 or 1.\n' >&2
  exit 2
fi
if [[ -n "${MATH_TUNING_GPU:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    printf 'Unset MATH_TUNING_GPU on Slurm; use the scheduler-assigned GPU.\n' >&2
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES="$MATH_TUNING_GPU"
fi

for source in "$SOURCE_DIR/rpo5.txt" "$SOURCE_DIR/rpo10.txt"; do
  if [[ ! -s "$source" ]]; then
    printf 'Missing source prompt: %s\n' "$source" >&2
    exit 2
  fi
done
if [[ "$DRY_RUN" == 0 ]]; then
  for dataset in data/processed/math500/train.jsonl data/processed/math500/validation.jsonl; do
    if [[ ! -s "$dataset" ]]; then
      printf 'Missing dataset: %s; extract promptopt_data.zip in the project root.\n' "$dataset" >&2
      exit 2
    fi
  done
fi

run_attempt() {
  # Run one fixed-budget GradPO-Gen configuration on one source prompt.
  local attempt="$1" source="$2" spans="$3" tokens="$4" threshold="$5"
  local h_tag="${threshold//./}"
  local code="math500_reasoning_gemma_rpo${source}_gradpo_tune_s${spans}_t${tokens}_h${h_tag}_c7_g200_b5_f050_pool600_vs900_tok4096_v3_images_enabled"
  local -a command=(
    python -u codes/run_qa_promptopt_gradpo.py
    --code "$code"
    --qa-task math500
    --qa-mode reasoning
    --train-path data/processed/math500/train.jsonl
    --validation-path data/processed/math500/validation.jsonl
    --initial-prompt-file "$SOURCE_DIR/rpo${source}.txt"
    --model google/gemma-3-4b-it
    --device cuda:0
    --hf-device cuda:0
    --backend dual
    --objective-scoring-backend vllm
    --objective-scoring-batch-size 128
    --final-evaluation-backend vllm
    --gpu-memory-utilization "$VLLM_RATIO"
    --dual-vllm-gpu-memory-utilization "$VLLM_RATIO"
    --vllm-max-model-len 32768
    --target-max-new-tokens 4096
    --validation-std-penalty 1.0
    --validation-fold-size 300
    --variant gen
    --train-sample-size 600
    --gradient-sample-size 200
    --gradient-batch-size 1
    --selection-batch-size 4
    --num-edit-regions "$spans"
    --max-region-tokens "$tokens"
    --region-expansion-threshold "$threshold"
    --num-region-candidates 7
    --beam-width 5
    --fluency-lambda 0.5
    --beam-replacement-mode llm_synthesis
    --candidate-max-new-tokens 10000
    --synthesis-max-new-tokens 10000
    --synthesis-batch-size 4
    --gradient-cache-root "$CACHE_ROOT/gradient"
    --source-validation-cache-root "$CACHE_ROOT/source_validation"
    --seed 42
    --output-root outputs/math_prompt_optimization
    --overwrite
  )
  printf '[math-gemma-small-edits] attempt %s/10 | source=RPO%s | S=%s T=%s H=%s | CODE=%s\n' \
    "$attempt" "$source" "$spans" "$tokens" "$threshold" "$code" >&2
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    "${command[@]}"
  fi
}

attempt=0
for configuration in "3 3 0.30" "3 3 0.45" "3 3 0.60" "5 3 0.30" "5 3 0.45"; do
  read -r spans tokens threshold <<< "$configuration"
  for source in 5 10; do
    attempt=$((attempt + 1))
    if [[ "$attempt" -ge "$START" && "$attempt" -le "$END" ]]; then
      run_attempt "$attempt" "$source" "$spans" "$tokens" "$threshold"
    fi
  done
done
