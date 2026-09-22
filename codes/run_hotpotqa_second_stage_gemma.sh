#!/usr/bin/env bash
set -euo pipefail

# HotpotQA reasoning: transferred final OpenBookQA settings for Gemma3-4B.
# The default ready set runs RPO-5, RPO-10, and ETGPO-1: 18 runs. After the
# current EvoPrompt rerun is selected and copied into the tracked source folder,
# use HOTPOT_GEMMA_SOURCE_SET=evoprompt for its 12 missing refinements.
# Gemma 3 image support intentionally stays enabled for vLLM 0.11.0.
#
# Local example:
#   HOTPOT_GEMMA_GPU=3 nohup bash codes/run_hotpotqa_second_stage_gemma.sh \
#     > codes/nohup_outs/hotpotqa_second_stage_gemma.log 2>&1 &
# Preview or resume:
#   HOTPOT_GEMMA_DRY_RUN=1 bash codes/run_hotpotqa_second_stage_gemma.sh
#   HOTPOT_GEMMA_START=7 HOTPOT_GEMMA_END=18 bash codes/run_hotpotqa_second_stage_gemma.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/.."

SOURCE_SET="${HOTPOT_GEMMA_SOURCE_SET:-ready}"
START="${HOTPOT_GEMMA_START:-1}"
DRY_RUN="${HOTPOT_GEMMA_DRY_RUN:-0}"
VLLM_RATIO="${HOTPOT_GEMMA_VLLM_RATIO:-0.50}"
OUTPUT_ROOT="outputs/qa_prompt_optimization"

# Slurm owns CUDA_VISIBLE_DEVICES. A local run defaults to physical GPU 3.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${HOTPOT_GEMMA_GPU:-3}"
fi

SOURCE_DIR="experiment_tracking/second_stage/hotpotqa_sources/gemma"

case "$SOURCE_SET" in
  ready)
    SOURCES=(
      "rpo5|$SOURCE_DIR/rpo5.txt"
      "rpo10|$SOURCE_DIR/rpo10.txt"
      "etgpo1|$SOURCE_DIR/etgpo1.txt"
    )
    ;;
  evoprompt)
    SOURCES=(
      "evoprompt5|$SOURCE_DIR/evoprompt5.txt"
      "evoprompt10|$SOURCE_DIR/evoprompt10.txt"
    )
    ;;
  full)
    SOURCES=(
      "rpo5|$SOURCE_DIR/rpo5.txt"
      "rpo10|$SOURCE_DIR/rpo10.txt"
      "evoprompt5|$SOURCE_DIR/evoprompt5.txt"
      "evoprompt10|$SOURCE_DIR/evoprompt10.txt"
      "etgpo1|$SOURCE_DIR/etgpo1.txt"
    )
    ;;
  *)
    printf 'HOTPOT_GEMMA_SOURCE_SET must be ready, evoprompt, or full.\n' >&2
    exit 2
    ;;
esac

TOTAL_ATTEMPTS=$((${#SOURCES[@]} * 6))
END="${HOTPOT_GEMMA_END:-$TOTAL_ATTEMPTS}"
if [[ ! "$START" =~ ^[0-9]+$ || ! "$END" =~ ^[0-9]+$ \
   || "$START" -lt 1 || "$END" -gt "$TOTAL_ATTEMPTS" \
   || "$START" -gt "$END" ]]; then
  printf 'HOTPOT_GEMMA_START/END must be within 1-%s with START <= END.\n' \
    "$TOTAL_ATTEMPTS" >&2
  exit 2
fi
if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  printf 'HOTPOT_GEMMA_DRY_RUN must be 0 or 1.\n' >&2
  exit 2
fi

for source_spec in "${SOURCES[@]}"; do
  IFS='|' read -r _ source_prompt <<< "$source_spec"
  if [[ ! -s "$source_prompt" ]]; then
    printf 'Missing HotpotQA source prompt: %s\n' "$source_prompt" >&2
    exit 2
  fi
done

run_attempt() {
  # Execute one selected attempt, while supporting dry runs and safe restarts.
  local attempt="$1" method="$2" code="$3"
  shift 3
  local output_dir="$OUTPUT_ROOT/reasoning/$method/$code"
  printf '[hotpot-gemma] attempt %s/%s | method=%s | CODE=%s\n' \
    "$attempt" "$TOTAL_ATTEMPTS" "$method" "$code" >&2
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "$@"
    printf '\n'
  elif [[ -f "$output_dir/summary.json" ]]; then
    printf '[hotpot-gemma] already complete; skipping %s\n' "$code" >&2
  else
    "$@"
  fi
}

attempt=0
for source_spec in "${SOURCES[@]}"; do
  IFS='|' read -r source_name source_prompt <<< "$source_spec"
  common_args=(
    --qa-task hotpotqa
    --qa-mode reasoning
    --train-path data/processed/hotpotqa/train.jsonl
    --validation-path data/processed/hotpotqa/validation.jsonl
    --validation-fold-size 500
    --initial-prompt-file "$source_prompt"
    --model google/gemma-3-4b-it
    --device cuda:0
    --target-max-new-tokens 4096
    --validation-std-penalty 1.0
    --seed 42
    --output-root "$OUTPUT_ROOT"
    --overwrite
  )

  for variant in gen prob gen_random; do
    attempt=$((attempt + 1))
    method="gradpo_${variant}"
    code="hotpotqa_reasoning_gemma_${source_name}_${method}_transfer_s5_t3_h060_c7_g200_b5_f050_pool500_vs1500_tok4096"
    command=(
      python -u codes/run_qa_promptopt_gradpo.py
      --code "$code"
      "${common_args[@]}"
      --hf-device cuda:0
      --backend dual
      --objective-scoring-backend vllm
      --objective-scoring-batch-size 128
      --final-evaluation-backend vllm
      --gpu-memory-utilization "$VLLM_RATIO"
      --dual-vllm-gpu-memory-utilization "$VLLM_RATIO"
      --vllm-max-model-len 32768
      --variant "$variant"
      --train-sample-size 500
      --gradient-sample-size 200
      --gradient-batch-size 2
      --selection-batch-size 4
      --num-edit-regions 5
      --max-region-tokens 3
      --region-expansion-threshold 0.60
      --num-region-candidates 7
      --beam-width 5
      --beam-replacement-mode llm_synthesis
      --fluency-lambda 0.5
      --candidate-max-new-tokens 10000
      --synthesis-max-new-tokens 10000
      --synthesis-batch-size 4
    )
    if [[ "$attempt" -ge "$START" && "$attempt" -le "$END" ]]; then
      run_attempt "$attempt" "$method" "$code" "${command[@]}"
    fi
  done

  attempt=$((attempt + 1))
  method="lpo"
  code="hotpotqa_reasoning_gemma_${source_name}_lpo_transfer_f5_s3_t3_c5_pool500_vs1500_tok4096"
  command=(
    python -u codes/run_qa_promptopt_lpo.py
    --code "$code"
    "${common_args[@]}"
    --optimizer-model google/gemma-3-12b-it
    --optimizer-device cuda:0
    --optimizer-max-new-tokens 10000
    --backend vllm
    --gpu-memory-utilization 0.90
    --vllm-max-model-len 32768
    --train-sample-size 500
    --feedback-examples 5
    --max-locations 3
    --max-words-per-location 3
    --num-candidates 5
  )
  if [[ "$attempt" -ge "$START" && "$attempt" -le "$END" ]]; then
    run_attempt "$attempt" "$method" "$code" "${command[@]}"
  fi

  for variant in greater greater_tg; do
    attempt=$((attempt + 1))
    method="$variant"
    code="hotpotqa_reasoning_gemma_${source_name}_${variant}_transfer_g300_topmu10_topu5_pool500_vs1500_tok4096"
    command=(
      python -u codes/run_qa_promptopt_greater.py
      --code "$code"
      "${common_args[@]}"
      --hf-device cuda:0
      --backend dual
      --objective-scoring-backend vllm
      --objective-scoring-batch-size 128
      --final-evaluation-backend vllm
      --gpu-memory-utilization "$VLLM_RATIO"
      --dual-vllm-gpu-memory-utilization "$VLLM_RATIO"
      --vllm-max-model-len 32768
      --variant "$variant"
      --train-sample-size 500
      --gradient-sample-size 300
      --gradient-batch-size 2
      --selection-batch-size 4
      --proposal-top-k 25
      --proposal-example-size 50
      --proposal-min-candidates 10
      --selection-top-mu 10
      --top-u 5
      --fluency-lambda 0.2
      --region-expansion-threshold 0.6
    )
    if [[ "$attempt" -ge "$START" && "$attempt" -le "$END" ]]; then
      run_attempt "$attempt" "$method" "$code" "${command[@]}"
    fi
  done
done

printf '[hotpot-gemma] selected attempts completed: %s-%s of %s\n' \
  "$START" "$END" "$TOTAL_ATTEMPTS"
