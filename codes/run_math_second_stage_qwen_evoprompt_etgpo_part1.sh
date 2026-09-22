#!/usr/bin/env bash
set -euo pipefail

# Qwen MATH-500 EvoPrompt/ETGPO second stage, attempts 1-9 of 18.
# Part 1 runs all six refiners on EvoPrompt-5, followed by the three
# GradPO variants on EvoPrompt-10. Validation remains 3 folds x 300 examples.
# Activate re_prompt_optimization_vllm_v2 before launching.
#
# nohup bash codes/run_math_second_stage_qwen_evoprompt_etgpo_part1.sh \
#   > codes/nohup_outs/math_second_stage_qwen_evoprompt_etgpo_part1.log 2>&1 &
#
# MATH_SECOND_STAGE_GPU=N changes the physical GPU (default: 1).
# MATH_SECOND_STAGE_DRY_RUN=1 prints commands without running them.

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

RUN_GPU="${MATH_SECOND_STAGE_GPU:-1}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  GPU_LABEL="Slurm-assigned"
else
  GPU_LABEL="$RUN_GPU"
fi
DRY_RUN="${MATH_SECOND_STAGE_DRY_RUN:-0}"
OUTPUT_ROOT="outputs/math_prompt_optimization"
GRADIENT_CACHE_ROOT="outputs/shared_gradient_cache"
SOURCE_CACHE_ROOT="outputs/shared_source_validation_cache"

EVOPROMPT5_SOURCE="outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_experimental_iteration_5_from_actual_iteration_1.txt"
EVOPROMPT10_SOURCE="outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_experimental_iteration_10_from_actual_iteration_2.txt"

if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
  printf 'MATH_SECOND_STAGE_DRY_RUN must be 0 or 1.\n' >&2
  exit 2
fi

for required_file in \
  "$EVOPROMPT5_SOURCE" \
  "$EVOPROMPT10_SOURCE" \
  data/processed/math500/train.jsonl \
  data/processed/math500/validation.jsonl; do
  if [[ ! -s "$required_file" ]]; then
    printf 'Missing required MATH-500 input: %s\n' "$required_file" >&2
    exit 2
  fi
done

run_command() {
  # Run one configured attempt, or skip it when its summary already exists.
  local attempt="$1" method="$2" code="$3"
  shift 3
  local output_dir="$OUTPUT_ROOT/reasoning/$method/$code"
  printf '[math-qwen-evo-etgpo] attempt %s/18 | method=%s | GPU=%s | CODE=%s\n' \
    "$attempt" "$method" "$GPU_LABEL" "$code" >&2
  if [[ "$DRY_RUN" == 1 ]]; then
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
      printf 'CUDA_VISIBLE_DEVICES=%q ' "$RUN_GPU"
    fi
    printf '%q ' "$@"
    printf '\n'
  elif [[ -f "$output_dir/summary.json" ]]; then
    printf '[math-qwen-evo-etgpo] already complete; skipping %s\n' "$code" >&2
  elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    "$@"
  else
    CUDA_VISIBLE_DEVICES="$RUN_GPU" "$@"
  fi
}

run_refiner() {
  # Build one refiner command with the selected Qwen RPO hyperparameters.
  local attempt="$1" source_name="$2" source_prompt="$3" method="$4"
  local code variant
  local -a common_args command

  common_args=(
    --qa-task math500
    --qa-mode reasoning
    --train-path data/processed/math500/train.jsonl
    --validation-path data/processed/math500/validation.jsonl
    --initial-prompt-file "$source_prompt"
    --model Qwen/Qwen3-4B
    --device cuda:0
    --target-max-new-tokens 4096
    --validation-std-penalty 1.0
    --validation-fold-size 300
    --gradient-cache-root "$GRADIENT_CACHE_ROOT"
    --source-validation-cache-root "$SOURCE_CACHE_ROOT"
    --seed 42
    --output-root "$OUTPUT_ROOT"
    --overwrite
  )

  case "$method" in
    gradpo_gen|gradpo_prob|gradpo_gen_random)
      case "$method" in
        gradpo_gen) variant="gen" ;;
        gradpo_prob) variant="prob" ;;
        gradpo_gen_random) variant="gen_random" ;;
      esac
      code="math500_reasoning_qwen_${source_name}_${method}_final_s3_t3_h030_c7_g200_b5_f050_pool1600_vs900_tok4096"
      command=(
        python -u codes/run_qa_promptopt_gradpo.py
        --code "$code"
        "${common_args[@]}"
        --hf-device cuda:0
        --backend dual
        --objective-scoring-backend vllm
        --objective-scoring-batch-size 128
        --final-evaluation-backend vllm
        --gpu-memory-utilization 0.50
        --dual-vllm-gpu-memory-utilization 0.50
        --vllm-max-model-len 32768
        --variant "$variant"
        --train-sample-size 1600
        --gradient-sample-size 200
        --gradient-batch-size 1
        --selection-batch-size 4
        --num-edit-regions 3
        --max-region-tokens 3
        --region-expansion-threshold 0.30
        --num-region-candidates 7
        --beam-width 5
        --beam-replacement-mode llm_synthesis
        --fluency-lambda 0.5
        --candidate-max-new-tokens 10000
        --synthesis-max-new-tokens 10000
        --synthesis-batch-size 4
      )
      ;;
    lpo)
      code="math500_reasoning_qwen_${source_name}_lpo_final_f3_s3_t3_c5_pool512_vs900_tok4096"
      command=(
        python -u codes/run_qa_promptopt_lpo.py
        --code "$code"
        "${common_args[@]}"
        --optimizer-model Qwen/Qwen3-14B
        --optimizer-device cuda:0
        --optimizer-max-new-tokens 10000
        --backend vllm
        --gpu-memory-utilization 0.90
        --vllm-max-model-len 32768
        --train-sample-size 512
        --feedback-examples 3
        --max-locations 3
        --max-words-per-location 3
        --num-candidates 5
      )
      ;;
    greater|greater_tg)
      code="math500_reasoning_qwen_${source_name}_${method}_final_g200_topu5_pool1600_vs900_tok4096"
      command=(
        python -u codes/run_qa_promptopt_greater.py
        --code "$code"
        "${common_args[@]}"
        --hf-device cuda:0
        --backend dual
        --objective-scoring-backend vllm
        --objective-scoring-batch-size 128
        --final-evaluation-backend vllm
        --gpu-memory-utilization 0.50
        --dual-vllm-gpu-memory-utilization 0.50
        --vllm-max-model-len 32768
        --variant "$method"
        --train-sample-size 1600
        --gradient-sample-size 200
        --gradient-batch-size 1
        --selection-batch-size 4
        --proposal-top-k 25
        --proposal-example-size 50
        --proposal-min-candidates 10
        --selection-top-mu 10
        --top-u 5
        --fluency-lambda 0.2
        --region-expansion-threshold 0.6
      )
      ;;
    *)
      printf 'Unknown second-stage method: %s\n' "$method" >&2
      exit 2
      ;;
  esac

  run_command "$attempt" "$method" "$code" "${command[@]}"
}

attempt=0
for method in gradpo_gen gradpo_prob gradpo_gen_random lpo greater greater_tg; do
  attempt=$((attempt + 1))
  run_refiner "$attempt" "evoprompt5" "$EVOPROMPT5_SOURCE" "$method"
done

for method in gradpo_gen gradpo_prob gradpo_gen_random; do
  attempt=$((attempt + 1))
  run_refiner "$attempt" "evoprompt10" "$EVOPROMPT10_SOURCE" "$method"
done
