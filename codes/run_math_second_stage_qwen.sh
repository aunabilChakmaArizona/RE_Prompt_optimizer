#!/usr/bin/env bash
set -euo pipefail

# Default: eight Qwen Math sensitivity runs on RPO5/RPO10:
# GreaTer at G=300 plus the three missing LPO span/word combinations.
# Activate re_prompt_optimization_vllm_v2 before launching.
# nohup bash codes/run_math_second_stage_qwen.sh > codes/nohup_outs/math_second_stage_qwen_sensitivity.log 2>&1 &
# MATH_SECOND_STAGE_DRY_RUN=1 prints all eight commands without loading models.
# MATH_SECOND_STAGE_GPU=N overrides the default physical GPU 1.
# Completed run directories with summary.json are skipped on a restart.
# MATH_SECOND_STAGE_RUN_SET=selected restores the completed ten-run comparison.
# MATH_SECOND_STAGE_RUN_SET=legacy enables the historical commands below.

RUN_SET="${MATH_SECOND_STAGE_RUN_SET:-sensitivity}"
if [[ "$RUN_SET" == "selected" || "$RUN_SET" == "sensitivity" ]]; then
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
  RUN_GPU="${MATH_SECOND_STAGE_GPU:-1}"
  DRY_RUN="${MATH_SECOND_STAGE_DRY_RUN:-0}"
  if [[ "$RUN_SET" == "sensitivity" ]]; then
    TOTAL_ATTEMPTS=8
  else
    TOTAL_ATTEMPTS=10
  fi
  if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
    printf 'MATH_SECOND_STAGE_DRY_RUN must be 0 or 1.\n' >&2
    exit 2
  fi

  SOURCE_DIR="outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit"
  RPO5_SOURCE="$SOURCE_DIR/prompt_experimental_iteration_5_from_actual_iteration_2.txt"
  RPO10_SOURCE="$SOURCE_DIR/prompt_experimental_iteration_10_from_actual_iteration_5.txt"
  for source_file in "$RPO5_SOURCE" "$RPO10_SOURCE" \
    data/processed/math500/train.jsonl data/processed/math500/validation.jsonl; do
    if [[ ! -s "$source_file" ]]; then
      printf 'Missing Math input: %s\n' "$source_file" >&2
      exit 2
    fi
  done

  run_selected_command() {
    # Print or run one method, skipping a previously completed CODE on restart.
    local method="$1" code="$2" attempt="$3"
    shift 3
    local output_dir="outputs/math_prompt_optimization/reasoning/$method/$code"
    printf '[math-qwen] attempt %s/%s | set=%s | method=%s | GPU=%s | CODE=%s\n' \
      "$attempt" "$TOTAL_ATTEMPTS" "$RUN_SET" "$method" "$RUN_GPU" "$code" >&2
    if [[ "$DRY_RUN" == 1 ]]; then
      printf 'CUDA_VISIBLE_DEVICES=%q ' "$RUN_GPU"
      printf '%q ' "$@"
      printf '\n'
    elif [[ -f "$output_dir/summary.json" ]]; then
      printf '[math-qwen] already completed; skipping %s\n' "$code" >&2
    else
      CUDA_VISIBLE_DEVICES="$RUN_GPU" "$@"
    fi
  }

  if [[ "$RUN_SET" == "sensitivity" ]]; then
    attempt=0
    for source in 5 10; do
      if [[ "$source" == 5 ]]; then
        source_prompt="$RPO5_SOURCE"
      else
        source_prompt="$RPO10_SOURCE"
      fi
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
        --gradient-cache-root outputs/shared_gradient_cache
        --seed 42
        --output-root outputs/math_prompt_optimization
        --overwrite
      )

      attempt=$((attempt + 1))
      code="math500_reasoning_qwen_rpo${source}_greater_sensitivity_g300_topu5_pool1600_vs900_tok4096_v2"
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
        --variant greater
        --train-sample-size 1600
        --gradient-sample-size 300
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
      run_selected_command "greater" "$code" "$attempt" "${command[@]}"

      for lpo_shape in 3:3 3:5 5:5; do
        max_locations="${lpo_shape%%:*}"
        max_words="${lpo_shape##*:}"
        attempt=$((attempt + 1))
        code="math500_reasoning_qwen_rpo${source}_lpo_sensitivity_f3_s${max_locations}_t${max_words}_c5_pool512_vs900_tok4096_v2"
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
          --max-locations "$max_locations"
          --max-words-per-location "$max_words"
          --num-candidates 5
        )
        run_selected_command "lpo" "$code" "$attempt" "${command[@]}"
      done
    done
    exit 0
  fi

  # Completed final comparison: five refiners on each retained RPO source.
  attempt=0
  for source in 5 10; do
    if [[ "$source" == 5 ]]; then
      source_prompt="$RPO5_SOURCE"
    else
      source_prompt="$RPO10_SOURCE"
    fi
    for method in gradpo_prob gradpo_gen_random greater greater_tg lpo; do
      attempt=$((attempt + 1))
      case "$method" in
        gradpo_prob|gradpo_gen_random)
          code="math500_reasoning_qwen_rpo${source}_${method}_final_s3_t3_h030_c7_g200_b5_f050_pool1600_vs900_tok4096_v2"
          ;;
        greater|greater_tg)
          code="math500_reasoning_qwen_rpo${source}_${method}_final_g200_topu5_pool1600_vs900_tok4096_v2"
          ;;
        lpo)
          code="math500_reasoning_qwen_rpo${source}_lpo_final_f3_s5_t3_c5_pool512_vs900_tok4096_v2"
          ;;
      esac
      common_args=(
        --code "$code"
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
        --gradient-cache-root outputs/shared_gradient_cache
        --seed 42
        --output-root outputs/math_prompt_optimization
        --overwrite
      )
      if [[ "$method" == gradpo_* ]]; then
        command=(
          python -u codes/run_qa_promptopt_gradpo.py
          "${common_args[@]}"
          --hf-device cuda:0
          --backend dual
          --objective-scoring-backend vllm
          --objective-scoring-batch-size 128
          --final-evaluation-backend vllm
          --gpu-memory-utilization 0.50
          --dual-vllm-gpu-memory-utilization 0.50
          --vllm-max-model-len 32768
          --variant "${method#gradpo_}"
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
      elif [[ "$method" == greater* ]]; then
        command=(
          python -u codes/run_qa_promptopt_greater.py
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
      else
        command=(
          python -u codes/run_qa_promptopt_lpo.py
          "${common_args[@]}"
          --optimizer-model Qwen/Qwen3-14B
          --optimizer-device cuda:0
          --optimizer-max-new-tokens 10000
          --backend vllm
          --gpu-memory-utilization 0.90
          --vllm-max-model-len 32768
          --train-sample-size 512
          --feedback-examples 3
          --max-locations 5
          --max-words-per-location 3
          --num-candidates 5
        )
      fi
      run_selected_command "$method" "$code" "$attempt" "${command[@]}"
    done
  done
  exit 0
fi

if [[ "$RUN_SET" != "legacy" ]]; then
  printf 'MATH_SECOND_STAGE_RUN_SET must be sensitivity, selected, or legacy.\n' >&2
  exit 2
fi

# Historical five-source matrix, preserved but inactive by default.
# Its RPO paths and 8192-token settings predate the selected Math comparison.

[[ -f outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_5.txt ]] || { echo 'Missing first-stage prompt: outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_5.txt'; exit 1; }
[[ -f outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_10.txt ]] || { echo 'Missing first-stage prompt: outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_10.txt'; exit 1; }
[[ -f outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_5.txt ]] || { echo 'Missing first-stage prompt: outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_5.txt'; exit 1; }
[[ -f outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_10.txt ]] || { echo 'Missing first-stage prompt: outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_10.txt'; exit 1; }
[[ -f outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_qwen_etgpo_qwen14opt_lambda1_vs900/final_prompt.txt ]] || { echo 'Missing first-stage prompt: outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_qwen_etgpo_qwen14opt_lambda1_vs900/final_prompt.txt'; exit 1; }

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_lpo.py \
  --code math500_reasoning_qwen_rpo5_lpo_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --optimizer-model Qwen/Qwen3-14B \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --train-sample-size 512 \
  --feedback-examples 3 \
  --max-locations 5 \
  --max-words-per-location 3 \
  --num-candidates 5

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_qwen_rpo5_greater_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_qwen_rpo5_greater_tg_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater_tg \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_qwen_rpo5_gradpo_gen_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_qwen_rpo5_gradpo_prob_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant prob \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_qwen_rpo5_gradpo_gen_random_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen_random \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_lpo.py \
  --code math500_reasoning_qwen_rpo10_lpo_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --optimizer-model Qwen/Qwen3-14B \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --train-sample-size 512 \
  --feedback-examples 3 \
  --max-locations 5 \
  --max-words-per-location 3 \
  --num-candidates 5

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_qwen_rpo10_greater_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_qwen_rpo10_greater_tg_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater_tg \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_qwen_rpo10_gradpo_gen_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_qwen_rpo10_gradpo_prob_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant prob \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_qwen_rpo10_gradpo_gen_random_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen_random \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_lpo.py \
  --code math500_reasoning_qwen_evoprompt5_lpo_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --optimizer-model Qwen/Qwen3-14B \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --train-sample-size 512 \
  --feedback-examples 3 \
  --max-locations 5 \
  --max-words-per-location 3 \
  --num-candidates 5

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_qwen_evoprompt5_greater_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_qwen_evoprompt5_greater_tg_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater_tg \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_qwen_evoprompt5_gradpo_gen_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_qwen_evoprompt5_gradpo_prob_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant prob \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_qwen_evoprompt5_gradpo_gen_random_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_5.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen_random \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_lpo.py \
  --code math500_reasoning_qwen_evoprompt10_lpo_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --optimizer-model Qwen/Qwen3-14B \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --train-sample-size 512 \
  --feedback-examples 3 \
  --max-locations 5 \
  --max-words-per-location 3 \
  --num-candidates 5

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_qwen_evoprompt10_greater_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_qwen_evoprompt10_greater_tg_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater_tg \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_qwen_evoprompt10_gradpo_gen_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_qwen_evoprompt10_gradpo_prob_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant prob \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_qwen_evoprompt10_gradpo_gen_random_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900/prompt_iteration_10.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen_random \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_lpo.py \
  --code math500_reasoning_qwen_etgpo1_lpo_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_qwen_etgpo_qwen14opt_lambda1_vs900/final_prompt.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --optimizer-model Qwen/Qwen3-14B \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --optimizer-max-new-tokens 10000 \
  --train-sample-size 512 \
  --feedback-examples 3 \
  --max-locations 5 \
  --max-words-per-location 3 \
  --num-candidates 5

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_qwen_etgpo1_greater_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_qwen_etgpo_qwen14opt_lambda1_vs900/final_prompt.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_qwen_etgpo1_greater_tg_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_qwen_etgpo_qwen14opt_lambda1_vs900/final_prompt.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater_tg \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-top-k 25 \
  --proposal-example-size 50 \
  --proposal-min-candidates 10 \
  --selection-top-mu 10 \
  --top-u 5 \
  --fluency-lambda 0.2 \
  --region-expansion-threshold 0.6 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_qwen_etgpo1_gradpo_gen_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_qwen_etgpo_qwen14opt_lambda1_vs900/final_prompt.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_qwen_etgpo1_gradpo_prob_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_qwen_etgpo_qwen14opt_lambda1_vs900/final_prompt.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant prob \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=2 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_qwen_etgpo1_gradpo_gen_random_lambda1_vs900 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_qwen_etgpo_qwen14opt_lambda1_vs900/final_prompt.txt \
  --model Qwen/Qwen3-4B \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --validation-fold-size 300 \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen_random \
  --train-sample-size 1200 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 5 \
  --max-region-tokens 2 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0
