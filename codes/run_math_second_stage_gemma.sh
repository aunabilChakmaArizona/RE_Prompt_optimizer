#!/usr/bin/env bash
set -euo pipefail

# Active DeltaAI matrix: remaining Gemma MATH-500 refiners on RPO5/RPO10.
# It runs 14 attempts: GradPO-Prob/Random, GreaTer G=200/300, and
# LPO S=3/5/7 with T=3. GradPO-Gen is already complete and is not repeated.
# Submit through codes/run_math_second_stage_gemma.slurm on DeltaAI.
# Preview locally: MATH_GEMMA_SECOND_STAGE_DRY_RUN=1 bash codes/run_math_second_stage_gemma.sh
# Resume at an attempt: MATH_GEMMA_SECOND_STAGE_START=N bash codes/run_math_second_stage_gemma.sh
# Set MATH_GEMMA_SECOND_STAGE_RUN_SET=legacy only to expose the old commands below.

RUN_SET="${MATH_GEMMA_SECOND_STAGE_RUN_SET:-delta_remaining}"
if [[ "$RUN_SET" == "delta_remaining" ]]; then
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  cd -- "$SCRIPT_DIR/.."

  START="${MATH_GEMMA_SECOND_STAGE_START:-1}"
  END="${MATH_GEMMA_SECOND_STAGE_END:-14}"
  DRY_RUN="${MATH_GEMMA_SECOND_STAGE_DRY_RUN:-0}"
  VLLM_RATIO="${MATH_GEMMA_SECOND_STAGE_VLLM_RATIO:-0.50}"
  SOURCE_DIR="experiment_tracking/second_stage/math_gemma_sources"
  CACHE_ROOT="outputs/deltaai_gemma_vllm011_images_enabled_cache"
  OUTPUT_ROOT="outputs/math_prompt_optimization"

  if [[ ! "$START" =~ ^([1-9]|1[0-4])$ || ! "$END" =~ ^([1-9]|1[0-4])$ || "$START" -gt "$END" ]]; then
    printf 'MATH_GEMMA_SECOND_STAGE_START/END must be 1-14 with START <= END.\n' >&2
    exit 2
  fi
  if [[ "$DRY_RUN" != 0 && "$DRY_RUN" != 1 ]]; then
    printf 'MATH_GEMMA_SECOND_STAGE_DRY_RUN must be 0 or 1.\n' >&2
    exit 2
  fi
  if [[ -n "${MATH_GEMMA_SECOND_STAGE_GPU:-}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
      printf 'Do not set MATH_GEMMA_SECOND_STAGE_GPU in Slurm; Slurm assigns the GPU.\n' >&2
      exit 2
    fi
    export CUDA_VISIBLE_DEVICES="$MATH_GEMMA_SECOND_STAGE_GPU"
  fi

  for required_file in \
    "$SOURCE_DIR/rpo5.txt" \
    "$SOURCE_DIR/rpo10.txt" \
    data/processed/math500/train.jsonl \
    data/processed/math500/validation.jsonl; do
    if [[ ! -s "$required_file" ]]; then
      printf 'Missing required input: %s\n' "$required_file" >&2
      exit 2
    fi
  done

  run_attempt() {
    # Run one method configuration or skip its completed output on restart.
    local attempt="$1" method="$2" code="$3"
    shift 3
    local output_dir="$OUTPUT_ROOT/reasoning/$method/$code"
    printf '[math-gemma-remaining] attempt %s/14 | method=%s | CODE=%s\n' \
      "$attempt" "$method" "$code" >&2
    if [[ "$DRY_RUN" == 1 ]]; then
      printf '%q ' "$@"
      printf '\n'
    elif [[ -f "$output_dir/summary.json" ]]; then
      printf '[math-gemma-remaining] already complete; skipping %s\n' "$code" >&2
    else
      "$@"
    fi
  }

  attempt=0
  for source in 5 10; do
    source_prompt="$SOURCE_DIR/rpo${source}.txt"
    common_args=(
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

    for variant in prob gen_random; do
      attempt=$((attempt + 1))
      method="gradpo_${variant}"
      code="math500_reasoning_gemma_rpo${source}_${method}_final_s7_t5_h045_c7_g200_b5_f050_pool600_vs900_tok4096_v3_images_enabled"
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
        --train-sample-size 600
        --gradient-sample-size 200
        --gradient-batch-size 1
        --selection-batch-size 4
        --num-edit-regions 7
        --max-region-tokens 5
        --region-expansion-threshold 0.45
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

    for gradient_size in 200 300; do
      attempt=$((attempt + 1))
      method="greater"
      code="math500_reasoning_gemma_rpo${source}_greater_sensitivity_g${gradient_size}_topu5_pool600_vs900_tok4096_v3_images_enabled"
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
        --variant greater
        --train-sample-size 600
        --gradient-sample-size "$gradient_size"
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
      if [[ "$attempt" -ge "$START" && "$attempt" -le "$END" ]]; then
        run_attempt "$attempt" "$method" "$code" "${command[@]}"
      fi
    done

    for max_locations in 3 5 7; do
      attempt=$((attempt + 1))
      method="lpo"
      code="math500_reasoning_gemma_rpo${source}_lpo_sensitivity_f3_s${max_locations}_t3_c5_pool512_vs900_tok4096_v3_images_enabled"
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
        --train-sample-size 512
        --feedback-examples 3
        --max-locations "$max_locations"
        --max-words-per-location 3
        --num-candidates 5
      )
      if [[ "$attempt" -ge "$START" && "$attempt" -le "$END" ]]; then
        run_attempt "$attempt" "$method" "$code" "${command[@]}"
      fi
    done
  done
  exit 0
fi

if [[ "$RUN_SET" != "legacy" ]]; then
  printf 'MATH_GEMMA_SECOND_STAGE_RUN_SET must be delta_remaining or legacy.\n' >&2
  exit 2
fi

# Run only after all five first-stage source prompts below exist.
# nohup bash codes/run_math_second_stage_gemma.sh > codes/nohup_outs/math_second_stage_gemma.log 2>&1 &

[[ -f outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt ]] || { echo 'Missing first-stage prompt: outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt'; exit 1; }
[[ -f outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt ]] || { echo 'Missing first-stage prompt: outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt'; exit 1; }
[[ -f outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt ]] || { echo 'Missing first-stage prompt: outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt'; exit 1; }
[[ -f outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt ]] || { echo 'Missing first-stage prompt: outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt'; exit 1; }
[[ -f outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_gemma_etgpo_gemma12opt_lambda1_vs1500/final_prompt.txt ]] || { echo 'Missing first-stage prompt: outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_gemma_etgpo_gemma12opt_lambda1_vs1500/final_prompt.txt'; exit 1; }

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_lpo.py \
  --code math500_reasoning_gemma_rpo5_lpo_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --optimizer-model google/gemma-3-12b-it \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --vllm-max-model-len 32768 \
  --optimizer-max-new-tokens 10000 \
  --train-sample-size 512 \
  --feedback-examples 3 \
  --max-locations 5 \
  --max-words-per-location 3 \
  --num-candidates 5

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_gemma_rpo5_greater_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater \
  --train-sample-size 600 \
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

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_gemma_rpo5_greater_tg_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater_tg \
  --train-sample-size 600 \
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

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_gemma_rpo5_gradpo_gen_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen \
  --train-sample-size 600 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 3 \
  --max-region-tokens 3 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_gemma_rpo5_gradpo_prob_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant prob \
  --train-sample-size 600 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 3 \
  --max-region-tokens 3 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_gemma_rpo5_gradpo_gen_random_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen_random \
  --train-sample-size 600 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 3 \
  --max-region-tokens 3 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_lpo.py \
  --code math500_reasoning_gemma_rpo10_lpo_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --optimizer-model google/gemma-3-12b-it \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --vllm-max-model-len 32768 \
  --optimizer-max-new-tokens 10000 \
  --train-sample-size 512 \
  --feedback-examples 3 \
  --max-locations 5 \
  --max-words-per-location 3 \
  --num-candidates 5

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_gemma_rpo10_greater_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater \
  --train-sample-size 600 \
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

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_gemma_rpo10_greater_tg_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater_tg \
  --train-sample-size 600 \
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

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_gemma_rpo10_gradpo_gen_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen \
  --train-sample-size 600 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 3 \
  --max-region-tokens 3 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_gemma_rpo10_gradpo_prob_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant prob \
  --train-sample-size 600 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 3 \
  --max-region-tokens 3 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_gemma_rpo10_gradpo_gen_random_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/rpo/math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen_random \
  --train-sample-size 600 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 3 \
  --max-region-tokens 3 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_lpo.py \
  --code math500_reasoning_gemma_evoprompt5_lpo_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --optimizer-model google/gemma-3-12b-it \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --vllm-max-model-len 32768 \
  --optimizer-max-new-tokens 10000 \
  --train-sample-size 512 \
  --feedback-examples 3 \
  --max-locations 5 \
  --max-words-per-location 3 \
  --num-candidates 5

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_gemma_evoprompt5_greater_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater \
  --train-sample-size 600 \
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

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_gemma_evoprompt5_greater_tg_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater_tg \
  --train-sample-size 600 \
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

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_gemma_evoprompt5_gradpo_gen_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen \
  --train-sample-size 600 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 3 \
  --max-region-tokens 3 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_gemma_evoprompt5_gradpo_prob_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant prob \
  --train-sample-size 600 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 3 \
  --max-region-tokens 3 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_gemma_evoprompt5_gradpo_gen_random_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_5.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen_random \
  --train-sample-size 600 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 3 \
  --max-region-tokens 3 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_lpo.py \
  --code math500_reasoning_gemma_evoprompt10_lpo_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --optimizer-model google/gemma-3-12b-it \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --vllm-max-model-len 32768 \
  --optimizer-max-new-tokens 10000 \
  --train-sample-size 512 \
  --feedback-examples 3 \
  --max-locations 5 \
  --max-words-per-location 3 \
  --num-candidates 5

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_gemma_evoprompt10_greater_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater \
  --train-sample-size 600 \
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

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_gemma_evoprompt10_greater_tg_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater_tg \
  --train-sample-size 600 \
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

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_gemma_evoprompt10_gradpo_gen_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen \
  --train-sample-size 600 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 3 \
  --max-region-tokens 3 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_gemma_evoprompt10_gradpo_prob_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant prob \
  --train-sample-size 600 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 3 \
  --max-region-tokens 3 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_gemma_evoprompt10_gradpo_gen_random_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/evoprompt_de/math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500/prompt_iteration_10.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen_random \
  --train-sample-size 600 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 3 \
  --max-region-tokens 3 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_lpo.py \
  --code math500_reasoning_gemma_etgpo1_lpo_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_gemma_etgpo_gemma12opt_lambda1_vs1500/final_prompt.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --optimizer-model google/gemma-3-12b-it \
  --optimizer-device cuda:0 \
  --backend vllm \
  --gpu-memory-utilization 0.90 \
  --vllm-max-model-len 32768 \
  --optimizer-max-new-tokens 10000 \
  --train-sample-size 512 \
  --feedback-examples 3 \
  --max-locations 5 \
  --max-words-per-location 3 \
  --num-candidates 5

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_gemma_etgpo1_greater_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_gemma_etgpo_gemma12opt_lambda1_vs1500/final_prompt.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater \
  --train-sample-size 600 \
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

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_greater.py \
  --code math500_reasoning_gemma_etgpo1_greater_tg_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_gemma_etgpo_gemma12opt_lambda1_vs1500/final_prompt.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant greater_tg \
  --train-sample-size 600 \
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

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_gemma_etgpo1_gradpo_gen_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_gemma_etgpo_gemma12opt_lambda1_vs1500/final_prompt.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen \
  --train-sample-size 600 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 3 \
  --max-region-tokens 3 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_gemma_etgpo1_gradpo_prob_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_gemma_etgpo_gemma12opt_lambda1_vs1500/final_prompt.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant prob \
  --train-sample-size 600 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 3 \
  --max-region-tokens 3 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0

CUDA_VISIBLE_DEVICES=3 python -u codes/run_qa_promptopt_gradpo.py \
  --code math500_reasoning_gemma_etgpo1_gradpo_gen_random_lambda1_vs1500 \
  --qa-task math500 \
  --qa-mode reasoning \
  --train-path data/processed/math500/train.jsonl \
  --validation-path data/processed/math500/validation.jsonl \
  --initial-prompt-file outputs/math_prompt_optimization/reasoning/etgpo/math500_reasoning_gemma_etgpo_gemma12opt_lambda1_vs1500/final_prompt.txt \
  --model google/gemma-3-4b-it \
  --device cuda:0 \
  --target-max-new-tokens 8192 \
  --validation-std-penalty 1.0 \
  --output-root outputs/math_prompt_optimization \
  --overwrite \
  --backend dual \
  --final-evaluation-backend vllm \
  --dual-vllm-gpu-memory-utilization 0.5 \
  --vllm-max-model-len 16384 \
  --variant gen_random \
  --train-sample-size 600 \
  --gradient-sample-size 200 \
  --gradient-batch-size 2 \
  --selection-batch-size 4 \
  --num-edit-regions 3 \
  --max-region-tokens 3 \
  --region-expansion-threshold 0.6 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --beam-replacement-mode llm_synthesis \
  --fluency-lambda 0.5 \
  --candidate-max-new-tokens 10000 \
  --synthesis-max-new-tokens 10000 \
  --synthesis-batch-size 4 \
  --hf-device cuda:0
