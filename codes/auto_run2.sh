#!/bin/bash

# PID currently running
PID1=4190234

check_pid () {
    ps -p "$1" > /dev/null 2>&1
}

echo "Monitoring PID: $PID1"

JOB_STARTED=0

while true; do
    check_pid $PID1
    ALIVE1=$?

    # If PID1 finished, start follow-up job once
    if [[ $ALIVE1 -ne 0 && $JOB_STARTED -eq 0 ]]; then
        echo "$(date): PID $PID1 finished. Starting follow-up job..."

        nohup python -u agents/agent_gradient_eval_debug.py \
          --model "google/gemma-3-4b-it" \
          --mode "LLM_CANDIDATE_SUGGESTION" \
          --device-map "cuda:0" \
          --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
          --prompt-node-id 18 \
          --dataset-type "fs_tacred" \
          --train-gradient-sample-size 30000 \
          --gradient-batch-size 2 \
          --max-regions 10 \
          --max-total-region-tokens 50 \
          --max-region-tokens 3 \
          --region-expansion-threshold-ratio 0.6 \
          --num-edit-regions 3 \
          --num-region-candidates 7 \
          --beam-width 5 \
          --selection-perplexity-lambda 0.5 \
          --use-log-fluency-score \
          --meta-prompt-max-new-tokens 10000 \
          --meta-prompt-batch-size 1 \
          --validation-batch-size 8 \
          --Q 1 \
          --selection-f1-std-penalty 2.0 \
          --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
          --full-eval-split final_step_dev \
          --output-root-dir "../gradients_experiments" \
          --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node18_v6test_final_updated_code" \
          > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node18_v6test_final_updated_code.out 2>&1 &

        JOB_STARTED=1
    fi

    # Exit after job has been started
    if [[ $JOB_STARTED -eq 1 ]]; then
        echo "$(date): Follow-up job launched."
        exit 0
    fi

    sleep 30
done