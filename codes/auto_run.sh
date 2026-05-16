#!/bin/bash

# PIDs currently running
PID1=2670511
PID2=2671187

check_pid () {
    ps -p "$1" > /dev/null 2>&1
}

echo "Monitoring PIDs: $PID1 and $PID2"

NODE2_STARTED=0
NODE10_STARTED=0

while true; do
    check_pid $PID1
    ALIVE1=$?

    check_pid $PID2
    ALIVE2=$?

    # If PID1 finished, start node2 job once
    if [[ $ALIVE1 -ne 0 && $NODE2_STARTED -eq 0 ]]; then
        echo "$(date): PID $PID1 finished. Starting node2 job..."

        nohup python -u GreaTer/experiments/relation_extraction_greater.py \
          --model "Qwen/Qwen3-4B" \
          --device-map "cuda:1" \
          --prompt-source population \
          --population-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
          --prompt-node-id 2 \
          --dataset-type "fs_tacred" \
          --train-gradient-sample-size 10000 \
          --gradient-batch-size 4 \
          --selection-batch-size 8 \
          --proposal-example-size 50 \
          --proposal-top-k 25 \
          --proposal-min-candidates 10 \
          --selection-top-mu 10 \
          --top-z 5 \
          --dev-f1-std-penalty 2.0 \
          --fluency-lambda 0.2 \
          --fluency-scope instruction \
          --fluency-metric nll \
          --n-steps 1 \
          --eval-every 1 \
          --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
          --full-eval-split final_step_dev \
          --output-root-dir "../greater_experiments" \
          --output-substring "seq_node2_1steps_lmd0.2_qwen4" \
          > nohup_outs/greater_seq_node2_1steps_n10000_lmd0.2_k25_u10_z5_qwen4.out 2>&1 &

        NODE2_STARTED=1
    fi

    # If PID2 finished, start node10 job once
    if [[ $ALIVE2 -ne 0 && $NODE10_STARTED -eq 0 ]]; then
        echo "$(date): PID $PID2 finished. Starting node10 job..."

        nohup python -u GreaTer/experiments/relation_extraction_greater.py \
          --model "Qwen/Qwen3-4B" \
          --device-map "cuda:1" \
          --prompt-source population \
          --population-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
          --prompt-node-id 10 \
          --dataset-type "fs_tacred" \
          --train-gradient-sample-size 10000 \
          --gradient-batch-size 4 \
          --selection-batch-size 8 \
          --proposal-example-size 50 \
          --proposal-top-k 25 \
          --proposal-min-candidates 10 \
          --selection-top-mu 10 \
          --top-z 5 \
          --dev-f1-std-penalty 2.0 \
          --fluency-lambda 0.2 \
          --fluency-scope instruction \
          --fluency-metric nll \
          --n-steps 1 \
          --eval-every 1 \
          --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
          --full-eval-split final_step_dev \
          --output-root-dir "../greater_experiments" \
          --output-substring "seq_node10_1steps_lmd0.2_qwen4" \
          > nohup_outs/greater_seq_node10_1steps_n10000_lmd0.2_k25_u10_z5_qwen4.out 2>&1 &

        NODE10_STARTED=1
    fi

    # Exit after both have been started
    if [[ $NODE2_STARTED -eq 1 && $NODE10_STARTED -eq 1 ]]; then
        echo "$(date): Both follow-up jobs launched."
        exit 0
    fi

    sleep 30
done