#!/bin/bash

# PIDs currently running
PID1=824691
PID2=825344

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

        nohup python -u core_trainer_evoprompt_de.py \
            --model "Qwen/Qwen3-4B" \
            --optimizer-model "Qwen/Qwen3-14B" \
            --device-map "cuda:2" \
            --max-iterations 20 \
            --population-size 5 \
            --evoprompt-train-episodes-per-iteration 1000 \
            --dataset fs_fewrel \
            --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
            --full-eval-split final_step_dev \
            --prompt-open-tag "[p]" \
            --prompt-close-tag "[/p]" \
            > nohup_outs/fewrel_nohup_evoprompt_de_qwen14opt_qwen4inf_itr20_train1000_final_step_dev.out 2>&1 &

        NODE2_STARTED=1
    fi

    # If PID2 finished, start node10 job once
    if [[ $ALIVE2 -ne 0 && $NODE10_STARTED -eq 0 ]]; then
        echo "$(date): PID $PID2 finished. Starting node10 job..."

        nohup python -u core_trainer_evoprompt_de.py \
            --model "google/gemma-3-4b-it" \
            --optimizer-model "google/gemma-3-12b-it" \
            --device-map "cuda:2" \
            --max-iterations 20 \
            --population-size 5 \
            --evoprompt-train-episodes-per-iteration 1000 \
            --dataset fs_fewrel \
            --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
            --full-eval-split final_step_dev \
            --prompt-open-tag "[p]" \
            --prompt-close-tag "[/p]" \
            > nohup_outs/fewrel_nohup_evoprompt_de_gemma12opt_gemma4inf_itr20_train1000_final_step_dev.out 2>&1 &

        NODE10_STARTED=1
    fi

    # Exit after both have been started
    if [[ $NODE2_STARTED -eq 1 && $NODE10_STARTED -eq 1 ]]; then
        echo "$(date): Both follow-up jobs launched."
        exit 0
    fi

    sleep 30
done