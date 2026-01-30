#!/usr/bin/env bash

python /storage2/home/aunabilchakma/codes/RE_Prompt_optimizer/codes/core_trainer.py \
  --model "meta-llama/Llama-3.2-3B-Instruct" \
  --dataset-type "fs_tacred" \
  --dataset-prefix "fs_tacred" \
  --train-samples "fs_tacred_train_samples.pkl" \
  --dev-split "dev" \
  --test-split "test" \
  --max-iterations 5 \
  --feedback-sample-size 100

nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:3" --max-iterations 20 &
nohup python -u core_trainer.py --model "google/gemma-3-12b-it" --device-map "cuda:2" --max-iterations 20 > nohup_gemma12.out 2>&1 &

nohup python -u core_trainer.py --model "google/gemma-3-12b-it" --device-map "cuda:1" --max-iterations 20 --feedback-prompt "correct_v1" > nohup_gemma12_corrects.out 2>&1 &


# models
Qwen/Qwen3-4B
Qwen/Qwen3-8B
Qwen/Qwen3-14B
google/gemma-3-4b-it
google/gemma-3-12b-it

#
"--feedback-prompt",
choices=["correct_and_mistakes_v1", "correct_v1", "mistakes_v1"]