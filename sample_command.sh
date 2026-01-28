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

python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:2"