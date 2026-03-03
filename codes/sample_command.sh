# general feedback
nohup python -u core_trainer.py --model "google/gemma-3-12b-it" --device-map "cuda:2" --max-iterations 20 > nohup_gemma12.out 2>&1 &

# specific feedback
nohup python -u core_trainer.py --model "google/gemma-3-12b-it" --device-map "cuda:3" --max-iterations 20 --feedback-prompt "correct_v1" --selection-mode "correct" > nohup_gemma12_corrects.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-14B" --device-map "cuda:3" --max-iterations 10 --feedback-prompt "correct_v1" --selection-mode "correct" > nohup_qwen14_corrects.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-14B" --device-map "cuda:1" --max-iterations 20 --feedback-prompt "mistakes_v1" --selection-mode "mistakes" > nohup_qwen14_mistakes.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" --max-iterations 10 --feedback-prompt "mistakes_v1" --selection-mode "mistakes" > nohup_qwen4_mistakes.out 2>&1 &
nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:2" --max-iterations 10 --feedback-prompt "mistakes_v1" --selection-mode "mistakes" > nohup_gemma4_mistakes.out 2>&1 &
nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:2" --max-iterations 10 --feedback-prompt "correct_v1" --selection-mode "correct" > nohup_gemma4_corrects.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:2" --max-iterations 10 --feedback-prompt "correct_v1" --selection-mode "correct" > nohup_qwen4_corrects.out 2>&1 &

# random (old wrong experiments)
nohup python -u core_trainer.py --model "google/gemma-3-12b-it" --device-map "cuda:3" --max-iterations 10 --update-mode "random" --mutation-prompt "random_v1" > nohup_gemma12_random_mutation.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-14B" --device-map "cuda:1" --max-iterations 10 --update-mode "random" --mutation-prompt "random_v1" > nohup_qwen14_random_mutation.out 2>&1 &

nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:3" --max-iterations 10 --update-mode "random" --mutation-prompt "random_v1" > nohup_gemma4_random_mutation.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:3" --max-iterations 10 --update-mode "random" --mutation-prompt "random_v1" > nohup_qwen4_random_mutation.out 2>&1 &


# no feedback
nohup python -u core_trainer.py --model "google/gemma-3-12b-it" --device-map "cuda:3" --max-iterations 10 --update-mode "no_feedback" --mutation-prompt "no_feedback_v1" > nohup_gemma12_no_feedback.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-14B" --device-map "cuda:2" --max-iterations 10 --update-mode "no_feedback" --mutation-prompt "no_feedback_v1" > nohup_qwen14_no_feedback.out 2>&1 &
nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:1" --max-iterations 10 --update-mode "no_feedback" --mutation-prompt "no_feedback_v1" > nohup_gemma4_no_feedback.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" --max-iterations 10 --update-mode "no_feedback" --mutation-prompt "no_feedback_v1" > nohup_qwen4_no_feedback.out 2>&1 &

# only random updates
nohup python -u core_trainer.py --model "google/gemma-3-12b-it" --device-map "cuda:3" --max-iterations 10 --update-mode "no_feedback" --mutation-prompt "random_v1" > nohup_gemma12_pure_random.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-14B" --device-map "cuda:2" --max-iterations 10 --update-mode "no_feedback" --mutation-prompt "random_v1" > nohup_qwen14_pure_random.out 2>&1 &
nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:3" --max-iterations 10 --update-mode "no_feedback" --mutation-prompt "random_v1" > nohup_gemma4_pure_random.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:2" --max-iterations 10 --update-mode "no_feedback" --mutation-prompt "random_v1" > nohup_qwen4_pure_random.out 2>&1 &

nohup python -u core_trainer.py --model "google/gemma-3-12b-it" --device-map "cuda:2" --max-iterations 10 --update-mode "no_feedback" --mutation-prompt "random_v2" > nohup_gemma12_pure_random_v2.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-14B" --device-map "cuda:2" --max-iterations 10 --update-mode "no_feedback" --mutation-prompt "random_v2" > nohup_qwen14_pure_random_v2.out 2>&1 &
nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:1" --max-iterations 10 --update-mode "no_feedback" --mutation-prompt "random_v2" --feedback-open-tag "[f]" --feedback-close-tag "[/f]" --prompt-open-tag "[p]" --prompt-close-tag "[/p]" > nohup_gemma4_pure_random_v2.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" --max-iterations 10 --update-mode "no_feedback" --mutation-prompt "random_v2" > nohup_qwen4_pure_random_v2.out 2>&1 &

#qwen large
nohup python -u core_trainer.py --model "Qwen/Qwen3-32B" --device-map "cuda:0" --max-iterations 10 --feedback-prompt "correct_v1" --selection-mode "correct" > nohup_qwen32_corrects.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-32B" --device-map "cuda:0" --max-iterations 10 --feedback-prompt "mistakes_v1" --selection-mode "mistakes" > nohup_qwen32_mistakes.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-32B" --device-map "cuda:1" --max-iterations 10 --update-mode "no_feedback" --mutation-prompt "random_v2" > nohup_qwen32_pure_random_v2.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-32B" --device-map "cuda:0" --max-iterations 10 --update-mode "no_feedback" --mutation-prompt "no_feedback_v1" > nohup_qwen32_no_feedback.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-32B" --device-map "cuda:0" --max-iterations 10 > nohup_qwen32.out 2>&1 &

##### new format of prompts_v2
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:3" --max-iterations 20 --update-mode "no_feedback" --mutation-prompt "no_feedback_v1" > nohup_outs/nohup_v2_qwen4_no_feedback.out 2>&1 &
nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:2" --max-iterations 20 --update-mode "no_feedback" --mutation-prompt "no_feedback_v1" > nohup_outs/nohup_v2_gemma4_no_feedback.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:3" --max-iterations 20 > nohup_outs/nohup_v2_qwen4_feedback_corrects_and_mistakes.out 2>&1 &
nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:2" --max-iterations 20 > nohup_outs/nohup_v2_gemma4_feedback_corrects_and_mistakes.out 2>&1 &

##### new multim-mode mutations
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:2" --max-iterations 20 --mutation-group-id "group_4" --update-mode "no_feedback" > nohup_outs/nohup_v2_qwen4_group4.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:2" --max-iterations 20 --mutation-group-id "group_5" --update-mode "no_feedback" > nohup_outs/nohup_v2_qwen4_group5.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" --max-iterations 20 --mutation-group-id "group_6" > nohup_outs/nohup_v2_qwen4_group6.out 2>&1 &

nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:3" --max-iterations 20 --mutation-group-id "group_4" --update-mode "no_feedback" > nohup_outs/nohup_v2_gemma4_group4.out 2>&1 &
nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:3" --max-iterations 20 --mutation-group-id "group_5" --update-mode "no_feedback" > nohup_outs/nohup_v2_gemma4_group5.out 2>&1 &
nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:1" --max-iterations 20 --mutation-group-id "group_6" > nohup_outs/nohup_v2_gemma4_group6.out 2>&1 &

nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" --max-iterations 20 --mutation-group-id "group_2" --feedback-prompt "mistakes_v1" --selection-mode "mistakes" > nohup_outs/nohup_v2_qwen4_group2_mistakes.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" --max-iterations 20 --mutation-group-id "group_2" --feedback-prompt "correct_v1" --selection-mode "correct" > nohup_outs/nohup_v2_qwen4_group2_corrects.out 2>&1 &

nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:3" --max-iterations 20 --mutation-group-id "group_8" --update-mode "no_feedback" > nohup_outs/nohup_v2_qwen4_group8.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:3" --max-iterations 20 --mutation-group-id "group_9" --update-mode "no_feedback" > nohup_outs/nohup_v2_qwen4_group9.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" --max-iterations 20 --mutation-group-id "group_10" > nohup_outs/nohup_v2_qwen4_group10.out 2>&1 &

# tags custom
--feedback-open-tag "[f]" --feedback-close-tag "[/f]" --prompt-open-tag "[p]" --prompt-close-tag "[/p]"

# models
Qwen/Qwen3-4B
Qwen/Qwen3-8B
Qwen/Qwen3-14B
google/gemma-3-4b-it
google/gemma-3-12b-it

#
"--feedback-prompt",
choices=["correct_and_mistakes_v1", "correct_v1", "mistakes_v1"]