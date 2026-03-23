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

-- with 5000 tokens limit
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:3" --max-iterations 20 --mutation-group-id "group_8" --update-mode "no_feedback" > nohup_outs/nohup_v2_qwen4_group8.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:3" --max-iterations 20 --mutation-group-id "group_9" --update-mode "no_feedback" > nohup_outs/nohup_v2_qwen4_group9.out 2>&1 &
nohup python -u core_trainer.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" --max-iterations 20 --mutation-group-id "group_10" > nohup_outs/nohup_v2_qwen4_group10.out 2>&1 &

nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:3" --max-iterations 20 --mutation-group-id "group_8" --update-mode "no_feedback" > nohup_outs/nohup_v2_gemma4_group8.out 2>&1 &
nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:3" --max-iterations 20 --mutation-group-id "group_9" --update-mode "no_feedback" > nohup_outs/nohup_v2_gemma4_group9.out 2>&1 &
nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:0" --max-iterations 20 --mutation-group-id "group_10" > nohup_outs/nohup_v2_gemma4_group10.out 2>&1 &

nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:3" --max-iterations 20 --mutation-group-id "group_2" --feedback-prompt "mistakes_v1" --selection-mode "mistakes" > nohup_outs/nohup_v2_gemma4_group2_mistakes.out 2>&1 &
nohup python -u core_trainer.py --model "google/gemma-3-4b-it" --device-map "cuda:3" --max-iterations 20 --mutation-group-id "group_2" --feedback-prompt "correct_v1" --selection-mode "correct" > nohup_outs/nohup_v2_gemma4_group2_corrects.out 2>&1 &

### cluster codes
nohup python -u core_trainer_cluster.py --model "Qwen/Qwen3-4B" --device-map "cuda:3" > nohup_outs/nohup_cluster_qwen4.out 2>&1 &


### taxonomy - dummy
nohup python -u core_trainer_taxonomy.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" > nohup_outs/nohup_taxonomy_qwen4.out 2>&1 &


### Current experiments
### etgpo taxonomy RE
## base taxonomy 3 runs
nohup python -u etgpo/etgpo_full.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:0" --valid_size 1000 --taxonomy_runs 1 \
--eval_runs 1 --num_guidance_prompts 5 --output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 \
--load_re_feedbacks "../unified_optimization_results/unified_opt_fs_tacred_20260321_204011/re_baseline_taxonomy_feedbacks.json" > nohup_outs/nohup_etgpo_re_taxonomy2.out 2>&1 &

nohup python -u etgpo/etgpo_full.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:2" --valid_size 1000 --taxonomy_runs 1 \
--eval_runs 1 --num_guidance_prompts 5 --output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 > nohup_outs/nohup_etgpo_re_taxonomy_run3.out 2>&1 &

nohup python -u etgpo/etgpo_full.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:3" --valid_size 1000 --taxonomy_runs 1 \
--eval_runs 1 --num_guidance_prompts 5 --output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 > nohup_outs/nohup_etgpo_re_taxonomy_run4.out 2>&1 &

## base taxonomy cluster 3 runs
nohup python -u etgpo/etgpo_full.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy_cluster \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:0" --valid_size 1000 --taxonomy_runs 1 \
--eval_runs 1 --num_guidance_prompts 3 --taxonomy_num_clusters 5 --output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 \
> nohup_outs/nohup_etgpo_re_taxonomy_cluster_run1.out 2>&1 &

nohup python -u etgpo/etgpo_full.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy_cluster \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:1" --valid_size 1000 --taxonomy_runs 1 \
--eval_runs 1 --num_guidance_prompts 3 --taxonomy_num_clusters 5 --output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 \
> nohup_outs/nohup_etgpo_re_taxonomy_cluster_run2.out 2>&1 &

nohup python -u etgpo/etgpo_full.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy_cluster \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:3" --valid_size 1000 --taxonomy_runs 1 \
--eval_runs 1 --num_guidance_prompts 3 --taxonomy_num_clusters 5 --output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 \
> nohup_outs/nohup_etgpo_re_taxonomy_cluster_run3.out 2>&1 &


## base taxonomy 3 runs with more coverage + guidlines

nohup python -u etgpo/etgpo_full.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:3" --valid_size 1000 --taxonomy_runs 1 \
--eval_runs 1 --num_guidance_prompts 5 --max_guidances 10 --coverage_threshold 0.9 \
--output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 > nohup_outs/nohup_etgpo_re_taxonomy_mg10_ct_0.9_run1.out 2>&1 &

nohup python -u etgpo/etgpo_full.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:0" --valid_size 1000 --taxonomy_runs 1 \
--eval_runs 1 --num_guidance_prompts 5 --max_guidances 10 --coverage_threshold 0.9 \
--output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 > nohup_outs/nohup_etgpo_re_taxonomy_mg10_ct_0.9_run2.out 2>&1 &

nohup python -u etgpo/etgpo_full.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:1" --valid_size 1000 --taxonomy_runs 1 \
--eval_runs 1 --num_guidance_prompts 5 --max_guidances 10 --coverage_threshold 0.9 \
--output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 > nohup_outs/nohup_etgpo_re_taxonomy_mg10_ct_0.9_run3.out 2>&1 &


## base taxonomy cluster 3 runs with more coverage + guidlines

nohup python -u etgpo/etgpo_full.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy_cluster \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:3" --valid_size 1000 --taxonomy_runs 1 \
--eval_runs 1 --num_guidance_prompts 5 --max_guidances 10 --coverage_threshold 0.9 \
--output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 > nohup_outs/nohup_etgpo_re_taxonomy_cluster_mg10_ct_0.9_run1.out 2>&1 &

nohup python -u etgpo/etgpo_full.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy_cluster \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:0" --valid_size 1000 --taxonomy_runs 1 \
--eval_runs 1 --num_guidance_prompts 5 --max_guidances 10 --coverage_threshold 0.9 \
--output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 > nohup_outs/nohup_etgpo_re_taxonomy_cluster_mg10_ct_0.9_run2.out 2>&1 &

nohup python -u etgpo/etgpo_full.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy_cluster \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:1" --valid_size 1000 --taxonomy_runs 1 \
--eval_runs 1 --num_guidance_prompts 5 --max_guidances 10 --coverage_threshold 0.9 \
--output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 > nohup_outs/nohup_etgpo_re_taxonomy_cluster_mg10_ct_0.9_run3.out 2>&1 &


##################################################
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