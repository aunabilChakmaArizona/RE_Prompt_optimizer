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


#### ensemble results
python3 codes/agents/agents_etgpo_results_ensemble.py \
  --root_dir unified_optimization_results \
  --method taxonomy \
  --q 3 \
  --w 3

python3 codes/agents/agents_etgpo_results_ensemble.py \
  --root_dir unified_optimization_results \
  --method taxonomy \
  --q 3 \
  --w 3 \
  --mode top_q_scores_only

#### extended runs

nohup python -u agents/agents_etgpo_extend_prompts.py \
  --source_run_dir ../unified_optimization_results/unified_opt_fs_tacred_20260322_015026 \
  --num_additional_prompts 10 \
  --device_map cuda:0 \
  > nohup_outs/nohup_extend_taxonomy_run1.out 2>&1 &

nohup python -u agents/agents_etgpo_extend_prompts.py \
  --source_run_dir ../unified_optimization_results/unified_opt_fs_tacred_20260322_023319 \
  --num_additional_prompts 10 \
  --device_map cuda:0 \
  > nohup_outs/nohup_extend_taxonomy_run2.out 2>&1 &

nohup python -u agents/agents_etgpo_extend_prompts.py \
  --source_run_dir ../unified_optimization_results/unified_opt_fs_tacred_20260322_023452 \
  --num_additional_prompts 10 \
  --device_map cuda:0 \
  > nohup_outs/nohup_extend_taxonomy_run3.out 2>&1 &

nohup python -u agents/agents_etgpo_extend_prompts.py \
  --source_run_dir ../unified_optimization_results/unified_opt_fs_tacred_20260322_181740 \
  --num_additional_prompts 10 \
  --device_map cuda:0 \
  > nohup_outs/nohup_extend_taxonomy_run4.out 2>&1 &

nohup python -u agents/agents_etgpo_extend_prompts.py \
  --source_run_dir ../unified_optimization_results/unified_opt_fs_tacred_20260322_182029 \
  --num_additional_prompts 10 \
  --device_map cuda:0 \
  > nohup_outs/nohup_extend_taxonomy_run5.out 2>&1 &

nohup python -u agents/agents_etgpo_extend_prompts.py \
  --source_run_dir ../unified_optimization_results/unified_opt_fs_tacred_20260322_182053 \
  --num_additional_prompts 10 \
  --device_map cuda:0 \
  > nohup_outs/nohup_extend_taxonomy_run6.out 2>&1 &

### taxonomy cluster with new count based selection

nohup python -u etgpo/etgpo_full_adjusted.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy_cluster \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:0" --valid_size 1000 \
--load_taxonomy "../unified_optimization_results/unified_opt_fs_tacred_20260322_015026" \
--eval_runs 1 --num_guidance_prompts 3 --taxonomy_num_clusters 5 --taxonomy_cluster_selection_mode error_count_weighted \
--output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 \
> nohup_outs/nohup_etgpo_re_taxonomy_cluster_err_cnt_selection_run1.out 2>&1 &

nohup python -u etgpo/etgpo_full_adjusted.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy_cluster \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:0" --valid_size 1000 \
--load_taxonomy "../unified_optimization_results/unified_opt_fs_tacred_20260322_023319" \
--eval_runs 1 --num_guidance_prompts 3 --taxonomy_num_clusters 5 --taxonomy_cluster_selection_mode error_count_weighted \
--output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 \
> nohup_outs/nohup_etgpo_re_taxonomy_cluster_err_cnt_selection_run2.out 2>&1 &

nohup python -u etgpo/etgpo_full_adjusted.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy_cluster \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:0" --valid_size 1000 \
--load_taxonomy "../unified_optimization_results/unified_opt_fs_tacred_20260322_023452" \
--eval_runs 1 --num_guidance_prompts 3 --taxonomy_num_clusters 5 --taxonomy_cluster_selection_mode error_count_weighted \
--output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 \
> nohup_outs/nohup_etgpo_re_taxonomy_cluster_err_cnt_selection_run3.out 2>&1 &

nohup python -u etgpo/etgpo_full_adjusted.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy_cluster \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:1" --valid_size 1000 \
--load_taxonomy "../unified_optimization_results/unified_opt_fs_tacred_20260322_181740" \
--eval_runs 1 --num_guidance_prompts 3 --taxonomy_num_clusters 5 --taxonomy_cluster_selection_mode error_count_weighted \
--output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 \
> nohup_outs/nohup_etgpo_re_taxonomy_cluster_err_cnt_selection_run4.out 2>&1 &

nohup python -u etgpo/etgpo_full_adjusted.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy_cluster \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:0" --valid_size 1000 \
--load_taxonomy "../unified_optimization_results/unified_opt_fs_tacred_20260322_182029" \
--eval_runs 1 --num_guidance_prompts 3 --taxonomy_num_clusters 5 --taxonomy_cluster_selection_mode error_count_weighted \
--output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 \
> nohup_outs/nohup_etgpo_re_taxonomy_cluster_err_cnt_selection_run5.out 2>&1 &

nohup python -u etgpo/etgpo_full_adjusted.py --task_mode relation_extraction_non_reasoning --dataset fs_tacred --methods taxonomy_cluster \
--main_model "Qwen/Qwen3-4B" --taxonomy_model "Qwen/Qwen3-14B" --device_map "cuda:0" --valid_size 1000 \
--load_taxonomy "../unified_optimization_results/unified_opt_fs_tacred_20260322_182053" \
--eval_runs 1 --num_guidance_prompts 3 --taxonomy_num_clusters 5 --taxonomy_cluster_selection_mode error_count_weighted \
--output_dir "../unified_optimization_results/" --taxonomy_batch_size 6 \
> nohup_outs/nohup_etgpo_re_taxonomy_cluster_err_cnt_selection_run6.out 2>&1 &


## gradient experiment sample analysis

python agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --eval-output-path "../trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_12_labels_predictions.json" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --output-file "../gradients_experiments/gradient_debug_node12_f1-37.json"

python agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --eval-output-path "../trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_12_labels_predictions.json" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --candidate-mode first_order_loss_approx \
  --output-file "../gradients_experiments/gradient_debug_FOLA_node12_f1-37.json"

python agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --eval-output-path "../trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_12_labels_predictions.json" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --candidate-mode first_order_loss_approx_normalized \
  --output-file "../gradients_experiments/gradient_debug_FOLAN_node12_f1-37.json"

## gradient experiment sample analysis --mistake-coverage
python agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --eval-output-path "../trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_12_labels_predictions.json" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --mistake-coverage 1.0 \
  --output-file "../gradients_experiments/gradient_debug_mc-1.0_node12_f1-37.json"

python agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --eval-output-path "../trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_12_labels_predictions.json" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --candidate-mode first_order_loss_approx \
  --mistake-coverage 1.0 \
  --output-file "../gradients_experiments/gradient_debug_mc-1.0_FOLA_node12_f1-37.json"

python agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --eval-output-path "../trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_12_labels_predictions.json" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --candidate-mode first_order_loss_approx_normalized \
  --mistake-coverage 1.0 \
  --output-file "../gradients_experiments/gradient_debug_mc-1.0_FOLAN_node12_f1-37.json"

## gradient experiment sample analysis --mistake-coverage with balance pick up train samples
python agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --eval-output-path "../trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_12_labels_predictions.json" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --mistake-coverage 1.0 \
  --output-file "../gradients_experiments/gradient_debug_mc-1.0_bln_node12_f1-37.json"

python agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --eval-output-path "../trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_12_labels_predictions.json" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --num-mistakes 90 \
  --output-file "../gradients_experiments/gradient_debug_n_mis-90_bln_node12_f1-37.json"

## gradient experiment sample analysis --mistake-coverage with balance pick up 4 bucket train samples
python agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --eval-output-path "../trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_12_labels_predictions.json" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --mistake-coverage 1.0 \
  --output-file "../gradients_experiments/gradient_debug_mc-1.0_4b_bln_node12_f1-37.json"

python agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --eval-output-path "../trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_12_labels_predictions.json" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --candidate-mode first_order_loss_approx_normalized \
  --mistake-coverage 1.0 \
  --output-file "../gradients_experiments/gradient_debug_mc-1.0_4b_bln_FOLAN_node12_f1-37.json"

python agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --eval-output-path "../trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_0_labels_predictions.json" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 0 \
  --candidate-mode first_order_loss_approx_normalized \
  --mistake-coverage 1.0 \
  --output-file "../gradients_experiments/gradient_debug_mc-1.0_4b_bln_FOLAN_node0_f1-25.json"

## gradient experiment sample analysis --mistake-coverage with balance pick up 4 bucket train samples with k new prompts generation
python agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --eval-output-path "../trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_12_labels_predictions.json" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --mistake-coverage 1.0 \
  --region-index 0 \
  --num-generated-prompts 5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 5 \
  --validation-batch-size 8 \
  --output-file "../gradients_experiments/gradient_debug_mc-1.0_4b_bln_node12_f1-37_region0_k5.json"

python agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --eval-output-path "../trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_12_labels_predictions.json" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --mistake-coverage 1.0 \
  --region-index 1 \
  --num-generated-prompts 5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 5 \
  --validation-batch-size 8 \
  --output-file "../gradients_experiments/gradient_debug_mc-1.0_4b_bln_node12_f1-37_region1_k5.json"



python agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --meta-prompt-model "Qwen/Qwen3-14B" \
  --device-map "cuda:1" \
  --eval-output-path "../trainings/20260303_002034_Qwen-Qwen3-4B/eval_outputs/EVALID_12_labels_predictions.json" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --mistake-coverage 1.0 \
  --num-edit-regions 3 \
  --num-generated-prompts 5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 5 \
  --validation-batch-size 1 \
  --output-file "../gradients_experiments/gradient_debug_mpm_mc-1.0_4b_bln_node12_f1-37_region_top3_k5.json"

## gradient based baseline stable run

python agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --meta-prompt-model "Qwen/Qwen3-14B" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --train-gradient-sample-size 10000 \
  --num-edit-regions 3 \
  --num-generated-prompts 5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 5 \
  --validation-batch-size 8 \
  --output-file "../gradients_experiments/gradient_baseline_train_node12_top3_k5.json"

## gradient based baseline stable run new mode

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 8 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-generated-prompts 5 \
  --num-region-candidates 5 \
  --top-k-prompts 3 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_initial1_qwen4" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_initial1_qwen4.out 2>&1 &


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
