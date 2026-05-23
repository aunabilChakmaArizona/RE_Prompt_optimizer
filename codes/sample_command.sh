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

# ** dummy run
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 4 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 10 \
  --num-generated-prompts 20 \
  --top-k-prompts 10 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "lm_prob_cand_sugg_prefix_context_node10_region3s_qwen4_dm" \
  > nohup_outs/nohup_gradient_lm_prob_cand_sugg_prefix_context_node10_region3s_qwen4_dm.out 2>&1 &

#################################################################
#################################################################
#################################################################
#################################################################

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

## gradient based baseline stable run new mode LLM_CANDIDATE_SUGGESTION

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 4 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --num-generated-prompts 20 \
  --top-k-prompts 10 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_region3s_qwen4" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_region3s_qwen4.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 9 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --num-generated-prompts 20 \
  --top-k-prompts 10 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_node9_region3s_qwen4" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_node9_region3s_qwen4.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 4 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --num-generated-prompts 20 \
  --top-k-prompts 10 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_node7_region3s_qwen4" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_node7_region3s_qwen4.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 0 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 4 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --num-generated-prompts 20 \
  --top-k-prompts 10 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_node0_region3s_qwen4" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_node0_region3s_qwen4.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 4 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --num-generated-prompts 20 \
  --top-k-prompts 10 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_node3_region3s_qwen4" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_node3_region3s_qwen4.out 2>&1 &

## gradient based baseline stable run new mode LLM_CANDIDATE_SUGGESTION + FULL_PROMPT_AS_CONTEXT

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 4 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --num-generated-prompts 20 \
  --top-k-prompts 10 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "lm_prob_cand_sugg_prefix_context_node10_region3s_qwen4" \
  > nohup_outs/nohup_gradient_lm_prob_cand_sugg_prefix_context_node10_region3s_qwen4.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 9 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 4 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --num-generated-prompts 20 \
  --top-k-prompts 10 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "lm_prob_cand_sugg_prefix_context_node9_region3s_qwen4" \
  > nohup_outs/nohup_gradient_lm_prob_cand_sugg_prefix_context_node9_region3s_qwen4.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 4 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --num-generated-prompts 20 \
  --top-k-prompts 10 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "lm_prob_cand_sugg_prefix_context_node7_region3s_qwen4" \
  > nohup_outs/nohup_gradient_lm_prob_cand_sugg_prefix_context_node7_region3s_qwen4.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 0 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 4 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --num-generated-prompts 20 \
  --top-k-prompts 10 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "lm_prob_cand_sugg_prefix_context_node0_region3s_qwen4" \
  > nohup_outs/nohup_gradient_lm_prob_cand_sugg_prefix_context_node0_region3s_qwen4.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 4 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --num-generated-prompts 20 \
  --top-k-prompts 10 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "lm_prob_cand_sugg_prefix_context_node3_region3s_qwen4" \
  > nohup_outs/nohup_gradient_lm_prob_cand_sugg_prefix_context_node3_region3s_qwen4.out 2>&1 &


## base run for gradient based updates
nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:3" --max-iterations 10 --mutation-group-id "group_2" > nohup_outs/nohup_v3_qwen4_group2_mixed.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:3" --max-iterations 10 --mutation-group-id "group_2" --initial-prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B" --parent-selection-mode "initial_prompt_only"  --initial-prompt-node-id 10 > nohup_outs/nohup_v3_qwen4_group2_mixed_node10_initial_prompt_only.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:3" --max-iterations 10 --mutation-group-id "group_2" \
--initial-prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B" --parent-selection-mode "initial_prompt_only" \
--initial-prompt-node-id 9 > nohup_outs/nohup_v3_qwen4_group2_mixed_node9_initial_prompt_only.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" --max-iterations 10 --mutation-group-id "group_2" \
--initial-prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B" --parent-selection-mode "initial_prompt_only" \
--initial-prompt-node-id 7 > nohup_outs/nohup_v3_qwen4_group2_mixed_node7_initial_prompt_only.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" --max-iterations 10 --mutation-group-id "group_2" \
--initial-prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B" --parent-selection-mode "initial_prompt_only" \
--initial-prompt-node-id 0 > nohup_outs/nohup_v3_qwen4_group2_mixed_node0_initial_prompt_only.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" --max-iterations 10 --mutation-group-id "group_2" \
--initial-prompt-source-path "../trainings/20260405_202156_Qwen-Qwen3-4B" --parent-selection-mode "initial_prompt_only" \
--initial-prompt-node-id 3 > nohup_outs/nohup_v3_qwen4_group2_mixed_node3_initial_prompt_only.out 2>&1 &




## comparison
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 13 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 4 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --num-generated-prompts 20 \
  --top-k-prompts 10 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_20260303_002034_node13_region3s_qwen4" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_20260303_002034_node13_region3s_qwen4.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" --max-iterations 10 --mutation-group-id "group_2" \
--initial-prompt-source-path "../trainings/20260303_002034_Qwen-Qwen3-4B" --parent-selection-mode "initial_prompt_only" \
--initial-prompt-node-id 13 > nohup_outs/nohup_v3_qwen4_group2_mixed_20260303_002034_node13_initial_prompt_only.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 4 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --num-generated-prompts 20 \
  --top-k-prompts 10 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_20260301_030139_node11_region3s_qwen4" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_20260301_030139_node11_region3s_qwen4.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" --max-iterations 10 --mutation-group-id "group_2" \
--initial-prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B" --parent-selection-mode "initial_prompt_only" \
--initial-prompt-node-id 11 > nohup_outs/nohup_v3_qwen4_group2_mixed_20260301_030139_node11_initial_prompt_only.out 2>&1 &

## updated code with beam search for lambda test

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 4 \
  --beam-width 2 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam3_20260301_030139_node11_region3s_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam3_20260301_030139_node11_region3s_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --beam-width 3 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam3real_20260301_030139_node11_region3s_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam3real_20260301_030139_node11_region3s_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --beam-width 3 \
  --selection-perplexity-lambda 0.0 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam3real_px_0.0_20260301_030139_node11_region3s_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam3real_px_0.0_20260301_030139_node11_region3s_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --beam-width 3 \
  --selection-perplexity-lambda 0.1 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam3real_px_0.1_20260301_030139_node11_region3s_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam3real_px_0.1_20260301_030139_node11_region3s_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --beam-width 3 \
  --selection-perplexity-lambda 0.4 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam3real_px_0.4_20260301_030139_node11_region3s_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam3real_px_0.4_20260301_030139_node11_region3s_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --beam-width 3 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam3real_px_0.5_20260301_030139_node11_region3s_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam3real_px_0.5_20260301_030139_node11_region3s_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --beam-width 3 \
  --selection-perplexity-lambda 1.0 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam3real_px_1.0_20260301_030139_node11_region3s_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam3real_px_1.0_20260301_030139_node11_region3s_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 5 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --beam-width 3 \
  --selection-perplexity-lambda 0.3 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam3real_Q1_px_0.3_r3_20260301_030139_node11_region3s_qwen4_region_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam3real_Q1_px_0.3_r3_20260301_030139_node11_region3s_qwen4_region_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_px_0.2_r5_20260301_030139_node11_region3s_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_px_0.2_r5_20260301_030139_node11_region3s_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 7 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_px_0.2_r7_20260301_030139_node11_region3s_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_px_0.2_r7_20260301_030139_node11_region3s_qwen4_lambda_tune.out 2>&1 &

## iteration based lambda tuning***

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q5_px_0.2_r5_20260301_030139_node11_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q5_px_0.2_r5_20260301_030139_node11_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.0 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q5_px_0.0_r5_20260301_030139_node11_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q5_px_0.0_r5_20260301_030139_node11_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.1 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q5_px_0.1_r5_20260301_030139_node11_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q5_px_0.1_r5_20260301_030139_node11_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.3 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q5_px_0.3_r5_20260301_030139_node11_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q5_px_0.3_r5_20260301_030139_node11_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q5_px_0.5_r5_20260301_030139_node11_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q5_px_0.5_r5_20260301_030139_node11_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.4 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q5_px_0.4_r5_20260301_030139_node11_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q5_px_0.4_r5_20260301_030139_node11_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.6 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q5_px_0.6_r5_20260301_030139_node11_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q5_px_0.6_r5_20260301_030139_node11_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.7 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q5_px_0.7_r5_20260301_030139_node11_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q5_px_0.7_r5_20260301_030139_node11_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.8 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q5_px_0.8_r5_20260301_030139_node11_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q5_px_0.8_r5_20260301_030139_node11_qwen4_lambda_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.9 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q5_px_0.9_r5_20260301_030139_node11_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q5_px_0.9_r5_20260301_030139_node11_qwen4_lambda_tune.out 2>&1 &


nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 1.0 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q5_px_1.0_r5_20260301_030139_node11_qwen4_lambda_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q5_px_1.0_r5_20260301_030139_node11_qwen4_lambda_tune.out 2>&1 &

## iteration based region 7 tuning***
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 7 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.1 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q1_px_0.1_r7_20260301_030139_node11_region3s_qwen4_region_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q1_px_0.1_r7_20260301_030139_node11_region3s_qwen4_region_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 7 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.0 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q1_px_0.0_r7_20260301_030139_node11_region3s_qwen4_region_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q1_px_0.0_r7_20260301_030139_node11_region3s_qwen4_region_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 7 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.3 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q1_px_0.3_r7_20260301_030139_node11_region3s_qwen4_region_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q1_px_0.3_r7_20260301_030139_node11_region3s_qwen4_region_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 7 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q1_px_0.5_r7_20260301_030139_node11_region3s_qwen4_region_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q1_px_0.5_r7_20260301_030139_node11_region3s_qwen4_region_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 7 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.4 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q1_px_0.4_r7_20260301_030139_node11_region3s_qwen4_region_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q1_px_0.4_r7_20260301_030139_node11_region3s_qwen4_region_tune.out 2>&1 &

## population tuning (will work as final run too)
nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" \
--max-iterations 20 --population-size 3 --mutation-group-id "group_2" > nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_3_itr_20.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" \
--max-iterations 20 --population-size 5 --mutation-group-id "group_2" > nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_5_itr_20.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" \
--max-iterations 20 --population-size 7 --mutation-group-id "group_2" > nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_7_itr_20.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" \
--max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_10_itr_20.out 2>&1 &

## expansion ratio tune (lambda optimized, region size optimized)

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.7 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 3 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q3_px_0.5_r5_er_0.7_20260301_030139_node11_qwen4_er_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q3_px_0.5_r5_er_0.7_20260301_030139_node11_qwen4_er_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.5 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 3 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q3_px_0.5_r5_er_0.5_20260301_030139_node11_qwen4_er_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q3_px_0.5_r5_er_0.5_20260301_030139_node11_qwen4_er_tune.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260301_030139_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.4 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 3 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5real_Q3_px_0.5_r5_er_0.4_20260301_030139_node11_qwen4_er_tune" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5real_Q3_px_0.5_r5_er_0.4_20260301_030139_node11_qwen4_er_tune.out 2>&1 &


## resume textual runs for 10 addtl. iterations
nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" \
--max-iterations 10 --population-size 3 --mutation-group-id "group_2" \
--load-population --initial-prompt-source-path "../trainings/20260421_030726_Qwen-Qwen3-4B" \
> nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_3_itr_10_resume_from_20260421_030726.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" \
--max-iterations 10 --population-size 5 --mutation-group-id "group_2" \
--load-population --initial-prompt-source-path "../trainings/20260421_030736_Qwen-Qwen3-4B" \
> nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_5_itr_10_resume_from_20260421_030736.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" \
--max-iterations 10 --population-size 7 --mutation-group-id "group_2" \
--load-population --initial-prompt-source-path "../trainings/20260421_031151_Qwen-Qwen3-4B" \
> nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_7_itr_10_resume_from_20260421_031151.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" \
--max-iterations 10 --population-size 10 --mutation-group-id "group_2" \
--load-population --initial-prompt-source-path "../trainings/20260421_085850_Qwen-Qwen3-4B" \
> nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_10_itr_10_resume_from_20260421_085850.out 2>&1 &

## resume textual runs for 20 more addtl. iterations
nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" \
--max-iterations 20 --population-size 3 --mutation-group-id "group_2" \
--load-population --initial-prompt-source-path "../trainings/20260422_010538_Qwen-Qwen3-4B" \
> nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_3_itr_20_resume_from_20260422_010538.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" \
--max-iterations 20 --population-size 5 --mutation-group-id "group_2" \
--load-population --initial-prompt-source-path "../trainings/20260422_083309_Qwen-Qwen3-4B" \
> nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_5_itr_20_resume_from_20260422_083309.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" \
--max-iterations 20 --population-size 7 --mutation-group-id "group_2" \
--load-population --initial-prompt-source-path "../trainings/20260422_083412_Qwen-Qwen3-4B" \
> nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_7_itr_20_resume_from_20260422_083412.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" \
--max-iterations 20 --population-size 10 --mutation-group-id "group_2" \
--load-population --initial-prompt-source-path "../trainings/20260422_083500_Qwen-Qwen3-4B" \
> nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_10_itr_20_resume_from_20260422_083500.out 2>&1 &


## resume textual runs for 30 more addtl. iterations
nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" \
--max-iterations 30 --population-size 3 --mutation-group-id "group_2" \
--load-population --initial-prompt-source-path "../trainings/20260422_202830_Qwen-Qwen3-4B" \
> nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_3_itr_30_resume_from_20260422_202830.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" \
--max-iterations 30 --population-size 5 --mutation-group-id "group_2" \
--load-population --initial-prompt-source-path "../trainings/20260422_202858_Qwen-Qwen3-4B" \
> nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_5_itr_30_resume_from_20260422_202858.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" \
--max-iterations 30 --population-size 7 --mutation-group-id "group_2" \
--load-population --initial-prompt-source-path "../trainings/20260422_202907_Qwen-Qwen3-4B" \
> nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_7_itr_30_resume_from_20260422_202907.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" \
--max-iterations 30 --population-size 10 --mutation-group-id "group_2" \
--load-population --initial-prompt-source-path "../trainings/20260422_202923_Qwen-Qwen3-4B" \
> nohup_outs/nohup_v4_ps_tuning_qwen4_group2_mixed_ps_10_itr_30_resume_from_20260422_202923.out 2>&1 &

## final run on gradient based optimization 

# error here: it selected best prompt from the train set score
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260422_202830_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 7 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_final_beam5_Q5_px_0.5_r5_er_0.6_20260422_202830_node12_qwen4_final" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_final_beam5_Q5_px_0.5_r5_er_0.6_20260422_202830_node12_qwen4_final.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260422_202830_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 12 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_final_beam5_Q5_px_0.5_r5_er_0.6_20260422_202830_node12_qwen4_final_updated" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_final_beam5_Q5_px_0.5_r5_er_0.6_20260422_202830_node12_qwen4_final_updated.out 2>&1 &


###################### Completely new run on the heldout set

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" \
--max-iterations 15 --population-size 10 --mutation-group-id "group_2" > nohup_outs/nohup_v5_final_run_qwen4_group2_mixed_ps_10_itr_15.out 2>&1 &

nohup python -u validate_top_population_prompts_final_dev.py \
  ../trainings/20260430_181825_Qwen-Qwen3-4B 10 --device-map cuda:0 > nohup_outs/nohup_v5_final_run_qwen4_group2_mixed_ps_10_itr_15_final_dev_step.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260430_181825_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 13 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 7 \
  --max-total-region-tokens 15 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 3 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_final_beam5_Q5_px_0.5_r5_er_0.6_20260430_181825_node13_qwen4_final" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_final_beam5_Q5_px_0.5_r5_er_0.6_20260430_181825_node13_qwen4_final.out 2>&1 &

nohup python -u validate_top_population_prompts_final_dev.py \
  ../gradients_experiments/20260501_083138_llm_cand_sugg_final_beam5_Q5_px_0.5_r5_er_0.6_20260430_181825_node13_qwen4_final \
  5 \
  --device-map cuda:1 > nohup_outs/nohup_gradient_llm_cand_sugg_final_beam5_Q5_px_0.5_r5_er_0.6_20260430_181825_node13_qwen4_final_final_dev_step.out 2>&1 &


###################### Completely new run on the extended heldout set
nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" \
--max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/nohup_v6_final_run_qwen4_group2_mixed_ps_10_itr_20.out 2>&1 &

# test 
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260430_181825_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 13 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 30 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.7 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 3 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam3_Q1_px_0.5_r5_er_0.7_mt3_20260430_181825_node13_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam3_Q1_px_0.5_r5_er_0.7_mt3_20260430_181825_node13_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260430_181825_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 13 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.7 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 3 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam3_Q1_px_0.5_r5_er_0.7_mt5_20260430_181825_node13_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam3_Q1_px_0.5_r5_er_0.7_mt5_20260430_181825_node13_qwen4_v6test.out 2>&1 &

# n_chunk 3
nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" \
--max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/nohup_v6_final_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260430_181825_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 13 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 3 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam3_Q1_px_0.5_r5_er_0.6_mt5_20260430_181825_node13_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam3_Q1_px_0.5_r5_er_0.6_mt5_20260430_181825_node13_qwen4_v6test.out 2>&1 &

# test on node 10
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260501_232502_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 3 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam3_Q1_px_0.5_r5_er_0.6_mt5_20260501_232502_node10_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam3_Q1_px_0.5_r5_er_0.6_mt5_20260501_232502_node10_qwen4_v6test.out 2>&1 &

# n_chunk 3 + new scoring formula
nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" \
--max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/nohup_v6_final_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.5.out 2>&1 &


# test on node 3
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260501_232502_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 3 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam3_Q1_px_0.5_r5_er_0.6_mt5_20260501_232502_node3_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam3_Q1_px_0.5_r5_er_0.6_mt5_20260501_232502_node3_qwen4_v6test.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" --validation-f1-std-penalty 2.0 \
--max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/nohup_v6_final_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

# final tuning regions, expansion ratio at mt5
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260502_135706_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.5 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r3_er_0.5_mt5_20260502_135706_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r3_er_0.5_mt5_20260502_135706_node1_qwen4_v6test.out 2>&1 &


nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260502_135706_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r3_er_0.6_mt5_20260502_135706_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r3_er_0.6_mt5_20260502_135706_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260502_135706_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.7 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r3_er_0.7_mt5_20260502_135706_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r3_er_0.7_mt5_20260502_135706_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260502_135706_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.5 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.5_mt5_20260502_135706_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.5_mt5_20260502_135706_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260502_135706_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.6_mt5_20260502_135706_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.6_mt5_20260502_135706_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260502_135706_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.7 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.5 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.7_mt5_20260502_135706_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.7_mt5_20260502_135706_node1_qwen4_v6test.out 2>&1 &

  # cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.7_mt5_20260502_151848_node1_qwen4_v6test.out | grep "prompt full evaluat"
  # trainings/20260502_151848_Qwen-Qwen3-4B

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.5 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r3_er_0.5_mt5_20260502_151848_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r3_er_0.5_mt5_20260502_151848_node1_qwen4_v6test.out 2>&1 &


nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r3_er_0.6_mt5_20260502_151848_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r3_er_0.6_mt5_20260502_151848_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.7 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r3_er_0.7_mt5_20260502_151848_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r3_er_0.7_mt5_20260502_151848_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.5 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.5_mt5_20260502_151848_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.5_mt5_20260502_151848_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.6_mt5_20260502_151848_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.6_mt5_20260502_151848_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.7 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.7_mt5_20260502_151848_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.7_mt5_20260502_151848_node1_qwen4_v6test.out 2>&1 &


nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 4 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r4_er_0.6_mt5_20260502_151848_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r4_er_0.6_mt5_20260502_151848_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.5 \
  --num-edit-regions 4 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r4_er_0.5_mt5_20260502_151848_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r4_er_0.5_mt5_20260502_151848_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.6_mt3_20260502_151848_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.6_mt3_20260502_151848_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 5000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.6_mt3_sz5000_20260502_151848_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.6_mt3_sz5000_20260502_151848_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 5000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.6_mt5_sz5000_20260502_151848_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r5_er_0.6_mt5_sz5000_20260502_151848_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.5 \
  --num-edit-regions 4 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r4_er_0.5_mt3_20260502_151848_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r4_er_0.5_mt3_20260502_151848_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 5000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.5 \
  --num-edit-regions 4 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_px_0.5_r4_er_0.5_mt3_sz5000_20260502_151848_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_px_0.5_r4_er_0.5_mt3_sz5000_20260502_151848_node1_qwen4_v6test.out 2>&1 &

## lambnda tuning

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 3 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q3_stdp2.0_px_0.5_r5_er_0.6_mt5_sz10000_20260502_151848_node7_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q3_stdp2.0_px_0.5_r5_er_0.6_mt5_sz10000_20260502_151848_node7_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.75 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 3 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q3_stdp2.0_px_0.75_r5_er_0.6_mt5_sz10000_20260502_151848_node7_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q3_stdp2.0_px_0.75_r5_er_0.6_mt5_sz10000_20260502_151848_node7_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 1.0 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 3 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q3_stdp2.0_px_1.0_r5_er_0.6_mt5_sz10000_20260502_151848_node7_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q3_stdp2.0_px_1.0_r5_er_0.6_mt5_sz10000_20260502_151848_node7_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 1.25 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 3 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q3_stdp2.0_px_1.25_r5_er_0.6_mt5_sz10000_20260502_151848_node7_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q3_stdp2.0_px_1.25_r5_er_0.6_mt5_sz10000_20260502_151848_node7_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260502_151848_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 1.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 3 \
  --selection-f1-std-penalty 2.0 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q3_stdp2.0_px_1.5_r5_er_0.6_mt5_sz10000_20260502_151848_node7_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q3_stdp2.0_px_1.5_r5_er_0.6_mt5_sz10000_20260502_151848_node7_qwen4_v6test.out 2>&1 &

# new final tunining 

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260503_132832_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt5_sz10000_20260503_132832_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt5_sz10000_20260503_132832_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260503_132832_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.75 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.75_r5_er_0.6_mt5_sz10000_20260503_132832_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.75_r5_er_0.6_mt5_sz10000_20260503_132832_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260503_132832_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 1.0 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_1.0_r5_er_0.6_mt5_sz10000_20260503_132832_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_1.0_r5_er_0.6_mt5_sz10000_20260503_132832_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260503_132832_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 1.25 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_1.25_r5_er_0.6_mt5_sz10000_20260503_132832_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_1.25_r5_er_0.6_mt5_sz10000_20260503_132832_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260503_132832_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.3 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.3_r5_er_0.6_mt5_sz10000_20260503_132832_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.3_r5_er_0.6_mt5_sz10000_20260503_132832_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260503_132832_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.4 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.4_r5_er_0.6_mt5_sz10000_20260503_132832_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.4_r5_er_0.6_mt5_sz10000_20260503_132832_node1_qwen4_v6test.out 2>&1 &

## Super final runs

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split dev_9000 --eval-n-chunks 3 \
--max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/super_final_nohup_v7_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "google/gemma-3-4b-it" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split dev_9000 --eval-n-chunks 3 \
--max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/super_final_nohup_v7_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split dev_9000 --eval-n-chunks 3 \
--max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/super_final_nohup_v8_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "google/gemma-3-4b-it" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split dev_9000 --eval-n-chunks 3 \
--max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/super_final_nohup_v8_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split dev_9000 --eval-n-chunks 3 \
--max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/super_final_nohup_v9_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "google/gemma-3-4b-it" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split dev_9000 --eval-n-chunks 3 \
--max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/super_final_nohup_v9_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "google/gemma-3-4b-it" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split dev_9000 --eval-n-chunks 3 \
--max-iterations 20 --population-size 10 --mutation-group-id "group_gemma" --feedback-prompt correct_and_mistakes_v1_gemma > nohup_outs/super_final_nohup_v9.1_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "google/gemma-3-4b-it" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split dev_9000 --eval-n-chunks 3 \
--max-iterations 20 --population-size 10 --mutation-group-id "group_gemma" --feedback-prompt correct_and_mistakes_v1_gemma > nohup_outs/super_final_nohup_v9.2_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" --validation-f1-std-penalty 2.0 --dev-split dev_9000 --eval-n-chunks 3 --seed 100 \
--max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/super_final_nohup_v9.1_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0_seed100.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "google/gemma-3-4b-it" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split dev_9000 --eval-n-chunks 3 \
--max-iterations 20 --population-size 10 --mutation-group-id "group_gemma" --feedback-prompt correct_and_mistakes_v1_gemma > nohup_outs/super_final_nohup_v9.3_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

# done & final 
nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" --validation-f1-std-penalty 2.0 --dev-split dev_9000 --eval-n-chunks 3 --seed 7 \
--max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/super_final_nohup_v9.2_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0_seed100.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "google/gemma-3-4b-it" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split dev_9000 --eval-n-chunks 3 --seed 7 \
--max-iterations 20 --population-size 10 --mutation-group-id "group_gemma" --feedback-prompt correct_and_mistakes_v1_gemma > nohup_outs/super_final_nohup_v9.4_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "google/gemma-3-4b-it" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split dev_9000 --eval-n-chunks 3 --seed 7 \
--max-iterations 20 --population-size 10 --mutation-group-id "group_gemma" --feedback-prompt correct_and_mistakes_v1_gemma > nohup_outs/super_final_nohup_v9.5_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &


trainings/20260505_163211_Qwen-Qwen3-4B

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 11 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 5 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.5_r5_er_0.6_mt5_sz10000_20260505_163211_node11_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.5_r5_er_0.6_mt5_sz10000_20260505_163211_node11_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 5 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.5_r5_er_0.6_mt5_sz10000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.5_r5_er_0.6_mt5_sz10000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.6 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.6_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.6r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.4 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.4_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.4_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.7 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.7_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.7_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.08 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.08_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.08_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.1 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.1_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.1_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

###### ********* now for gemma 
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.2_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.2_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.15 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.15_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.15_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.12 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.12_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.12_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.15 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.15_r5_er_0.6_mt5_sz20000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.15_r5_er_0.6_mt5_sz20000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.15 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.15_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.15_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &


######### *** now for qwen
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.3 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.3_r5_er_0.6_mt5_sz10000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.3_r5_er_0.6_mt5_sz10000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.35 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.35_r5_er_0.6_mt5_sz10000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.3_r5_er_0.6_mt5_sz10000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.6 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.6_r5_er_0.6_mt5_sz10000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.6_r5_er_0.6_mt5_sz10000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.7 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.7_r5_er_0.6_mt5_sz10000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.7_r5_er_0.6_mt5_sz10000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.4 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.4_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.4_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.4 \
  --meta-prompt-max-new-tokens 20000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.4_r3_er_0.6_mt3_sz20000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.4_r3_er_0.6_mt3_sz20000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 20000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r3_er_0.6_mt3_sz20000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r3_er_0.6_mt3_sz20000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.7 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.7_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.7_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.6 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.6_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.6_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

##
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.8 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.8_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.8_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.9 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.9_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.9_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
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
  --selection-perplexity-lambda 0.15 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.15_r3_er_0.6_mt3_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.15_r3_er_0.6_mt3_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
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
  --selection-perplexity-lambda 0.1 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.1_r3_er_0.6_mt3_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.1_r3_er_0.6_mt3_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.1 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.1_r5_er_0.6_mt5_sz20000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.1_r5_er_0.6_mt5_sz20000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.1 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.1_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.1_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 5000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.7 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.7_r3_er_0.6_mt3_sz5000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.7_r3_er_0.6_mt3_sz5000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

##
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.08 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.08_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.08_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.20 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.2_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.2_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.4 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.4_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.4_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.3 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.3_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.3_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.125 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.125_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.125_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

##

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 15000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.7 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.7_r3_er_0.6_mt3_sz15000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.7_r3_er_0.6_mt3_sz15000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_163211_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 5000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.7 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.7_r3_er_0.6_mt3_sz5000_20260505_163211_node1_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.7_r3_er_0.6_mt3_sz5000_20260505_163211_node1_qwen4_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.0 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.0_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.0_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 5 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.05 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.05_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.05_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out 2>&1 &

# [agent_evaluate] evaluate_fn: done in 769.47s, precision=13.15±1.54, recall=34.63±6.65, f1=18.68±0.48
# [agent_evaluate] evaluate_fn: stable_f1=17.71
# [agent_evaluate] evaluate_fn: done in 1149.33s, precision=17.89±2.03, recall=31.54±5.96, f1=22.35±0.83
# [agent_evaluate] evaluate_fn: stable_f1=20.68
# [agent_evaluate] evaluate_fn: done in 684.35s, precision=6.17±0.88, recall=21.74±1.55, f1=9.54±0.90
# [agent_evaluate] evaluate_fn: stable_f1=7.75

# trainings/20260502_151848_Qwen-Qwen3-4B
# λg​∈{0.04, 0.06, 0.085, 0.12, 0.16}​
#   0.06
# 0.08
# 0.10
# 0.12


cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.5_r5_er_0.6_mt5_sz10000_20260505_163211_node1_qwen4_v6test.out  | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.3_r5_er_0.6_mt5_sz10000_20260505_163211_node1_qwen4_v6test.out  | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.6_r5_er_0.6_mt5_sz10000_20260505_163211_node1_qwen4_v6test.out  | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.7_r5_er_0.6_mt5_sz10000_20260505_163211_node1_qwen4_v6test.out  | grep "prompt full evaluat"

cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.4_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test.out  | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test.out  | grep "prompt full evaluat" # 2nd best
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.6_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test.out  | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.7_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test.out  | grep "prompt full evaluat" # best so far
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.8_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.9_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test.out | grep "prompt full evaluat"

cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.4_r3_er_0.6_mt3_sz20000_20260505_163211_node1_qwen4_v6test.out  | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r3_er_0.6_mt3_sz20000_20260505_163211_node1_qwen4_v6test.out  | grep "prompt full evaluat"

cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.7_r3_er_0.6_mt3_sz5000_20260505_163211_node1_qwen4_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.7_r3_er_0.6_mt3_sz15000_20260505_163211_node1_qwen4_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q5_stdp2.0_px_0.7_r3_er_0.6_mt3_sz10000_20260505_163211_node1_qwen4_v6test.out  | grep "prompt full evaluat"

cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.08_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.1_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.12_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.15_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.2_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.4_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.6r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.7_r5_er_0.6_mt5_sz10000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.15_r5_er_0.6_mt5_sz20000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.1_r5_er_0.6_mt5_sz20000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out  | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.15_r3_er_0.6_mt3_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.1_r3_er_0.6_mt3_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out  | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.08_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat" 
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.1_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out  | grep "prompt full evaluat" # best so far
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.15_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat" # best so far
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.2_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.3_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.125_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.4_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat" # best so far

cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.0_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat" 
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.05_r5_er_0.6_mt5_sz30000_20260505_220303_google-gemma-3-4b-it_node5_v6test.out | grep "prompt full evaluat" # best so far ***
### final runs (bullshit!!!)
nohup python -u core_trainer_evolutionary_search.py --model "google/gemma-3-4b-it" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split final_step_dev --eval-n-chunks 3 --seed 7 \
--train-samples "fs_tacred_train_non_split_original_samples.pkl" --max-iterations 20 --population-size 10 --mutation-group-id "group_gemma" --feedback-prompt correct_and_mistakes_v1_gemma > nohup_outs/super_final_nohup_v10.1_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "google/gemma-3-4b-it" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split final_step_dev --eval-n-chunks 3 --seed 100 \
--train-samples "fs_tacred_train_non_split_original_samples.pkl" --max-iterations 20 --population-size 10 --mutation-group-id "group_gemma" --feedback-prompt correct_and_mistakes_v1_gemma > nohup_outs/super_final_nohup_v10.2_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" --validation-f1-std-penalty 2.0 --dev-split final_step_dev --eval-n-chunks 3 --seed 7 \
--train-samples "fs_tacred_train_non_split_original_samples.pkl" --max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/super_final_nohup_v10.1_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

##
nohup python -u core_trainer_evolutionary_search.py --model "google/gemma-3-4b-it" --device-map "cuda:0" --validation-f1-std-penalty 2.0 --dev-split final_step_dev --eval-n-chunks 3 \
--train-samples "fs_tacred_train_non_split_original_samples.pkl" --max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/super_final_nohup_v10.0_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0_normal.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" --validation-f1-std-penalty 2.0 --dev-split final_step_dev --eval-n-chunks 3 \
--train-samples "fs_tacred_train_non_split_original_samples.pkl" --max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/super_final_nohup_v10.0_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --validation-f1-std-penalty 2.0 \
  --dev-split final_step_dev \
  --eval-n-chunks 3 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --max-iterations 20 \
  --population-size 10 \
  --mutation-group-id "group_2" \
  --resume-from-nohup-log "nohup_outs/super_final_nohup_v10.0_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out" \
  > nohup_outs/super_final_nohup_v10.0_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0_resume.out 2>&1 &

# 20260507_090049
# sometthing went wrong
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260505_220303_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 18 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.05 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split dev_9000 \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.05_r5_er_0.6_mt5_sz30000_20260507_090049_google-gemma-3-4b-it_node18_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.05_r5_er_0.6_mt5_sz30000_20260507_090049_google-gemma-3-4b-it_node18_v6test.out 2>&1 &

########################
########################
########################
########################
########################

## final#1 qwen
nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:0" --validation-f1-std-penalty 2.0 --dev-split final_step_dev --eval-n-chunks 3 \
--train-samples "fs_tacred_train_non_split_original_samples.pkl" --max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/super_final_nohup_v10.0_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

nohup python -u core_trainer_evolutionary_search.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --validation-f1-std-penalty 2.0 \
  --dev-split final_step_dev \
  --eval-n-chunks 3 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --max-iterations 20 \
  --population-size 10 \
  --mutation-group-id "group_2" \
  --resume-from-nohup-log "nohup_outs/super_final_nohup_v10.0_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out" \
  > nohup_outs/super_final_nohup_v10.0_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0_resume.out 2>&
20260508_122409

## final#2 gemma
nohup python -u core_trainer_evolutionary_search.py --model "google/gemma-3-4b-it" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split final_step_dev --eval-n-chunks 3 \
--train-samples "fs_tacred_train_non_split_original_samples.pkl" --max-iterations 20 --population-size 10 --mutation-group-id "group_gemma" --feedback-prompt correct_and_mistakes_v1_gemma > nohup_outs/super_final_nohup_v10.0_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

# final#1 gemma
nohup python -u core_trainer_evolutionary_search.py --model "google/gemma-3-4b-it" --optimizer-model "google/gemma-3-12b-it" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split final_step_dev --eval-n-chunks 3 \
--train-samples "fs_tacred_train_non_split_original_samples.pkl" --max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/super_final_nohup_v10.0_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0_normal_opt_model.out 2>&1 &
cat nohup_outs/super_final_nohup_v10.0_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0_normal_opt_model.out | grep "f1="
20260508_120655

# final#2 qwen
nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --optimizer-model "Qwen/Qwen3-14B" --device-map "cuda:2" --validation-f1-std-penalty 2.0 --dev-split final_step_dev --eval-n-chunks 3 \
--train-samples "fs_tacred_train_non_split_original_samples.pkl" --max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/super_final_nohup_v10.0_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0_normal_opt_model.out 2>&1 &
cat nohup_outs/super_final_nohup_v10.0_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0_normal_opt_model.out | grep "f1="
20260508_141358


nohup python -u core_trainer_evolutionary_search.py --model "google/gemma-3-4b-it" --optimizer-model "google/gemma-3-12b-it" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dev-split final_step_dev --eval-n-chunks 3 \
--train-samples "fs_tacred_train_non_split_original_samples.pkl" --max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/super_final_nohup_v10.1_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0_normal_opt_model.out 2>&1 &

####### runs for fewrel
#done
nohup python -u core_trainer_evolutionary_search.py --model "Qwen/Qwen3-4B" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dataset-type "fs_fewrel" --dev-split final_step_dev --eval-n-chunks 3 \
--train-samples "fs_fewrel_train_non_split_original_samples.pkl" --max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/fewrel_super_final_nohup_v10.0_run_qwen4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0.out 2>&1 &

#done
nohup python -u core_trainer_evolutionary_search.py --model "google/gemma-3-4b-it" --optimizer-model "google/gemma-3-12b-it" --device-map "cuda:1" --validation-f1-std-penalty 2.0 --dataset-type "fs_fewrel" --dev-split final_step_dev --eval-n-chunks 3 \
--train-samples "fs_fewrel_train_non_split_original_samples.pkl" --max-iterations 20 --population-size 10 --mutation-group-id "group_2" > nohup_outs/fewrel_super_final_nohup_v10.0_run_gemma4_group2_mixed_ps_10_itr_20_nchunk3_stdp2.0_normal_opt_model.out 2>&1 &

#done
nohup python -u etgpo/etgpo_full_adjusted.py \
  --task_mode relation_extraction_non_reasoning \
  --dataset fs_fewrel \
  --methods taxonomy \
  --main_model "google/gemma-3-4b-it" \
  --taxonomy_model "google/gemma-3-12b-it" \
  --device_map "cuda:2" \
  --train_samples_file "fs_fewrel_train_non_split_original_samples.pkl" \
  --dev_split final_step_dev \
  --valid_size 1000 \
  --taxonomy_runs 1 \
  --eval_runs 1 \
  --num_guidance_prompts 5 \
  --max_guidances 5 \
  --output_dir "../unified_optimization_results/" \
  --taxonomy_batch_size 3 \
  --max_new_tokens 20000 \
  > nohup_outs/fewrel_nohup_etgpo_re_taxonomy_fresh_noncluster_train_original_final_dev_5newprompts_g5_run1_gemma.out 2>&1 &

#done
nohup python -u etgpo/etgpo_full_adjusted.py \
  --task_mode relation_extraction_non_reasoning \
  --dataset fs_fewrel \
  --methods taxonomy \
  --main_model "Qwen/Qwen3-4B" \
  --taxonomy_model "Qwen/Qwen3-14B" \
  --device_map "cuda:2" \
  --train_samples_file "fs_fewrel_train_non_split_original_samples.pkl" \
  --dev_split final_step_dev \
  --valid_size 1000 \
  --taxonomy_runs 1 \
  --eval_runs 1 \
  --num_guidance_prompts 5 \
  --max_guidances 5 \
  --output_dir "../unified_optimization_results/" \
  --taxonomy_batch_size 6 \
  > nohup_outs/fewrel_nohup_etgpo_re_taxonomy_fresh_noncluster_train_original_final_dev_5newprompts_g5_run1_qwen.out 2>&1 &

# done
nohup python -u core_trainer_evoprompt_de.py \
  --model "Qwen/Qwen3-4B" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --device-map "cuda:1" \
  --max-iterations 20 \
  --population-size 5 \
  --evoprompt-train-episodes-per-iteration 1000 \
  --dataset fs_fewrel \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --prompt-open-tag "[p]" \
  --prompt-close-tag "[/p]" \
  > nohup_outs/fewrel_nohup_evoprompt_de_qwen14opt_qwen4inf_itr20_train1000_final_step_dev.out 2>&1 &

#done
nohup python -u core_trainer_evoprompt_de.py \
  --model "google/gemma-3-4b-it" \
  --optimizer-model "google/gemma-3-12b-it" \
  --device-map "cuda:1" \
  --max-iterations 20 \
  --population-size 5 \
  --evoprompt-train-episodes-per-iteration 1000 \
  --dataset fs_fewrel \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --prompt-open-tag "[p]" \
  --prompt-close-tag "[/p]" \
  > nohup_outs/fewrel_nohup_evoprompt_de_gemma12opt_gemma4inf_itr20_train1000_final_step_dev.out 2>&1 &

#done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:2" \
  --prompt-source scratch \
  --dataset-type "fs_fewrel" \
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
  --n-steps 5 \
  --eval-every 1 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_scratch_5steps_qwen4" > nohup_outs/fewrel_scratch_5steps_qwen4_greater_scratch_5steps_n10000_lmd0.2_k25_u10_z5_qwen4.out 2>&1 &
#../greater_experiments/20260518_015356_fewrel_scratch_5steps_qwen4

# done again
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:2" \
  --prompt-source scratch \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
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
  --n-steps 5 \
  --eval-every 1 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_scratch_5steps_gemma4_n30000" > nohup_outs/fewrel_scratch_5steps_gemma4_greater_scratch_5steps_n30000_lmd0.2_k25_u10_z5_gemma4.out 2>&1 &

#done #scratch
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:1" \
  --prompt-source scratch \
  --dataset-type "fs_fewrel" \
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
  --n-steps 10 \
  --eval-every 1 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_scratch_10steps_qwen4_resume" \
  --resume-out-file "nohup_outs/fewrel_scratch_5steps_qwen4_greater_scratch_5steps_n10000_lmd0.2_k25_u10_z5_qwen4.out" \
  > nohup_outs/fewrel_scratch_5steps_qwen4_greater_scratch_10steps_n10000_lmd0.2_k25_u10_z5_qwen4_resume.out 2>&1 &

# done #scratch
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:1" \
  --prompt-source scratch \
  --position-selection-mode top_gradient \
  --dataset-type "fs_fewrel" \
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
  --n-steps 10 \
  --eval-every 1 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_scratch_10steps_qwen4_top_grad" > nohup_outs/fewrel_scratch_10steps_qwen4_greater_scratch_5steps_n10000_lmd0.2_k25_u10_z5_qwen4_top_grad.out 2>&1 &

# done #scratch
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:2" \
  --prompt-source scratch \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
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
  --n-steps 10 \
  --eval-every 1 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_scratch_10steps_gemma4_n30000_resume" \
  --resume-out-file "nohup_outs/fewrel_scratch_5steps_gemma4_greater_scratch_5steps_n30000_lmd0.2_k25_u10_z5_gemma4.out" \
  > nohup_outs/fewrel_scratch_10steps_gemma4_greater_scratch_10steps_n30000_lmd0.2_k25_u10_z5_gemma4_resume.out 2>&1 &

#done #scratch
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:1" \
  --prompt-source scratch \
  --position-selection-mode top_gradient \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
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
  --n-steps 10 \
  --eval-every 1 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_scratch_10steps_gemma4_top_grad" \
  > nohup_outs/fewrel_scratch_10steps_gemma4_greater_scratch_10steps_n30000_lmd0.2_k25_u10_z5_gemma4_top_grad.out 2>&1 &

# nohup python -u GreaTer/experiments/relation_extraction_greater.py \
#   --model "Qwen/Qwen3-4B" \
#   --device-map "cuda:2" \
#   --prompt-source scratch \
#   --dataset-type "fs_fewrel" \
#   --train-gradient-sample-size 10000 \
#   --gradient-batch-size 4 \
#   --selection-batch-size 8 \
#   --proposal-example-size 50 \
#   --proposal-top-k 25 \
#   --proposal-min-candidates 10 \
#   --selection-top-mu 10 \
#   --top-z 5 \
#   --dev-f1-std-penalty 2.0 \
#   --fluency-lambda 0.2 \
#   --fluency-scope instruction \
#   --fluency-metric nll \
#   --n-steps 5 \
#   --eval-every 1 \
#   --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
#   --full-eval-split final_step_dev \
#   --output-root-dir "../greater_experiments" \
#   --output-substring "fewrel_scratch_5steps_qwen4" > nohup_outs/fewrel_scratch_5steps_qwen4_greater_scratch_5steps_n10000_lmd0.2_k25_u10_z5_qwen4.out 2>&1 &
#   --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
#   --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
######################################################## Grads run below

### currently
# done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --prompt-source scratch \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 4 \
  --selection-batch-size 8 \
  --proposal-example-size 50 \
  --proposal-top-k 10 \
  --proposal-min-candidates 3 \
  --selection-top-mu 3 \
  --fluency-lambda 0.2 \
  --fluency-scope instruction \
  --fluency-metric nll \
  --n-steps 5 \
  --eval-every 1 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "scratch_5steps_qwen4" > nohup_outs/greater_scratch_5steps_n10000_lmd0.2_u3_qwen4.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_141358_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r3_er_0.6_mt3_sz10000_20260508_141358_node2_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r3_er_0.6_mt3_sz10000_20260508_141358_node2_qwen4_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_141358_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r5_er_0.6_mt3_sz10000_20260508_141358_node2_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r5_er_0.6_mt3_sz10000_20260508_141358_node2_qwen4_v6test.out 2>&1 &

# stopped
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.05 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.05_r5_er_0.6_mt5_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.05_r5_er_0.6_mt5_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

# stopped
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.1 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r5_er_0.6_mt5_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r5_er_0.6_mt5_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r5_er_0.6_mt5_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out | grep "prompt full evaluation"
cat nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.05_r5_er_0.6_mt5_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out | grep "prompt full evaluation"

# done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --prompt-source scratch \
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
  --n-steps 5 \
  --eval-every 1 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "scratch_5steps_qwen4" > nohup_outs/greater_scratch_5steps_n10000_lmd0.2_k25_u10_z5_qwen4.out 2>&1 &

# done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --prompt-source scratch \
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
  --fluency-lambda 0.5 \
  --fluency-scope instruction \
  --fluency-metric nll \
  --n-steps 5 \
  --eval-every 1 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "scratch_5steps_qwen4" > nohup_outs/greater_scratch_5steps_n10000_lmd0.5_k25_u10_z5_qwen4.out 2>&1 &

#done best so far
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --prompt-source scratch \
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
  --fluency-lambda 1.0 \
  --fluency-scope instruction \
  --fluency-metric nll \
  --n-steps 5 \
  --eval-every 1 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "scratch_5steps_qwen4" > nohup_outs/greater_scratch_5steps_n10000_lmd1.0_k25_u10_z5_qwen4.out 2>&1 &

#done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --prompt-source scratch \
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
  --fluency-lambda 2.0 \
  --fluency-scope instruction \
  --fluency-metric nll \
  --n-steps 5 \
  --eval-every 1 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "scratch_5steps_qwen4" > nohup_outs/greater_scratch_5steps_n10000_lmd2.0_k25_u10_z5_qwen4.out 2>&1 &
#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.1 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r5_er_0.6_mt5_sz20000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r5_er_0.6_mt5_sz20000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.1 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r5_er_0.6_mt5_sz10000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r5_er_0.6_mt5_sz10000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#done
  nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.2_r5_er_0.6_mt5_sz10000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.2_r5_er_0.6_mt5_sz10000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 5 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.3 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.3_r5_er_0.6_mt5_sz10000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.3_r5_er_0.6_mt5_sz10000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#stopped
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.1 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r3_er_0.6_mt3_sz10000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r3_er_0.6_mt3_sz10000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#stopped
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.2_r3_er_0.6_mt3_sz10000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.2_r3_er_0.6_mt3_sz10000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

# done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r3_er_0.6_mt3_sz10000_20260508_122409_node2_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r3_er_0.6_mt3_sz10000_20260508_122409_node2_qwen4_v6test.out 2>&1 &

# done Final**
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r5_er_0.6_mt3_sz10000_20260508_122409_node2_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r5_er_0.6_mt3_sz10000_20260508_122409_node2_qwen4_v6test.out 2>&1 &

# done 
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
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
  --selection-perplexity-lambda 0.1 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

# done 
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
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
  --selection-perplexity-lambda 0.2 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.2_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.2_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

# done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
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
  --selection-perplexity-lambda 0.15 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.15_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.15_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

# stopped
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.15 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.15_r5_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.15_r5_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 50000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.1 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r3_er_0.6_mt3_sz50000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r3_er_0.6_mt3_sz50000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 70000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.1 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r3_er_0.6_mt3_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r3_er_0.6_mt3_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
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
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
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
  --selection-perplexity-lambda 1.0 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_1.0_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_1.0_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#stopped
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r5_er_0.6_mt3_sz10000_20260508_122409_node10_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r5_er_0.6_mt3_sz10000_20260508_122409_node10_qwen4_v6test.out 2>&1 &

#stopped
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r3_er_0.6_mt3_sz10000_20260508_122409_node10_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r3_er_0.6_mt3_sz10000_20260508_122409_node10_qwen4_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 70000 \
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
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 70000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_log_r5_er_0.6_mt3_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_log_r5_er_0.6_mt3_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:3" \
  --prompt-source scratch \
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
  --fluency-lambda 1.0 \
  --fluency-scope instruction \
  --fluency-metric nll \
  --n-steps 5 \
  --eval-every 1 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "scratch_5steps_qwen4" > nohup_outs/greater_scratch_5steps_n10000_lmd1.0_k25_u10_z5_gemma4.out 2>&1 &

#done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:3" \
  --prompt-source scratch \
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
  --n-steps 5 \
  --eval-every 1 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "scratch_5steps_qwen4" > nohup_outs/greater_scratch_5steps_n10000_lmd0.2_k25_u10_z5_gemma4.out 2>&1 &

#stopped
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r5_er_0.6_mt3_sz10000_20260508_122409_node10_qwen4_v6test2" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r5_er_0.6_mt3_sz10000_20260508_122409_node10_qwen4_v6test2.out 2>&1 &

#stopped
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r3_er_0.6_mt3_sz10000_20260508_122409_node10_qwen4_v6test2" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_r3_er_0.6_mt3_sz10000_20260508_122409_node10_qwen4_v6test2.out 2>&1 &

#stopped
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 70000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
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
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt1_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt1_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#stopped
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 70000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
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
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz10000_20260508_122409_node10_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz10000_20260508_122409_node10_qwen4_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node10_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node10_qwen4_v6test.out 2>&1 &

#done
nohup python -u core_trainer_evoprompt_de.py \
  --model "Qwen/Qwen3-4B" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --device-map "cuda:3" \
  --max-iterations 5 \
  --population-size 5 \
  --evoprompt-train-episodes-per-iteration 1000 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --prompt-open-tag "[p]" \
  --prompt-close-tag "[/p]" \
  > nohup_outs/nohup_evoprompt_de_qwen14opt_qwen4inf_itr5_train1000_final_step_dev.out 2>&1 &

#done
nohup python -u core_trainer_evoprompt_de.py \
  --model "google/gemma-3-4b-it" \
  --optimizer-model "google/gemma-3-12b-it" \
  --device-map "cuda:3" \
  --max-iterations 5 \
  --population-size 5 \
  --evoprompt-train-episodes-per-iteration 1000 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --prompt-open-tag "[p]" \
  --prompt-close-tag "[/p]" \
  > nohup_outs/nohup_evoprompt_de_gemma12opt_gemma4inf_itr5_train1000_final_step_dev.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 70000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
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
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt2_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt2_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 70000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
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
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 10000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_r5_er_0.6_mt2_sz10000_20260508_122409_node10_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_r5_er_0.6_mt2_sz10000_20260508_122409_node10_qwen4_v6test.out 2>&1 &

# done
nohup python -u core_trainer_evoprompt_de.py \
  --model "Qwen/Qwen3-4B" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --device-map "cuda:3" \
  --max-iterations 10 \
  --population-size 5 \
  --evoprompt-train-episodes-per-iteration 1000 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --prompt-open-tag "[p]" \
  --prompt-close-tag "[/p]" \
  --evoprompt-resume-from-nohup "nohup_outs/nohup_evoprompt_de_qwen14opt_qwen4inf_itr5_train1000_final_step_dev.out" \
  > nohup_outs/nohup_evoprompt_de_qwen14opt_qwen4inf_itr10_train1000_final_step_dev_resume.out 2>&1 &


#done
nohup python -u core_trainer_evoprompt_de.py \
  --model "google/gemma-3-4b-it" \
  --optimizer-model "google/gemma-3-12b-it" \
  --device-map "cuda:3" \
  --max-iterations 10 \
  --population-size 5 \
  --evoprompt-train-episodes-per-iteration 1000 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --prompt-open-tag "[p]" \
  --prompt-close-tag "[/p]" \
  --evoprompt-resume-from-nohup "nohup_outs/nohup_evoprompt_de_gemma12opt_gemma4inf_itr5_train1000_final_step_dev.out" \
  > nohup_outs/nohup_evoprompt_de_gemma12opt_gemma4inf_itr10_train1000_final_step_dev_resume.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_r5_er_0.6_mt2_sz20000_20260508_122409_node10_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_r5_er_0.6_mt2_sz20000_20260508_122409_node10_qwen4_v6test.out 2>&1 &


# done
nohup python -u core_trainer_evoprompt_de.py \
  --model "Qwen/Qwen3-4B" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --device-map "cuda:3" \
  --max-iterations 20 \
  --population-size 5 \
  --evoprompt-train-episodes-per-iteration 1000 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --prompt-open-tag "[p]" \
  --prompt-close-tag "[/p]" \
  --evoprompt-resume-from-nohup "nohup_outs/nohup_evoprompt_de_qwen14opt_qwen4inf_itr10_train1000_final_step_dev_resume.out" \
  > nohup_outs/nohup_evoprompt_de_qwen14opt_qwen4inf_itr20_train1000_final_step_dev_resume.out 2>&1 &

# done
nohup python -u core_trainer_evoprompt_de.py \
  --model "google/gemma-3-4b-it" \
  --optimizer-model "google/gemma-3-12b-it" \
  --device-map "cuda:3" \
  --max-iterations 20 \
  --population-size 5 \
  --evoprompt-train-episodes-per-iteration 1000 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --prompt-open-tag "[p]" \
  --prompt-close-tag "[/p]" \
  --evoprompt-resume-from-nohup "nohup_outs/nohup_evoprompt_de_gemma12opt_gemma4inf_itr10_train1000_final_step_dev_resume.out" \
  > nohup_outs/nohup_evoprompt_de_gemma12opt_gemma4inf_itr20_train1000_final_step_dev_resume.out 2>&1 &

#done #scratch
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:3" \
  --prompt-source scratch \
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
  --n-steps 10 \
  --eval-every 1 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "resume_10steps_lmd0.2_gemma4" \
  --resume-out-file "nohup_outs/greater_scratch_5steps_n10000_lmd0.2_k25_u10_z5_gemma4.out" \
  > nohup_outs/greater_resume_10steps_n10000_lmd0.2_k25_u10_z5_gemma4.out 2>&1 &

#done #scratch
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --prompt-source scratch \
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
  --n-steps 10 \
  --eval-every 1 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "resume_10steps_lmd0.2_qwen4" \
  --resume-out-file "nohup_outs/greater_scratch_5steps_n10000_lmd0.2_k25_u10_z5_qwen4.out" \
  > nohup_outs/greater_resume_10steps_n10000_lmd0.2_k25_u10_z5_qwen4.out 2>&1 &

#scratch running
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --prompt-source scratch \
  --dataset-type "fs_tacred" \
  --position-selection-mode top_gradient \
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
  --n-steps 10 \
  --eval-every 1 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "greater_topgrad_scratch_10steps_qwen4" \
  > nohup_outs/greater_topgrad_scratch_10steps_qwen4.out 2>&1 &

#scratch running
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:2" \
  --prompt-source scratch \
  --dataset-type "fs_tacred" \
  --position-selection-mode top_gradient \
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
  --n-steps 10 \
  --eval-every 1 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "greater_topgrad_scratch_10steps_gemma4" \
  > nohup_outs/greater_topgrad_scratch_10steps_gemma4.out 2>&1 &

#scratch running
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 0 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 10 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_cand_sugg_scratch_beam5_qwen4" \
  > nohup_outs/superstart_llm_cand_sugg_scratch_beam5_qwen4.out 2>&1 &

#scratch running
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 0 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 10 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_llm_cand_sugg_scratch_beam5_qwen4" \
  > nohup_outs/fewrel_superstart_llm_cand_sugg_scratch_beam5_qwen4.out 2>&1 &

#scratch running
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 0 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
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
  --output-substring "superstart_llm_cand_sugg_scratch_beam5_gemma4" \
  > nohup_outs/superstart_llm_cand_sugg_scratch_beam5_gemma4.out 2>&1 &

#scratch next
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 0 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_llm_cand_sugg_scratch_beam5_gemma4" \
  > nohup_outs/fewrel_superstart_llm_cand_sugg_scratch_beam5_gemma4.out 2>&1 &

#scratch later
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 0 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_prob_scratch_beam5_qwen4" \
  > nohup_outs/superstart_llm_prob_scratch_beam5_qwen4.out 2>&1 &

#done 
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_r5_er_0.6_mt2_sz20000_20260508_122409_node2_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_r5_er_0.6_mt2_sz20000_20260508_122409_node2_qwen4_v6test.out 2>&1 &

#done 
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node2_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node2_qwen4_v6test.out 2>&1 &


#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 70000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
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
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
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
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt2_sz20000_20260508_122409_node2_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt2_sz20000_20260508_122409_node2_qwen4_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node2_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node2_qwen4_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
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
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node18_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node18_v6test.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 18 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 70000 \
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
  --Q 2 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz70000_20260508_120655_google-gemma-3-4b-it_node18_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz70000_20260508_120655_google-gemma-3-4b-it_node18_v6test.out 2>&1 &

#done
nohup python -u core_trainer_evoprompt_de.py \
  --model "Qwen/Qwen3-4B" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --device-map "cuda:2" \
  --max-iterations 5 \
  --population-size 5 \
  --evoprompt-train-episodes-per-iteration 1000 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --prompt-open-tag "[p]" \
  --prompt-close-tag "[/p]" \
  > nohup_outs/nohup_evoprompt_de_qwen14opt_qwen4inf_itr5_train1000_final_step_dev2.out 2>&1 &

#done
nohup python -u core_trainer_evoprompt_de.py \
  --model "google/gemma-3-4b-it" \
  --optimizer-model "google/gemma-3-12b-it" \
  --device-map "cuda:2" \
  --max-iterations 5 \
  --population-size 5 \
  --evoprompt-train-episodes-per-iteration 1000 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --prompt-open-tag "[p]" \
  --prompt-close-tag "[/p]" \
  > nohup_outs/nohup_evoprompt_de_gemma12opt_gemma4inf_itr5_train1000_final_step_dev2.out 2>&1 &

#done
nohup python -u core_trainer_evoprompt_de.py \
  --model "Qwen/Qwen3-4B" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --device-map "cuda:1" \
  --max-iterations 20 \
  --population-size 5 \
  --evoprompt-train-episodes-per-iteration 1000 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --prompt-open-tag "[p]" \
  --prompt-close-tag "[/p]" \
  --evoprompt-resume-from-nohup "nohup_outs/nohup_evoprompt_de_qwen14opt_qwen4inf_itr5_train1000_final_step_dev2.out" \
  > nohup_outs/nohup_evoprompt_de_qwen14opt_qwen4inf_itr20_train1000_final_step_dev2_resume.out 2>&1 &

#done
nohup python -u core_trainer_evoprompt_de.py \
  --model "google/gemma-3-4b-it" \
  --optimizer-model "google/gemma-3-12b-it" \
  --device-map "cuda:1" \
  --max-iterations 20 \
  --population-size 5 \
  --evoprompt-train-episodes-per-iteration 1000 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --prompt-open-tag "[p]" \
  --prompt-close-tag "[/p]" \
  --evoprompt-resume-from-nohup "nohup_outs/nohup_evoprompt_de_gemma12opt_gemma4inf_itr5_train1000_final_step_dev2.out" \
  > nohup_outs/nohup_evoprompt_de_gemma12opt_gemma4inf_itr20_train1000_final_step_dev2_resume.out 2>&1 &

#done
nohup python -u etgpo/etgpo_full_adjusted.py \
  --task_mode relation_extraction_non_reasoning \
  --dataset fs_tacred \
  --methods taxonomy \
  --main_model "Qwen/Qwen3-4B" \
  --taxonomy_model "Qwen/Qwen3-14B" \
  --device_map "cuda:1" \
  --train_samples_file "fs_tacred_train_non_split_original_samples.pkl" \
  --dev_split final_step_dev \
  --valid_size 1000 \
  --taxonomy_runs 1 \
  --eval_runs 1 \
  --num_guidance_prompts 5 \
  --max_guidances 7 \
  --output_dir "../unified_optimization_results/" \
  --taxonomy_batch_size 6 \
  > nohup_outs/nohup_etgpo_re_taxonomy_fresh_noncluster_train_original_final_dev_5newprompts_g7_run1.out 2>&1 &

#done
nohup python -u etgpo/etgpo_full_adjusted.py \
  --task_mode relation_extraction_non_reasoning \
  --dataset fs_tacred \
  --methods taxonomy \
  --main_model "Qwen/Qwen3-4B" \
  --taxonomy_model "Qwen/Qwen3-14B" \
  --device_map "cuda:1" \
  --train_samples_file "fs_tacred_train_non_split_original_samples.pkl" \
  --dev_split final_step_dev \
  --valid_size 1000 \
  --taxonomy_runs 1 \
  --eval_runs 1 \
  --num_guidance_prompts 5 \
  --max_guidances 5 \
  --output_dir "../unified_optimization_results/" \
  --taxonomy_batch_size 6 \
  > nohup_outs/nohup_etgpo_re_taxonomy_fresh_noncluster_train_original_final_dev_5newprompts_g5_run1.out 2>&1 &

#done best etgpo for gemma
nohup python -u etgpo/etgpo_full_adjusted.py \
  --task_mode relation_extraction_non_reasoning \
  --dataset fs_tacred \
  --methods taxonomy \
  --main_model "google/gemma-3-4b-it" \
  --taxonomy_model "google/gemma-3-12b-it" \
  --device_map "cuda:1" \
  --train_samples_file "fs_tacred_train_non_split_original_samples.pkl" \
  --dev_split final_step_dev \
  --valid_size 1000 \
  --taxonomy_runs 1 \
  --eval_runs 1 \
  --num_guidance_prompts 5 \
  --max_guidances 5 \
  --output_dir "../unified_optimization_results/" \
  --taxonomy_batch_size 3 \
  --max_new_tokens 20000 \
  > nohup_outs/nohup_etgpo_re_taxonomy_fresh_noncluster_train_original_final_dev_5newprompts_g5_run1_gemma.out 2>&1 &

#done
nohup python -u etgpo/etgpo_full_adjusted.py \
  --task_mode relation_extraction_non_reasoning \
  --dataset fs_tacred \
  --methods taxonomy \
  --main_model "google/gemma-3-4b-it" \
  --taxonomy_model "google/gemma-3-12b-it" \
  --device_map "cuda:1" \
  --train_samples_file "fs_tacred_train_non_split_original_samples.pkl" \
  --dev_split final_step_dev \
  --valid_size 1000 \
  --taxonomy_runs 1 \
  --eval_runs 1 \
  --num_guidance_prompts 7 \
  --max_guidances 5 \
  --output_dir "../unified_optimization_results/" \
  --taxonomy_batch_size 3 \
  --max_new_tokens 20000 \
  > nohup_outs/nohup_etgpo_re_taxonomy_fresh_noncluster_train_original_final_dev_5newprompts_g7_run1_gemma.out 2>&1 &

#done
nohup python -u etgpo/etgpo_full_adjusted.py \
  --task_mode relation_extraction_non_reasoning \
  --dataset fs_tacred \
  --methods taxonomy \
  --main_model "google/gemma-3-4b-it" \
  --taxonomy_model "google/gemma-3-12b-it" \
  --device_map "cuda:1" \
  --train_samples_file "fs_tacred_train_non_split_original_samples.pkl" \
  --dev_split final_step_dev \
  --valid_size 1000 \
  --taxonomy_runs 1 \
  --eval_runs 1 \
  --num_guidance_prompts 5 \
  --max_guidances 7 \
  --output_dir "../unified_optimization_results/" \
  --taxonomy_batch_size 3 \
  --max_new_tokens 20000 \
  > nohup_outs/nohup_etgpo_re_taxonomy_fresh_noncluster_train_original_final_dev_5newprompts_g7_run1_gemma_real.out 2>&1 &

#done
nohup python -u etgpo/etgpo_full_adjusted.py \
  --task_mode relation_extraction_non_reasoning \
  --dataset fs_tacred \
  --methods taxonomy \
  --main_model "google/gemma-3-4b-it" \
  --taxonomy_model "google/gemma-3-12b-it" \
  --device_map "cuda:1" \
  --train_samples_file "fs_tacred_train_non_split_original_samples.pkl" \
  --dev_split final_step_dev \
  --valid_size 1000 \
  --taxonomy_runs 1 \
  --eval_runs 1 \
  --num_guidance_prompts 5 \
  --max_guidances 7 \
  --output_dir "../unified_optimization_results/" \
  --taxonomy_batch_size 3 \
  --max_new_tokens 20000 \
  > nohup_outs/nohup_etgpo_re_taxonomy_fresh_noncluster_train_original_final_dev_5newprompts_g7_run1_gemma_real2.out 2>&1 &

#done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --position-selection-mode top_gradient \
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
  --output-substring "topgrad_node2_1steps_lmd0.2_qwen4" \
  > nohup_outs/greater_topgrad_node2_1steps_n10000_lmd0.2_k25_u10_z5_qwen4.out 2>&1 &

#done ### need to read the nohup out file only
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --position-selection-mode top_gradient \
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
  --output-substring "topgrad_node2_1steps_lmd0.2_qwen4" \
  > nohup_outs/greater_topgrad_node10_1steps_n10000_lmd0.2_k25_u10_z5_qwen4.out 2>&1 &

#done
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
  --output-substring "topgrad_node2_1steps_lmd0.2_qwen4" \
  > nohup_outs/greater_seq_node2_1steps_n10000_lmd0.2_k25_u10_z5_qwen4.out 2>&1 &
  
#done
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
  --output-substring "topgrad_node10_1steps_lmd0.2_qwen4" \
  > nohup_outs/greater_seq_node10_1steps_n10000_lmd0.2_k25_u10_z5_qwen4.out 2>&1 &

#done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
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
  --output-substring "topgrad_node3_1steps_lmd0.2_gemma4" \
  > nohup_outs/greater_seq_node3_1steps_n30000_lmd0.2_k25_u10_z5_gemma4.out 2>&1 &

#done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 18 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
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
  --output-substring "topgrad_node18_1steps_lmd0.2_gemma4" \
  > nohup_outs/greater_seq_node18_1steps_n30000_lmd0.2_k25_u10_z5_gemma4.out 2>&1 &

#done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
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
  --output-substring "topgrad2_node3_1steps_lmd0.2_gemma4" \
  > nohup_outs/greater_topgrad_node3_1steps_n30000_lmd0.2_k25_u10_z5_gemma4.out 2>&1 &

#done
  nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 18 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
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
  --output-substring "topgrad2_node18_1steps_lmd0.2_gemma4" \
  > nohup_outs/greater_topgrad_node18_1steps_n30000_lmd0.2_k25_u10_z5_gemma4.out 2>&1 &

#done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 18 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
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
  --output-substring "topgrad2_node18_1steps_lmd0.2_gemma4_" \
  > nohup_outs/greater_topgrad_node18_1steps_n30000_lmd0.2_k25_u10_z5_gemma4_.out 2>&1 &

#done
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:1" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --optimizer-device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-feedback-sample-size 10000 \
  --feedback-example-size 5 \
  --mistake-example-size 5 \
  --n-steps 1 \
  --max-locations 5 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 5 \
  --num-candidates 10 \
  --top-z 5 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "lpo_roleswitch_node2_step1_qwen4_optimizer" \
  > nohup_outs/lpo_roleswitch_node2_step1_n100_qwen4_optqwen14.out 2>&1 &

#done
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:1" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --optimizer-device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-feedback-sample-size 10000 \
  --feedback-example-size 5 \
  --mistake-example-size 5 \
  --n-steps 1 \
  --max-locations 5 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 5 \
  --num-candidates 8 \
  --top-z 5 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "lpo_roleswitch_node10_step1_qwen4_optimizer" \
  > nohup_outs/lpo_roleswitch_node10_step1_n100_qwen4_optqwen14.out 2>&1 &

#done
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:1" \
  --optimizer-model "google/gemma-3-12b-it" \
  --optimizer-device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-feedback-sample-size 30000 \
  --feedback-example-size 5 \
  --mistake-example-size 5 \
  --n-steps 1 \
  --max-locations 5 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 5 \
  --num-candidates 8 \
  --top-z 5 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "lpo_roleswitch_node3_step1_gemma4_optimizer" \
  > nohup_outs/lpo_roleswitch_node3_step1_n100_gemma4_optgemma12.out 2>&1 &

#done
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:1" \
  --optimizer-model "google/gemma-3-12b-it" \
  --optimizer-device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 18 \
  --dataset-type "fs_tacred" \
  --train-feedback-sample-size 30000 \
  --feedback-example-size 5 \
  --mistake-example-size 5 \
  --n-steps 1 \
  --max-locations 5 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 5 \
  --num-candidates 8 \
  --top-z 5 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "lpo_roleswitch_node18_step1_gemma4_optimizer" \
  > nohup_outs/lpo_roleswitch_node18_step1_n100_gemma4_optgemma12.out 2>&1 &

#done
nohup python -u core_trainer_evoprompt_de.py \
  --model "Qwen/Qwen3-4B" \
  --optimizer-model "Qwen/Qwen3-4B" \
  --device-map "cuda:1" \
  --max-iterations 20 \
  --population-size 5 \
  --evoprompt-train-episodes-per-iteration 1000 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --prompt-open-tag "[p]" \
  --prompt-close-tag "[/p]" \
  > nohup_outs/nohup_evoprompt_de_self_opt_qwen4inf_itr20_train1000.out 2>&1 &

#done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
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
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test_final_updated_code" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test_final_updated_code.out 2>&1 &

#done
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

#done ## run again
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node10_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node10_qwen4_v6test.out 2>&1 &

#done - great!!
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
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
  --output-substring "llm_prob_sugg_beam5_C7_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node18_v6test_final_updated_code" \
  > nohup_outs/nohup_gradient_llm_prob_sugg_beam5_C7_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node18_v6test_final_updated_code.out 2>&1 &

# done & need to check & run again
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
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
  --output-substring "llm_prob_sugg_beam5_C7_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test_final_updated_code" \
  > nohup_outs/nohup_gradient_llm_prob_sugg_beam5_C7_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test_final_updated_code.out 2>&1 &

# done try again
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node10_qwen4_v6test2" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node10_qwen4_v6test2.out 2>&1 &

# done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_prob_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node10_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_prob_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node10_qwen4_v6test.out 2>&1 &

# done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_prob_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node2_qwen4_v6test" \
  > nohup_outs/nohup_gradient_llm_prob_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node2_qwen4_v6test.out 2>&1 &

# done - rerun # need a rerun perhaps a lower mt value
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
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
  --output-substring "llm_prob_sugg_beam5_C7_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test_final_updated_code2" \
  > nohup_outs/nohup_gradient_llm_prob_sugg_beam5_C7_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test_final_updated_code2.out 2>&1 &

#done - rerun
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node10_qwen4_v6test3" \
  > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_r5_er_0.6_mt1_sz20000_20260508_122409_node10_qwen4_v6test3.out 2>&1 &

  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r5_er_0.6_mt5_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r5_er_0.6_mt5_sz20000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r5_er_0.6_mt5_sz10000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.2_r5_er_0.6_mt5_sz10000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.3_r5_er_0.6_mt5_sz10000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r3_er_0.6_mt3_sz10000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.2_r3_er_0.6_mt3_sz10000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.2_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.15_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.15_r5_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r3_er_0.6_mt3_sz50000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.1_r3_er_0.6_mt3_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_1.0_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C7_Q2_stdp2.0_px_0.5_log_r5_er_0.6_mt3_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt1_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt2_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &
  # > nohup_outs/nohup_gradient_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_sz70000_20260508_120655_google-gemma-3-4b-it_node3_v6test.out 2>&1 &



##### Superstar tuning



# done - worked
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_sz20000_20260508_122409_node2_qwen4_v7" \
  > nohup_outs/superstart_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_sz20000_20260508_122409_node2_qwen4_v7.out 2>&1 &

#aunabil1 colab done - not good
python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_sz20000_20260508_122409_node2_qwen4_v7" \

#tanvir1 colab  done - worked
python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_sz20000_20260508_122409_node10_qwen4_v7" \

#tanvir2 colab done worked for both nodes (GEN mode)
python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_sz20000_20260508_122409_node10_qwen4_v7" \

#aunabil2 colab done
python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_sz20000_20260508_122409_node2_qwen4_v7"

#tanvir3 colab done
python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_sz20000_20260508_122409_node10_qwen4_v7"

# # 
# nohup python -u agents/agent_gradient_eval_debug.py \
#   --model "Qwen/Qwen3-4B" \
#   --mode "LLM_CANDIDATE_SUGGESTION" \
#   --device-map "cuda:1" \
#   --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
#   --prompt-node-id 2 \
#   --dataset-type "fs_fewrel" \
#   --train-gradient-sample-size 20000 \
#   --gradient-batch-size 2 \
#   --max-regions 10 \
#   --max-total-region-tokens 50 \
#   --max-region-tokens 2 \
#   --region-expansion-threshold-ratio 0.6 \
#   --num-edit-regions 5 \
#   --num-region-candidates 5 \
#   --beam-width 5 \
#   --selection-perplexity-lambda 0.5 \
#   --use-log-fluency-score \
#   --meta-prompt-max-new-tokens 10000 \
#   --meta-prompt-batch-size 1 \
#   --validation-batch-size 8 \
#   --Q 1 \
#   --selection-f1-std-penalty 2.0 \
#   --full-eval-split final_step_dev \
#   --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
#   --output-root-dir "../gradients_experiments" \
#   --output-substring "superstart_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_sz20000_20260508_122409_node2_qwen4_v7" \
#   > nohup_outs/superstart_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_sz20000_20260508_122409_node2_qwen4_v7.out 2>&1 &

#aunabil3 colab done
aunabil2 -> mt 1
#aunabil4 colab done
aunabil2 -> mt 3

#tanvir4 colab done
tanvir3 -> mt 1
#tanvir5 colab done
tanvir3 -> mt 3 

# done poor candidates with 0.9
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_prob_beam5_C7_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_top0.9_sz20000_20260508_122409_node2_qwen4_v8" \
  > nohup_outs/superstart_llm_prob_beam5_C7_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_top0.9_sz20000_20260508_122409_node2_qwen4_v8.out 2>&1 &

# done poor candidates with 0.9
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 7 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_prob_beam5_C7_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt3_top0.9_sz20000_20260508_122409_node2_qwen4_v8" \
  > nohup_outs/superstart_llm_prob_beam5_C7_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt3_top0.9_sz20000_20260508_122409_node2_qwen4_v8.out 2>&1 &

# aunabil g1 - worked for both nodes (GEN mode)
python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
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
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v8"

# aunabil g2
python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
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
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt2_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v8"

# tanvir g1
python -u agents/agent_gradient_eval_debug.py \
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
  --num-region-candidates 5 \
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
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node18_v8"

# tanvir g2 
python -u agents/agent_gradient_eval_debug.py \
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
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
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
  --output-substring "llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt2_sz30000_20260508_120655_google-gemma-3-4b-it_node18_v8"


# done with 0.95
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_top0.95_sz20000_20260508_122409_node2_qwen4_v8" \
  > nohup_outs/superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_top0.95_sz20000_20260508_122409_node2_qwen4_v8.out 2>&1 &


# done with 0.95
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz20000_20260508_122409_node2_qwen4_v8" \
  > nohup_outs/superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz20000_20260508_122409_node2_qwen4_v8.out 2>&1 &


# aunabil g3 done
python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
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
  --output-substring "llm_cand_sugg_beam5_C7_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v8"

# aunabil g4
python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
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
  --output-substring "llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_top0.95_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v8"

# tanvir g3
python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
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
  --num-region-candidates 5 \
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
  --output-substring "llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_top0.95_sz30000_20260508_120655_google-gemma-3-4b-it_node18_v8"

# trying again - done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_top0.95_sz20000_20260508_122409_node2_qwen4_v9" \
  > nohup_outs/superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_top0.95_sz20000_20260508_122409_node2_qwen4_v9.out 2>&1 &

# trying again - done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
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
  --output-substring "superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_top0.95_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v9" \
  > nohup_outs/superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_top0.95_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v9.out 2>&1 &

# aunabil lpo 1 done => final done --- amlan m1 round3 lpo 1
python -u LPO/experiments/relation_extraction_lpo.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --optimizer-device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "superstart_lpo_roleswitch_node2_step1_n1000_qwen4_optqwen14"

# aunabil lpo 2 done ** again => final done --- amlan m1 round3 lpo 2
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --optimizer-device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "superstart_lpo_roleswitch_node10_step1_n1000_qwen4_optqwen14" \
  > nohup_outs/superstart_lpo_roleswitch_node10_step1_n1000_qwen4_optqwen14.out 2>&1 &

# tanvir lpo 1 done => final done --- amlan m1 round3 lpo 3
python -u LPO/experiments/relation_extraction_lpo.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --optimizer-model "google/gemma-3-12b-it" \
  --optimizer-device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "superstart_lpo_roleswitch_node3_step1_n1000_gemma4_optgemma12"


# tanvir lpo 2 done => final done --- amlan m2 round3 lpo 1
python -u LPO/experiments/relation_extraction_lpo.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --optimizer-model "google/gemma-3-12b-it" \
  --optimizer-device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 18 \
  --dataset-type "fs_tacred" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "superstart_lpo_roleswitch_node18_step1_n1000_gemma4_optgemma12"

# trying with r5 done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
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
  --output-substring "superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt3_top0.95_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v9" \
  > nohup_outs/superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt3_top0.95_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v9.out 2>&1 &

# trying again with mt1 done --- selected somehow
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz20000_20260508_122409_node2_qwen4_v9" \
  > nohup_outs/superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz20000_20260508_122409_node2_qwen4_v9.out 2>&1 &

## Evoprompt DE
# Qwen trainings/20260513_093517_Qwen-Qwen3-4B/population.json - node 14 and 42 : 14=	19.97,	26.04; 42 = 22.72,	32.35
# Gemma trainings/20260513_093554_google-gemma-3-4b-it/population.json - node 26 and 34: 17.47	20.34	; 34 =	18.13	22.72	

# done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260513_093517_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 14 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_evoprompt_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_sz20000_20260513_093517_Qwen_node14_qwen4_v9" \
  > nohup_outs/superstart_evoprompt_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_sz20000_20260513_093517_Qwen_node14_qwen4_v9.out 2>&1 &

# done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --prompt-source-path "../trainings/20260513_093517_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 42 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_evoprompt_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_sz20000_20260513_093517_Qwen_node42_qwen4_v9" \
  > nohup_outs/superstart_evoprompt_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt2_sz20000_20260513_093517_Qwen_node42_qwen4_v9.out 2>&1 &

# trying with r5 mt1 done - assuming selected
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
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
  --output-substring "superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v9" \
  > nohup_outs/superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz30000_20260508_120655_google-gemma-3-4b-it_node3_v9.out 2>&1 &

# done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_122409_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 10 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz20000_20260508_122409_node10_qwen4_v9" \
  > nohup_outs/superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz20000_20260508_122409_node10_qwen4_v9.out 2>&1 &

# done
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --prompt-source-path "../trainings/20260508_120655_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 18 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
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
  --output-substring "superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz30000_20260508_120655_google-gemma-3-4b-it_node18_v9" \
  > nohup_outs/superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz30000_20260508_120655_google-gemma-3-4b-it_node18_v9.out 2>&1 &

# done 
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --prompt-source-path "../trainings/20260513_093554_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 26 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
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
  --output-substring "superstart_evoprompt_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node26_v8" \
  > nohup_outs/superstart_evoprompt_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node26_v8.out 2>&1 & 

# done
python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260513_093554_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 34 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
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
  --output-substring "superstart_evoprompt_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node34_v8" \
  > nohup_outs/superstart_evoprompt_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_20260508_120655_google-gemma-3-4b-it_node34_v8.out 2>&1 & 

# aunabil evo 1 done
python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260513_093517_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 14 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz20000_20260513_093517_node14_qwen4_v9"

# aunabil evo 2 done
python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260513_093517_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 42 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz20000_20260513_093517_node42_qwen4_v9"

# tanvir evo 1 done
python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260513_093554_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 26 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
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
  --output-substring "superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz30000_20260513_093554_google-gemma-3-4b-it_node26_v9"

# tanvir evo 2 done
python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:0" \
  --prompt-source-path "../trainings/20260513_093554_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 34 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
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
  --output-substring "superstart_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz30000_20260513_093554_google-gemma-3-4b-it_node34_v9"

# done => final done --- amlan m2 round3 lpo 2
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:1" \
  --optimizer-model "google/gemma-3-12b-it" \
  --optimizer-device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260513_093554_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 26 \
  --dataset-type "fs_tacred" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "superstart_lpo_roleswitch_node26_step1_n1000_gemma4_optgemma12_20260513_093554" \
  > nohup_outs/superstart_lpo_roleswitch_node26_step1_n1000_gemma4_optgemma12_20260513_093554.out 2>&1 & 

# done => final done --- amlan m2 round3 lpo 3
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:1" \
  --optimizer-model "google/gemma-3-12b-it" \
  --optimizer-device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260513_093554_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 34 \
  --dataset-type "fs_tacred" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "superstart_lpo_roleswitch_node34_step1_n1000_gemma4_optgemma12_20260513_093554" \
  > nohup_outs/superstart_lpo_roleswitch_node34_step1_n1000_gemma4_optgemma12_20260513_093554.out 2>&1 & 

# done => final done --- amlan m3 round3 lpo 1
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --optimizer-device-map "cuda:3" \
  --prompt-source population \
  --population-path "../trainings/20260513_093517_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 14 \
  --dataset-type "fs_tacred" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "superstart_lpo_roleswitch_node14_step1_n1000_qwen4_optqwen14_20260513_093517" \
  > nohup_outs/superstart_lpo_roleswitch_node14_step1_n1000_qwen4_optqwen14_20260513_093517.out 2>&1 & 

# done => final done --- amlan m3 round3 lpo 2
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --optimizer-device-map "cuda:3" \
  --prompt-source population \
  --population-path "../trainings/20260513_093517_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 42 \
  --dataset-type "fs_tacred" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "superstart_lpo_roleswitch_node42_step1_n1000_qwen4_optqwen14_20260513_093517" \
  > nohup_outs/superstart_lpo_roleswitch_node42_step1_n1000_qwen4_optqwen14_20260513_093517.out 2>&1 & 

# aunabil_m1 evo_greater 1 done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260513_093517_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 14 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
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
  --output-substring "superstart_greater_seq_node14_1steps_lmd0.2_qwen4_20260513_093517" \
  > nohup_outs/superstart_greater_seq_node14_1steps_lmd0.2_qwen4_20260513_093517.out 2>&1 &

# aunabil_m1 evo_greater 2 done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260513_093554_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 26 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
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
  --output-substring "superstart_greater_seq_node26_1steps_lmd0.2_gemma4_20260513_093554" \
  > nohup_outs/superstart_greater_seq_node26_1steps_lmd0.2_gemma4_20260513_093554.out 2>&1 &

# aunabil_m2 evo_greater 1 done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260513_093517_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 42 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
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
  --output-substring "superstart_greater_seq_node42_1steps_lmd0.2_qwen4_20260513_093517" \
  > nohup_outs/superstart_greater_seq_node42_1steps_lmd0.2_qwen4_20260513_093517.out 2>&1 &

# aunabil_m2 evo_greater 2 done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260513_093554_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 34 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
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
  --output-substring "superstart_greater_seq_node34_1steps_lmd0.2_gemma4_20260513_093554" \
  > nohup_outs/superstart_greater_seq_node34_1steps_lmd0.2_gemma4_20260513_093554.out 2>&1 &

# done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --prompt-source population \
  --population-path "../trainings/20260513_093517_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 14 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
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
  --output-substring "superstart_greater_topgrad_node14_1steps_lmd0.2_qwen4_20260513_093517" \
  > nohup_outs/superstart_greater_topgrad_node14_1steps_lmd0.2_qwen4_20260513_093517.out 2>&1 &

# done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:3" \
  --prompt-source population \
  --population-path "../trainings/20260513_093554_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 26 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
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
  --output-substring "superstart_greater_topgrad_node26_1steps_lmd0.2_gemma4_20260513_093554" \
  > nohup_outs/superstart_greater_topgrad_node26_1steps_lmd0.2_gemma4_20260513_093554.out 2>&1 &

# done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260513_093517_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 42 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
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
  --output-substring "superstart_greater_topgrad_node42_1steps_lmd0.2_qwen4_20260513_093517" \
  > nohup_outs/superstart_greater_topgrad_node42_1steps_lmd0.2_qwen4_20260513_093517.out 2>&1 &


# done
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260513_093554_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 34 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
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
  --output-substring "superstart_greater_topgrad_node34_1steps_lmd0.2_gemma4_20260513_093554" \
  > nohup_outs/superstart_greater_topgrad_node34_1steps_lmd0.2_gemma4_20260513_093554.out 2>&1 &

#done ==> etgpo tacred rerun: # done amlan m2 round5 etgpo1
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../unified_optimization_results/unified_opt_fs_tacred_20260513_230331/generated_prompt_candidates.json" \
  --prompt-node-id 2 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
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
  --output-substring "superstart_etgpo_greater_topgrad_node2_1steps_lmd0.2_qwen4_unified_opt_fs_tacred_20260513_230331" \
  > nohup_outs/superstart_etgpo_greater_topgrad_node2_1steps_lmd0.2_qwen4_unified_opt_fs_tacred_20260513_230331.out 2>&1 &

#done ==> etgpo tacred rerun: # done amlan m2 round5 etgpo2
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../unified_optimization_results/unified_opt_fs_tacred_20260513_230331/generated_prompt_candidates.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
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
  --output-substring "superstart_etgpo_greater_node2_1steps_lmd0.2_qwen4_unified_opt_fs_tacred_20260513_230331" \
  > nohup_outs/superstart_etgpo_greater_node2_1steps_lmd0.2_qwen4_unified_opt_fs_tacred_20260513_230331.out 2>&1 &

# done ==> etgpo tacred rerun: # done amlan m2 round5 etgpo3
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:3" \
  --prompt-source population \
  --population-path "../unified_optimization_results/unified_opt_fs_tacred_20260514_025450/generated_prompt_candidates.json" \
  --prompt-node-id 0 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
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
  --output-substring "superstart_etgpo_greater_topgrad_node1_1steps_lmd0.2_gemma4_unified_opt_fs_tacred_20260514_025450" \
  > nohup_outs/superstart_etgpo_greater_topgrad_node1_1steps_lmd0.2_gemma4_unified_opt_fs_tacred_20260514_025450.out 2>&1 &

# done ==> etgpo tacred rerun: # done amlan m3 round5 etgpo1
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:3" \
  --prompt-source population \
  --population-path "../unified_optimization_results/unified_opt_fs_tacred_20260514_025450/generated_prompt_candidates.json" \
  --prompt-node-id 0 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
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
  --output-substring "superstart_etgpo_greater_node1_1steps_lmd0.2_gemma4_unified_opt_fs_tacred_20260514_025450" \
  > nohup_outs/superstart_etgpo_greater_node1_1steps_lmd0.2_gemma4_unified_opt_fs_tacred_20260514_025450.out 2>&1 &

# next aunabil m1 etgpo lpo 1 done => final done --- amlan m3 round3 lpo 3 ==> etgpo tacred rerun: # done amlan m3 round5 etgpo2
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --optimizer-device-map "cuda:0" \
  --prompt-source population \
  --population-path "../unified_optimization_results/unified_opt_fs_tacred_20260513_230331/generated_prompt_candidates.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "superstart_lpo_roleswitch_node2_step1_n1000_qwen4_optqwen14_unified_opt_fs_tacred_20260513_230331" \
  > nohup_outs/superstart_lpo_roleswitch_node2_step1_n1000_qwen4_optqwen14_unified_opt_fs_tacred_20260513_230331.out 2>&1 & 

# next aunabil m2 etgpo lpo 1 done => final done --- local only  ==> etgpo tacred rerun: # done amlan m3 round5 etgpo3
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --optimizer-model "google/gemma-3-12b-it" \
  --optimizer-device-map "cuda:0" \
  --prompt-source population \
  --population-path "../unified_optimization_results/unified_opt_fs_tacred_20260514_025450/generated_prompt_candidates.json" \
  --prompt-node-id 0 \
  --dataset-type "fs_tacred" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "superstart_etgpo_lpo_roleswitch_node1_step1_n1000_gemma4_optgemma12_unified_opt_fs_tacred_20260514_025450_ultra" \
  > nohup_outs/superstart_etgpo_lpo_roleswitch_node1_step1_n1000_gemma4_optgemma12_unified_opt_fs_tacred_20260514_025450_ultra.out 2>&1 & 

# done ==> etgpo tacred rerun: # done local
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --population-path "../unified_optimization_results/unified_opt_fs_tacred_20260513_230331/generated_prompt_candidates.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_etgpo_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz20000_unified_opt_fs_tacred_20260513_230331_node2_qwen4_v9_ultra" \
  > nohup_outs/superstart_etgpo_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz20000_unified_opt_fs_tacred_20260513_230331_node2_qwen4_v9_ultra.out 2>&1 & 

# done ==> etgpo tacred rerun: # done local
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:2" \
  --population-path "../unified_optimization_results/unified_opt_fs_tacred_20260513_230331/generated_prompt_candidates.json" \
  --prompt-node-id 2 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_tacred_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "superstart_etgpo_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz20000_unified_opt_fs_tacred_20260513_230331_node2_qwen4_v9_ultra" \
  > nohup_outs/superstart_etgpo_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz20000_unified_opt_fs_tacred_20260513_230331_node2_qwen4_v9_ultra.out 2>&1 & 


# next aunabil m1 gem 1 done ==> etgpo tacred rerun: done local
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --population-path "../unified_optimization_results/unified_opt_fs_tacred_20260514_025450/generated_prompt_candidates.json" \
  --prompt-node-id 0 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
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
  --output-substring "superstart_etgpo_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz30000_unified_opt_fs_tacred_20260514_025450_google-gemma-3-4b-it_node1_v9_ultra" \
  > nohup_outs/superstart_etgpo_llm_prob_beam5_C5_Q1_stdp2.0_px_0.5_log_r5_er_0.6_mt1_top0.95_sz30000_unified_opt_fs_tacred_20260514_025450_google-gemma-3-4b-it_node1_v9_ultra.out 2>&1 & 

# next aunabil m2 gem 1 done ==> etgpo tacred rerun: # done amlan m1 round4 etgpo1
python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --population-path "../unified_optimization_results/unified_opt_fs_tacred_20260514_025450/generated_prompt_candidates.json" \
  --prompt-node-id 0 \
  --dataset-type "fs_tacred" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
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
  --output-substring "superstart_etgpo_llm_cand_sugg_beam5_C5_Q1_stdp2.0_px_0.5_log_r3_er_0.6_mt3_sz30000_unified_opt_fs_tacred_20260514_025450_google-gemma-3-4b-it_node1_v8"





### fs_fewrel 2nd stage runs 

#RPO
Qwen: trainings/20260517_021906_Qwen-Qwen3-4B/population.json node:3	26.99	, 25.81 ... node: 13	32.06	30.75
Gemma: trainings/20260517_022119_google-gemma-3-4b-it/population.json node:	1	17.61	15.86  node: 7	20.79	19.55 

#EvoPrompt-DE
Qwen: trainings/20260517_193615_Qwen-Qwen3-4B/population.json node: 25	25.94	24.44, (one only)
Gemma: trainings/20260517_193624_google-gemma-3-4b-it/population.json node: 7	18.04	14.39, node: 14	19.71	18.22

#ETGPO
Qwen: unified_optimization_results/unified_opt_fs_fewrel_20260517_023848/generated_prompt_candidates.json   node 0: 26.76	26.21
Gemma: unified_optimization_results/unified_opt_fs_fewrel_20260517_023518/generated_prompt_candidates.json 	node 4: 23.08	17.65

### LPO Qwen
# done local => final done --- local only
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --optimizer-device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_021906_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_fewrel" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --seed 7 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "fewrel_superstart_rpo_lpo_node3_20260517_021906_Qwen_ultra" \
  > nohup_outs/fewrel_superstart_rpo_lpo_node3_20260517_021906_Qwen_ultra.out 2>&1 & 

# done local => final done --- local only
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --optimizer-device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_021906_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 13 \
  --dataset-type "fs_fewrel" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "fewrel_superstart_rpo_lpo_node13_20260517_021906_Qwen" \
  > nohup_outs/fewrel_superstart_rpo_lpo_node13_20260517_021906_Qwen.out 2>&1 & 

# done local  => final done --- local only
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --optimizer-device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_193615_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 25 \
  --dataset-type "fs_fewrel" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "fewrel_superstart_evoprompt_lpo_node25_20260517_193615_Qwen" \
  > nohup_outs/fewrel_superstart_evoprompt_lpo_node25_20260517_193615_Qwen.out 2>&1 & 

# done local => final skpping --- local only ==> etgpo again run fewrel # done amlan m1 round4 lpo1
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:1" \
  --optimizer-model "Qwen/Qwen3-14B" \
  --optimizer-device-map "cuda:1" \
  --prompt-source population \
  --population-path "../unified_optimization_results/unified_opt_fs_fewrel_20260517_023848/generated_prompt_candidates.json" \
  --prompt-node-id 0 \
  --dataset-type "fs_fewrel" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "fewrel_superstart_etgpo_lpo_node0_unified_opt_fs_fewrel_20260517_023848_Qwen" \
  > nohup_outs/fewrel_superstart_etgpo_lpo_node0_unified_opt_fs_fewrel_20260517_023848_Qwen.out 2>&1 & 

### LPO Gemma
# done tanvir m1 round2 few1 => final done --- amlan m2 round4 lpo 1
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --optimizer-model "google/gemma-3-12b-it" \
  --optimizer-device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_022119_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_fewrel" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "fewrel_superstart_rpo_lpo_node1_gemma_20260517_022119" \
  > nohup_outs/fewrel_superstart_rpo_lpo_node1_gemma_20260517_022119.out 2>&1 & 

# done tanvir m1 round2 few2 => final done --- amlan m2 round4 lpo 2
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --optimizer-model "google/gemma-3-12b-it" \
  --optimizer-device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_022119_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_fewrel" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "fewrel_superstart_rpo_lpo_node7_gemma_20260517_022119" \
  > nohup_outs/fewrel_superstart_rpo_lpo_node7_gemma_20260517_022119.out 2>&1 & 

# done tanvir m1 round2 few3 => final done --- amlan m3 round4 lpo 1
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --optimizer-model "google/gemma-3-12b-it" \
  --optimizer-device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_193624_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_fewrel" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "fewrel_superstart_evoprompt_lpo_node7_gemma_20260517_193624" \
  > nohup_outs/fewrel_superstart_evoprompt_lpo_node7_gemma_20260517_193624.out 2>&1 & 

# done tanvir m1 round2 few4 => final done --- amlan m3 round4 lpo 2
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --optimizer-model "google/gemma-3-12b-it" \
  --optimizer-device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_193624_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 14 \
  --dataset-type "fs_fewrel" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "fewrel_superstart_evoprompt_lpo_node14_gemma_20260517_193624" \
  > nohup_outs/fewrel_superstart_evoprompt_lpo_node14_gemma_20260517_193624.out 2>&1 &

# done local => final skpping --- local only ==> etgpo again run fewrel # done amlan m1 round4 lpo2
nohup python -u LPO/experiments/relation_extraction_lpo.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --optimizer-model "google/gemma-3-12b-it" \
  --optimizer-device-map "cuda:0" \
  --prompt-source population \
  --population-path "../unified_optimization_results/unified_opt_fs_fewrel_20260517_023518/generated_prompt_candidates.json" \
  --prompt-node-id 4 \
  --dataset-type "fs_fewrel" \
  --train-feedback-sample-size 1000 \
  --feedback-example-size 3 \
  --mistake-example-size 3 \
  --n-steps 1 \
  --max-edit-tags 5 \
  --max-words-per-edit-tag 3 \
  --num-candidates 5 \
  --top-z 5 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../lpo_experiments" \
  --output-substring "fewrel_superstart_etgpo_lpo_node4_gemma_unified_opt_fs_fewrel_20260517_023518" \
  > nohup_outs/fewrel_superstart_etgpo_lpo_node4_gemma_unified_opt_fs_fewrel_20260517_023518.out 2>&1 & 

# GreaTer Qwen
# done local
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --prompt-source population \
  --population-path "../trainings/20260517_021906_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_rpo_greater_node3_20260517_021906_Qwen" \
  > nohup_outs/fewrel_superstart_rpo_greater_node3_20260517_021906_Qwen.out 2>&1 & 

# done local
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:3" \
  --prompt-source population \
  --population-path "../trainings/20260517_021906_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 13 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_rpo_greater_node13_20260517_021906_Qwen" \
  > nohup_outs/fewrel_superstart_rpo_greater_node13_20260517_021906_Qwen.out 2>&1 & 

# done local
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260517_021906_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 3 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_rpo_greater_topgrad_node3_20260517_021906_Qwen" \
  > nohup_outs/fewrel_superstart_rpo_greater_topgrad_node3_20260517_021906_Qwen.out 2>&1 & 

# done local
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_021906_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 13 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_rpo_greater_topgrad_node13_20260517_021906_Qwen" \
  > nohup_outs/fewrel_superstart_rpo_greater_topgrad_node13_20260517_021906_Qwen.out 2>&1 & 

# done amlan m3 round2 few 1
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_193615_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 25 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_evoprompt_greater_node25_20260517_193615_Qwen" \
  > nohup_outs/fewrel_superstart_evoprompt_greater_node25_20260517_193615_Qwen.out 2>&1 & 

# done amlan m3 round2 few 2
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_193615_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 25 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_evoprompt_greater_topgrad_node25_20260517_193615_Qwen" \
  > nohup_outs/fewrel_superstart_evoprompt_greater_topgrad_node25_20260517_193615_Qwen.out 2>&1 & 

# done local
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:1" \
  --prompt-source population \
  --population-path "../unified_optimization_results/unified_opt_fs_fewrel_20260517_023848/generated_prompt_candidates.json" \
  --prompt-node-id 0 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_etgpo_greater_node0_unified_opt_fs_fewrel_20260517_023848_Qwen" \
  > nohup_outs/fewrel_superstart_etgpo_greater_node0_unified_opt_fs_fewrel_20260517_023848_Qwen.out 2>&1 & 


# done amlan m3 round2 few 3
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "Qwen/Qwen3-4B" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../unified_optimization_results/unified_opt_fs_fewrel_20260517_023848/generated_prompt_candidates.json" \
  --prompt-node-id 0 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_etgpo_greater_topgrad_node0_unified_opt_fs_fewrel_20260517_023848_Qwen" \
  > nohup_outs/fewrel_superstart_etgpo_greater_topgrad_node0_unified_opt_fs_fewrel_20260517_023848_Qwen.out 2>&1 & 

# GreaTer Gemma
# done amlan m3 round2 few 4
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_022119_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_rpo_greater_node1_gemma_20260517_022119" \
  > nohup_outs/fewrel_superstart_rpo_greater_node1_gemma_20260517_022119.out 2>&1 & 

# done local
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260517_022119_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_rpo_greater_node7_gemma_20260517_022119" \
  > nohup_outs/fewrel_superstart_rpo_greater_node7_gemma_20260517_022119.out 2>&1 & 

# done local
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_022119_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 1 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_rpo_greater_topgrad_node1_gemma_20260517_022119" \
  > nohup_outs/fewrel_superstart_rpo_greater_topgrad_node1_gemma_20260517_022119.out 2>&1 & 

# done amlan m2 round2 few3
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_022119_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 7 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_rpo_greater_topgrad_node7_gemma_20260517_022119" \
  > nohup_outs/fewrel_superstart_rpo_greater_topgrad_node7_gemma_20260517_022119.out 2>&1 & 

# done amlan m2 round2 few4
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_193624_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_evoprompt_greater_node7_gemma_20260517_193624" \
  > nohup_outs/fewrel_superstart_evoprompt_greater_node7_gemma_20260517_193624.out 2>&1 & 

# done amlan m1 round2 few3
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_193624_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 14 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_evoprompt_greater_node14_gemma_20260517_193624" \
  > nohup_outs/fewrel_superstart_evoprompt_greater_node14_gemma_20260517_193624.out 2>&1 & 

# done amlan m1 round2 few4
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../trainings/20260517_193624_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 7 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_evoprompt_greater_topgrad_node7_gemma_20260517_193624" \
  > nohup_outs/fewrel_superstart_evoprompt_greater_topgrad_node7_gemma_20260517_193624.out 2>&1 & 

# done local
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:1" \
  --prompt-source population \
  --population-path "../trainings/20260517_193624_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 14 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_evoprompt_greater_topgrad_node14_gemma_20260517_193624" \
  > nohup_outs/fewrel_superstart_evoprompt_greater_topgrad_node14_gemma_20260517_193624.out 2>&1 & 


# done local
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:3" \
  --prompt-source population \
  --population-path "../unified_optimization_results/unified_opt_fs_fewrel_20260517_023518/generated_prompt_candidates.json" \
  --prompt-node-id 4 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_etgpo_greater_node4_gemma_unified_opt_fs_fewrel_20260517_023518" \
  > nohup_outs/fewrel_superstart_etgpo_greater_node4_gemma_unified_opt_fs_fewrel_20260517_023518.out 2>&1 & 

# done amlan m3 round2 few5
nohup python -u GreaTer/experiments/relation_extraction_greater.py \
  --model "google/gemma-3-4b-it" \
  --device-map "cuda:0" \
  --prompt-source population \
  --population-path "../unified_optimization_results/unified_opt_fs_fewrel_20260517_023518/generated_prompt_candidates.json" \
  --prompt-node-id 4 \
  --position-selection-mode top_gradient \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
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
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../greater_experiments" \
  --output-substring "fewrel_superstart_etgpo_greater_topgrad_node4_gemma_unified_opt_fs_fewrel_20260517_023518" \
  > nohup_outs/fewrel_superstart_etgpo_greater_topgrad_node4_gemma_unified_opt_fs_fewrel_20260517_023518.out 2>&1 & 


## GradPO Qwen
#done local
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --population-path "../trainings/20260517_021906_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_rpo_gradpo_node3_20260517_021906_Qwen" \
  > nohup_outs/fewrel_superstart_rpo_gradpo_node3_20260517_021906_Qwen.out 2>&1 & 

#done local
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --population-path "../trainings/20260517_021906_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 13 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_rpo_gradpo_node13_20260517_021906_Qwen" \
  > nohup_outs/fewrel_superstart_rpo_gradpo_node13_20260517_021906_Qwen.out 2>&1 & 

# done local
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --population-path "../trainings/20260517_021906_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 3 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_rpo_gradpo_prob_node3_20260517_021906_Qwen" \
  > nohup_outs/fewrel_superstart_rpo_gradpo_prob_node3_20260517_021906_Qwen.out 2>&1 & 

# done local
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:1" \
  --population-path "../trainings/20260517_021906_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 13 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_rpo_gradpo_prob_node13_20260517_021906_Qwen" \
  > nohup_outs/fewrel_superstart_rpo_gradpo_prob_node13_20260517_021906_Qwen.out 2>&1 & 

# done local
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:3" \
  --population-path "../trainings/20260517_193615_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 25 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_evoprompt_gradpo_node25_20260517_193615_Qwen" \
  > nohup_outs/fewrel_superstart_evoprompt_gradpo_node25_20260517_193615_Qwen.out 2>&1 & 

# done local
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:3" \
  --population-path "../trainings/20260517_193615_Qwen-Qwen3-4B/population.json" \
  --prompt-node-id 25 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_evoprompt_gradpo_prob_node25_20260517_193615_Qwen" \
  > nohup_outs/fewrel_superstart_evoprompt_gradpo_prob_node25_20260517_193615_Qwen.out 2>&1 & 

# done tanvir m1 few1 --- run again => # done amlan m2 round2 few1
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --population-path "../unified_optimization_results/unified_opt_fs_fewrel_20260517_023848/generated_prompt_candidates.json" \
  --prompt-node-id 0 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 2 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_etgpo_gradpo_node0_unified_opt_fs_fewrel_20260517_023848_Qwen" \
  > nohup_outs/fewrel_superstart_etgpo_gradpo_node0_unified_opt_fs_fewrel_20260517_023848_Qwen.out 2>&1 & 

# done tanvir m1 few2 --- run again => # done amlan m2 round2 few2
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "Qwen/Qwen3-4B" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:0" \
  --population-path "../unified_optimization_results/unified_opt_fs_fewrel_20260517_023848/generated_prompt_candidates.json" \
  --prompt-node-id 0 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 20000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --full-eval-split final_step_dev \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_etgpo_gradpo_prob_node0_unified_opt_fs_fewrel_20260517_023848_Qwen" \
  > nohup_outs/fewrel_superstart_etgpo_gradpo_prob_node0_unified_opt_fs_fewrel_20260517_023848_Qwen.out 2>&1 & 


## GradPO gemma
# done tanvir m2 few1
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --population-path "../trainings/20260517_022119_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_rpo_gradpo_node1_gemma_20260517_022119" \
  > nohup_outs/fewrel_superstart_rpo_gradpo_node1_gemma_20260517_022119.out 2>&1 & 

# done tanvir m2 few2
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --population-path "../trainings/20260517_022119_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_rpo_gradpo_node7_gemma_20260517_022119" \
  > nohup_outs/fewrel_superstart_rpo_gradpo_node7_gemma_20260517_022119.out 2>&1 & 

# done amlan m1 few1
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:0" \
  --population-path "../trainings/20260517_022119_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 1 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_rpo_gradpo_prob_node1_gemma_20260517_022119" \
  > nohup_outs/fewrel_superstart_rpo_gradpo_prob_node1_gemma_20260517_022119.out 2>&1 & 

# done amlan m1 few2
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:0" \
  --population-path "../trainings/20260517_022119_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_rpo_gradpo_prob_node7_gemma_20260517_022119" \
  > nohup_outs/fewrel_superstart_rpo_gradpo_prob_node7_gemma_20260517_022119.out 2>&1 & 

# done amlan m2 few1
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --population-path "../trainings/20260517_193624_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_evoprompt_gradpo_node7_gemma_20260517_193624" \
  > nohup_outs/fewrel_superstart_evoprompt_gradpo_node7_gemma_20260517_193624.out 2>&1 & 

# done amlan m2 few2
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --population-path "../trainings/20260517_193624_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 14 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_evoprompt_gradpo_node14_gemma_20260517_193624" \
  > nohup_outs/fewrel_superstart_evoprompt_gradpo_node14_gemma_20260517_193624.out 2>&1 &

# done amlan m3 few1
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:0" \
  --population-path "../trainings/20260517_193624_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 7 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_evoprompt_gradpo_prob_node7_gemma_20260517_193624" \
  > nohup_outs/fewrel_superstart_evoprompt_gradpo_prob_node7_gemma_20260517_193624.out 2>&1 & 

# done amlan m3 few2
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:0" \
  --population-path "../trainings/20260517_193624_google-gemma-3-4b-it/population.json" \
  --prompt-node-id 14 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_evoprompt_gradpo_prob_node14_gemma_20260517_193624" \
  > nohup_outs/fewrel_superstart_evoprompt_gradpo_prob_node14_gemma_20260517_193624.out 2>&1 &

# done amlan m1 round2 few1 
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LLM_CANDIDATE_SUGGESTION" \
  --device-map "cuda:0" \
  --population-path "../unified_optimization_results/unified_opt_fs_fewrel_20260517_023518/generated_prompt_candidates.json" \
  --prompt-node-id 4 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 3 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 3 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_etgpo_gradpo_node4_gemma_unified_opt_fs_fewrel_20260517_023518" \
  > nohup_outs/fewrel_superstart_etgpo_gradpo_node4_gemma_unified_opt_fs_fewrel_20260517_023518.out 2>&1 & 

# done amlan m1 round2 few2 
nohup python -u agents/agent_gradient_eval_debug.py \
  --model "google/gemma-3-4b-it" \
  --mode "LM_PROBABILITY_CANDIDATE_SUGGESTION" \
  --lm-probability-submode "FULL_PROMPT_AS_CONTEXT" \
  --device-map "cuda:0" \
  --population-path "../unified_optimization_results/unified_opt_fs_fewrel_20260517_023518/generated_prompt_candidates.json" \
  --prompt-node-id 4 \
  --dataset-type "fs_fewrel" \
  --train-gradient-sample-size 30000 \
  --gradient-batch-size 2 \
  --max-regions 10 \
  --max-total-region-tokens 50 \
  --max-region-tokens 1 \
  --region-expansion-threshold-ratio 0.6 \
  --num-edit-regions 5 \
  --num-region-candidates 5 \
  --beam-width 5 \
  --selection-perplexity-lambda 0.5 \
  --use-log-fluency-score \
  --meta-prompt-max-new-tokens 10000 \
  --meta-prompt-batch-size 1 \
  --validation-batch-size 8 \
  --Q 1 \
  --selection-f1-std-penalty 2.0 \
  --train-samples "fs_fewrel_train_non_split_original_samples.pkl" \
  --full-eval-split final_step_dev \
  --output-root-dir "../gradients_experiments" \
  --output-substring "fewrel_superstart_etgpo_gradpo_prob_node4_gemma_unified_opt_fs_fewrel_20260517_023518" \
  > nohup_outs/fewrel_superstart_etgpo_gradpo_prob_node4_gemma_unified_opt_fs_fewrel_20260517_023518.out 2>&1 & 