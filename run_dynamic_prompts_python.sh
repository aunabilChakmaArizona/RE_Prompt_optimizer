#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/logs outputs/opt_prompt

# Generated from all_the_prompts.txt
# Each CODE block is passed as the instruction split; answer/input splits come from agents.agent_prompts.
# Plain python commands. Run this script to execute commands sequentially.
# No shell log redirection is added in this script.
# All commands use cuda:0.
# Prompt blocks: 148
# Empty-status non-empty prompt blocks: 8
# Empty-status empty prompt blocks skipped: 34
# Status running/done blocks skipped: 106
# Other non-empty status blocks skipped: 0
# Exact duplicate non-empty blocks skipped: 0

python -u codes/final_model_run_icl_dynamic_prompt.py --model Qwen/Qwen3-4B --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_qwen_greater_longer --prompt 'Based are in a single token in a description from the relation in brackets, a support sentence exemplifying the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. 
You need to decide whether the relation holds between the Subject and the Object entities in the query sentence.

If the relation holds between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_gemma_greater_longer --prompt 'The classifier given a relation name, support description of the relation in brackets, a support sentence exemplifying the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. 
You need to decide whether the relation holds between the Subject and the Object entities in the query sentence.

If the relation holds between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_tacred_test_episodes_1shots.pkl --dataset_core tacred --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code tacred_gemma_greater_longer --prompt 'Rel are given the relation name, support description of the relation in brackets, a support sentence exemplifying the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. 
You need to decide whether the relation holds between the Subject and the Object entities in the query sentence.

If the relation holds between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model Qwen/Qwen3-4B --dataset fs_tacred_test_episodes_1shots.pkl --dataset_core tacred --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code tacred_qwen_greater_longer --prompt 'You classify each a relation type as follows description ( the relation in brackets, a support sentence exemplifying the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. 
You need to decide whether the relation holds between the Subject and the Object entities in the query sentence.

If the relation holds between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model Qwen/Qwen3-4B --dataset fs_tacred_test_episodes_1shots.pkl --dataset_core tacred --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code tacred_qwen_gradpo-gen_longer --prompt 'You are given a relation name, a description of the relation in curly braces, a support illustrative sentence exemplifying the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. 
You need to decide whether the relation holds between the Subject and the Object entities in the query sentence.

If the relation holds between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model Qwen/Qwen3-4B --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_qwen_gradpo-gen_longer --prompt 'The agent is supplied a relation name, a description of the relation in set notation format, a support example of the relation, and a query inquiry. A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. You need to decide whether the relation is applicable to the Subject and the Object entities in the query inquiry. If the relation is applicable to the Subject and the Object entities in the query inquiry, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_tacred_test_episodes_1shots.pkl --dataset_core tacred --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code tacred_gemma_gradpo-gen_longer --prompt 'You are given a relation property, a description of the relation in brackets, a sentence exemplifying the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. 
You need to decide whether the relation found is present in between the Subject and the Object entities in the query sentence.

If the relation is present in between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:0 --ep_start 0 --ep_end 150000 --batch_size 15 --data_root data --output_dir outputs/opt_prompt --code fewrel_gemma_gradpo-gen_longer --prompt 'You are given a relation name, a description of the relation in brackets, a sentence exemplifying the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. 
You need to decide whether the relation is present in between the Subject and the Object entities in the query sentence.

If the relation is present in between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'
