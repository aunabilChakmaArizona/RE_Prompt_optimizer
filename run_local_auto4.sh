python -u codes/final_model_run_icl_dynamic_prompt.py --model Qwen/Qwen3-4B --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:3 --ep_start 0 --ep_end 150000 --batch_size 8 --data_root data --output_dir outputs/opt_prompt --code fewrel_qwen_initial_prompt --prompt 'You are given a relation name, a description of the relation in brackets, a support sentence exemplifying the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. 
You need to decide whether the relation holds between the Subject and the Object entities in the query sentence.

If the relation holds between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'
python -u codes/final_model_run_icl_dynamic_prompt.py --model google/gemma-3-4b-it --dataset fs_fewrel_test_episodes_1shots.pkl --dataset_core fewrel --ways 5 --shots 1 --query 0 --cuda cuda:3 --ep_start 0 --ep_end 150000 --batch_size 8 --data_root data --output_dir outputs/opt_prompt --code fewrel_gemma_initial_prompt --prompt 'You are given a relation name, a description of the relation in brackets, a support sentence exemplifying the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with the subject and object tags, respectively. 
You need to decide whether the relation holds between the Subject and the Object entities in the query sentence.

If the relation holds between the Subject and the Object entities in the query sentence, answer "yes"; otherwise, answer "no".'