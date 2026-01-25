##### Arg parse and time

import argparse

parser = argparse.ArgumentParser(description="Read named arguments from the command line.")

parser.add_argument('--model', type=str, required=True, help='Model')
parser.add_argument('--dataset', type=str, required=True, help='Your age')
parser.add_argument('--dataset_core', type=str, required=True, help='Your age')
parser.add_argument('--ways', type=int, required=True, help='Your city')
parser.add_argument('--shots', type=int, required=True, help='Your city')
parser.add_argument('--query', type=int, required=True, help='Your city')
parser.add_argument('--cuda', type=str, required=True, help='Your city')
parser.add_argument('--batch_size', type=int, required=True, help='Your city')
parser.add_argument('--ep_start', type=int, required=True, help='Your city')
parser.add_argument('--ep_end', type=int, required=True, help='Your city')
parser.add_argument('--is_summ', action='store_true', help='Your city')
parser.add_argument('--is_rtx', type=int, required=False, default=0)

args = parser.parse_args()


from datetime import datetime
today = datetime.today()
formatted_date = today.strftime("%Y%m%d")  # %Y = 4-digit year, %m = 2-digit month, %d = 2-digit day

print(f"*********** Running Experiment on: {formatted_date}, Model:{args.model}, Data: {args.dataset}, Shots: {args.shots}, Query: {args.query}, CUDA: {args.cuda} ********, IS_SUMM: {args.is_summ}")

####### imports

import logging
import os
import sys
import time

# import openai
import torch
import torch.nn as nn
import transformers
from huggingface_hub import login, notebook_login
from openai import OpenAI
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.utils.data import Dataset, DataLoader
from functools import partial

# import scorer
# import util
import prompts

transformers.logging.set_verbosity_error()

####### Logging

access_token = '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'
login(token=access_token)

# model_id = "Qwen/Qwen3-4B"
# model_id = "microsoft/Phi-4-mini-instruct"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
model_id = args.model

DEVICE_MAP = args.cuda

print(model_id)
print(f"Device: {DEVICE_MAP}")

# quant_config = BitsAndBytesConfig(
#     load_in_8bit=True,       # enables 8-bit weights
#     llm_int8_threshold=6.0,  # default threshold for outlier handling
#     llm_int8_has_fp16_weight=False
# )
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    # quantization_config=quant_config,
    device_map=DEVICE_MAP,
    # attn_implementation= ("eager" if DEVICE_MAP != "cuda:0" else "flash_attention_2")
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.padding_side = 'left'

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]
no_token_id = tokenizer.encode("no", add_special_tokens=False)[0]
print(f"Yes token ID: {yes_token_id}")
print(f"No token ID: {no_token_id}")


PROMPT_TEMPLATE = dict()
PROMPT_TEMPLATE['1_shots_per_relation_w_des'] = '''You are given below a Relation name, a Description of the relation in brackets, a Support sentence (an example sentence) that holds the given relation between the Object and the Subject, and a Query sentence.

A relation connects the Subject and the Object. The Subject and the Object are given within the subject and object tags respectively. You need to decide whether the relation between the Subject and the Object of the Query sentence holds the given relation or not.

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)
Support Sentence: #SUPPORT_SENTENCE#

Query Sentence: #QUERY_SENTENCE#

If the relation between the subject and the object in Query sentence matches the given Relation name given say yes, otherwise no.
Just output "yes" or "no", and nothing else
'''

PROMPT_TEMPLATE['5_shots_per_relation_w_des'] = '''You are given below a Relation name, a Description of the relation in brackets, 5 Support sentences (example sentences) that holds the given relation between the Object and the Subject, and a Query sentence.

A relation connects the Subject and the Object. The Subject and the Object are given within the subject and object tags respectively. You need to decide whether the relation between the Subject and the Object of the Query sentence holds the given relation or not.

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)
Support Sentence 1: #SUPPORT_SENTENCE_1#
Support Sentence 2: #SUPPORT_SENTENCE_2#
Support Sentence 3: #SUPPORT_SENTENCE_3#
Support Sentence 4: #SUPPORT_SENTENCE_4#
Support Sentence 5: #SUPPORT_SENTENCE_5#

Query Sentence: #QUERY_SENTENCE#

If the relation between the subject and the object in Query sentence matches the given Relation name given say yes, otherwise no.
Just output "yes" or "no", and nothing else
'''


PROMPT_TEMPLATE['5_shots_per_relation_w_des_summ'] = '''You are given below a Relation name, a Description of the relation in brackets, Summerized version of 5 Support sentences (example sentences) that holds the given relation between the Object and the Subject, and a Query sentence.

A relation connects the Subject and the Object. The Subject and the Object are given within the subject and object tags respectively. Note that you are given summerized version of the support sentence. You need to decide whether the relation between the Subject and the Object of the Query sentence holds the given relation or not.

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)
Support Sentence 1: #SUPPORT_SENTENCE_1#
Support Sentence 2: #SUPPORT_SENTENCE_2#
Support Sentence 3: #SUPPORT_SENTENCE_3#
Support Sentence 4: #SUPPORT_SENTENCE_4#
Support Sentence 5: #SUPPORT_SENTENCE_5#

Query Sentence: #QUERY_SENTENCE#

If the relation between the subject and the object in Query sentence matches the given Relation name given say yes, otherwise no.
Just output "yes" or "no", and nothing else
'''

PROMPT_TEMPLATE['10_shots_per_relation_w_des'] = '''You are given below a Relation name, a Description of the relation in brackets, 10 Support sentences (example sentences) that holds the given relation between the Object and the Subject, and a Query sentence.

A relation connects the Subject and the Object. The Subject and the Object are given within the subject and object tags respectively. You need to decide whether the relation between the Subject and the Object of the Query sentence holds the given relation or not.

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)
Support Sentence 1: #SUPPORT_SENTENCE_1#
Support Sentence 2: #SUPPORT_SENTENCE_2#
Support Sentence 3: #SUPPORT_SENTENCE_3#
Support Sentence 4: #SUPPORT_SENTENCE_4#
Support Sentence 5: #SUPPORT_SENTENCE_5#
Support Sentence 6: #SUPPORT_SENTENCE_6#
Support Sentence 7: #SUPPORT_SENTENCE_7#
Support Sentence 8: #SUPPORT_SENTENCE_8#
Support Sentence 9: #SUPPORT_SENTENCE_9#
Support Sentence 10: #SUPPORT_SENTENCE_10#

Query Sentence: #QUERY_SENTENCE#

If the relation between the subject and the object in Query sentence matches the given Relation name given say yes, otherwise no.
Just output "yes" or "no", and nothing else
'''

PROMPT_TEMPLATE['10_shots_per_relation_w_des_summ'] = '''You are given below a Relation name, a Description of the relation in brackets, Summerized version of 10 Support sentences (example sentences) that holds the given relation between the Object and the Subject, and a Query sentence.

A relation connects the Subject and the Object. The Subject and the Object are given within the subject and object tags respectively. Note that you are given summerized version of the support sentence. You need to decide whether the relation between the Subject and the Object of the Query sentence holds the given relation or not.

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)
Support Sentence 1: #SUPPORT_SENTENCE_1#
Support Sentence 2: #SUPPORT_SENTENCE_2#
Support Sentence 3: #SUPPORT_SENTENCE_3#
Support Sentence 4: #SUPPORT_SENTENCE_4#
Support Sentence 5: #SUPPORT_SENTENCE_5#
Support Sentence 6: #SUPPORT_SENTENCE_6#
Support Sentence 7: #SUPPORT_SENTENCE_7#
Support Sentence 8: #SUPPORT_SENTENCE_8#
Support Sentence 9: #SUPPORT_SENTENCE_9#
Support Sentence 10: #SUPPORT_SENTENCE_10#

Query Sentence: #QUERY_SENTENCE#

If the relation between the subject and the object in Query sentence matches the given Relation name given say yes, otherwise no.
Just output "yes" or "no", and nothing else
'''

PROMPT_TEMPLATE['20_shots_per_relation_w_des'] = '''You are given below a Relation name, a Description of the relation in brackets, 20 Support sentences (example sentences) that holds the given relation between the Object and the Subject, and a Query sentence.

A relation connects the Subject and the Object. The Subject and the Object are given within the subject and object tags respectively. You need to decide whether the relation between the Subject and the Object of the Query sentence holds the given relation or not.

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)
Support Sentence 1: #SUPPORT_SENTENCE_1#
Support Sentence 2: #SUPPORT_SENTENCE_2#
Support Sentence 3: #SUPPORT_SENTENCE_3#
Support Sentence 4: #SUPPORT_SENTENCE_4#
Support Sentence 5: #SUPPORT_SENTENCE_5#
Support Sentence 6: #SUPPORT_SENTENCE_6#
Support Sentence 7: #SUPPORT_SENTENCE_7#
Support Sentence 8: #SUPPORT_SENTENCE_8#
Support Sentence 9: #SUPPORT_SENTENCE_9#
Support Sentence 10: #SUPPORT_SENTENCE_10#
Support Sentence 11: #SUPPORT_SENTENCE_11#
Support Sentence 12: #SUPPORT_SENTENCE_12#
Support Sentence 13: #SUPPORT_SENTENCE_13#
Support Sentence 14: #SUPPORT_SENTENCE_14#
Support Sentence 15: #SUPPORT_SENTENCE_15#
Support Sentence 16: #SUPPORT_SENTENCE_16#
Support Sentence 17: #SUPPORT_SENTENCE_17#
Support Sentence 18: #SUPPORT_SENTENCE_18#
Support Sentence 19: #SUPPORT_SENTENCE_19#
Support Sentence 20: #SUPPORT_SENTENCE_20#

Query Sentence: #QUERY_SENTENCE#

If the relation between the subject and the object in Query sentence matches the given Relation name given say yes, otherwise no.
Just output "yes" or "no", and nothing else
'''

PROMPT_TEMPLATE['20_shots_per_relation_w_des_summ'] = '''You are given below a Relation name, a Description of the relation in brackets, Summerized version of 20 Support sentences (example sentences) that holds the given relation between the Object and the Subject, and a Query sentence.

A relation connects the Subject and the Object. The Subject and the Object are given within the subject and object tags respectively. Note that you are given summerized version of the support sentence. You need to decide whether the relation between the Subject and the Object of the Query sentence holds the given relation or not.

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)
Support Sentence 1: #SUPPORT_SENTENCE_1#
Support Sentence 2: #SUPPORT_SENTENCE_2#
Support Sentence 3: #SUPPORT_SENTENCE_3#
Support Sentence 4: #SUPPORT_SENTENCE_4#
Support Sentence 5: #SUPPORT_SENTENCE_5#
Support Sentence 6: #SUPPORT_SENTENCE_6#
Support Sentence 7: #SUPPORT_SENTENCE_7#
Support Sentence 8: #SUPPORT_SENTENCE_8#
Support Sentence 9: #SUPPORT_SENTENCE_9#
Support Sentence 10: #SUPPORT_SENTENCE_10#
Support Sentence 11: #SUPPORT_SENTENCE_11#
Support Sentence 12: #SUPPORT_SENTENCE_12#
Support Sentence 13: #SUPPORT_SENTENCE_13#
Support Sentence 14: #SUPPORT_SENTENCE_14#
Support Sentence 15: #SUPPORT_SENTENCE_15#
Support Sentence 16: #SUPPORT_SENTENCE_16#
Support Sentence 17: #SUPPORT_SENTENCE_17#
Support Sentence 18: #SUPPORT_SENTENCE_18#
Support Sentence 19: #SUPPORT_SENTENCE_19#
Support Sentence 20: #SUPPORT_SENTENCE_20#

Query Sentence: #QUERY_SENTENCE#

If the relation between the subject and the object in Query sentence matches the given Relation name given say yes, otherwise no.
Just output "yes" or "no", and nothing else
'''


def get_sentence_with_tags(meta_data):
    token_with_tags = meta_data['token'].copy()

    token_with_tags[meta_data['subj_start']] = '<subject>' + token_with_tags[meta_data['subj_start']]
    token_with_tags[meta_data['subj_end']] = token_with_tags[meta_data['subj_end']] + '</subject>'
    token_with_tags[meta_data['obj_start']] = '<object>' + token_with_tags[meta_data['obj_start']]
    token_with_tags[meta_data['obj_end']] = token_with_tags[meta_data['obj_end']] + '</object>'

    return ' '.join(token_with_tags)

import json
import pickle

def read_json_file(file_path):
  with open(file_path, 'r') as f:
    data = json.load(f)
  return data

def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_dict(data: dict, filename: str) -> None:
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


prompt_type = f'{args.shots}_shots'
prompt_details = '_per_relation_w_des'
if args.is_summ == True:
    prompt_details += "_summ"

full_data = []
print()

file_ = args.dataset
NEW_ROOT_DATA_PATH = '/storage2/home/aunabilchakma/data/relation_extraction/'
file_path = os.path.join(NEW_ROOT_DATA_PATH, file_)
episodes = read_pickle_file(file_path)[args.ep_start:args.ep_end]
shots = read_pickle_file(NEW_ROOT_DATA_PATH + f"fs_{args.dataset_core}_shots_details.pkl")

print("Format: " + prompt_type + prompt_details)
print("Running: " + file_)

if args.is_summ == True:
    summ_map = read_pickle_file(NEW_ROOT_DATA_PATH + f"fs_{args.dataset_core}_v3_summ_map_{(model_id).replace('/','-')}.pkl")
else:
    summ_map = dict()

entity_map = read_pickle_file(NEW_ROOT_DATA_PATH + f"fs_{args.dataset_core}_entity_resolution_{(model_id).replace('/','-')}.pkl")
is_NYT = False

import hashlib
def sentence_to_id(sentence):
    normalized = sentence.lower().strip()
    hash_object = hashlib.sha256(normalized.encode('utf-8'))
    return hash_object.hexdigest()

time_start = time.time()
batch_size = int(args.batch_size)
prompts_ = []
answers = []
votes = []
pairs = []

print(f"Total episodes: {len(episodes)}")
for idx_episode, episode in enumerate(episodes):
    ways = episode['meta_train']
    query = episode['meta_test'][args.query]

    query = shots["queries"][query]
    qr = query['id']

    prompt = PROMPT_TEMPLATE[prompt_type + prompt_details]

    query_relation = query['relation']
    sentence = get_sentence_with_tags(query)
    prompt = prompt.replace('#QUERY_SENTENCE#', sentence)
    relation_list = []
    

    for idx_way, way in enumerate(ways):
        way_ = []
        way_.append(shots["shots"][way[0]])
        for w in way[1:]:
            way_.append(shots["umbc_shots"][w])
        way = way_


        idx_way = idx_way + 1

        relation = way[0]['relation']
        prompt_ = prompt.replace(f'#RELATION#', relation)
        prompt_ = prompt_.replace(f'#RELATION_DESCRIPTION#', prompts.get_relation_description(relation, dt= args.dataset_core))

        relation_list.append(relation)

        if '1_shots' not  in prompt_type:
            for idx_shot, shot in enumerate(way):
                idx_shot = idx_shot + 1

                sentence = get_sentence_with_tags(shot).strip()
                id256 = sentence_to_id(sentence) 

                if args.is_summ == True:
                    assert id256 in summ_map 
                    sentence = summ_map[id256]

                prompt_ = prompt_.replace(f'#SUPPORT_SENTENCE_{idx_shot}#', sentence)
        else:
            for idx_shot, shot in enumerate(way):
                idx_shot = idx_shot + 1

                assert shot['relation'] == relation

                sentence = get_sentence_with_tags(shot).strip()
                id256 = sentence_to_id(sentence) 
                
                if args.is_summ == True:
                    assert id256 in summ_map 
                    sentence = summ_map[id256]

                prompt_ = prompt_.replace(f'#SUPPORT_SENTENCE#', sentence)

                break

        messages = [{"role": "user", "content": prompt_}]
        prompts_.append(messages)

    #     print(prompt_)
    #     break
    # break

        if entity_map[qr][relation] == False:
            votes.append("no")
        else:
            votes.append("pending")
        pairs.append((query['id'], way[0]['id']))
        
    full_data.append(([], query_relation, relation_list))

assert len(votes) == len(prompts_)
assert len(votes) ==  args.ways * len(full_data)

to_do = []
for idx, (v, p, pa) in enumerate(zip(votes, prompts_, pairs)):
    if v == "pending":
        text = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tok_len = len(tokenizer(text, add_special_tokens=False).input_ids)
        to_do.append((idx, text, tok_len, pa[0], pa[1]))

to_do = sorted(to_do, key=lambda x: x[2], reverse=True)

checked_pa = dict()
unique_to_do = []
for entry in to_do:
    if (entry[-2], entry[-1]) not in checked_pa:
        unique_to_do.append(entry)
        checked_pa[(entry[-2], entry[-1])] = "pending"

batch_prompts = []
for i in range(0, len(unique_to_do), batch_size):
    batch_prompts.append(unique_to_do[i:i+batch_size])

print(f"Total votes {len(votes)}")
print(f"To prompt LLM: {len(unique_to_do)}")
print(f"To distribute in batches: {len(batch_prompts)}")
total_batches = len(batch_prompts)

# run batches
time_start = time.time()
for i, batch in enumerate(batch_prompts):
    pa_list = [(pa1, pa2) for _, _, _, pa1, pa2 in batch]
    idx_list = [idx for idx, _, _, _, _ in batch]
    texts    = [txt for _, txt, _, _, _ in batch]

    model_inputs = tokenizer(texts, return_tensors="pt", padding="longest", truncation=True).to(model.device)

    with torch.inference_mode():
        outputs = model(**model_inputs, use_cache = False)
        logits = outputs.logits[:, -1, [yes_token_id, no_token_id]]  # shape: (batch_size, seq_len, vocab_size)
    # last_token_logits = logits

    # yes_logits = last_token_logits[:, yes_token_id]  # shape: (batch_size,)
    # no_logits = last_token_logits[:, no_token_id]    # shape: (batch_size,)
    yes_logits = logits[:, 0]
    no_logits  = logits[:, 1]

    predictions = ["yes" if y > n else "no" for y, n in zip(yes_logits, no_logits)]
    
    assert len(predictions) == len(pa_list)
    for pa, pr in zip(pa_list, predictions):
        checked_pa[pa] = pr

    if i % 10 == 0 and i > 0:
        current_time = time.time()
        elapsed_time = current_time - time_start

        print(f"\rBatch: {i} / {total_batches}, time passed: {(elapsed_time/3600):.2f} hours", end="")

print()

for entry in to_do:
    pa = (entry[-2], entry[-1])
    res = checked_pa[pa]
    votes[entry[0]] =  res

assert len(votes) == (args.ways * len(episodes)), f"len votes: {len(votes)}, len epi: {len(episodes)}"
assert len(full_data) == len(episodes)

ep_votes = []
cur_votes = []
for ii, vote in enumerate(votes):
    assert vote == "yes" or vote == "no"
    cur_votes.append(vote)
    
    if (ii + 1) % args.ways == 0:
        updates_votes = []
        yes_found = False
        for v in cur_votes:
            if v == "yes" and yes_found == False:
                yes_found = True
                updates_votes.append("yes")
            else:
                updates_votes.append("no")


        ep_votes.append(updates_votes)
        cur_votes = []

assert len(ep_votes) == len(episodes)
assert len(ep_votes[0]) == args.ways

with open(NEW_ROOT_DATA_PATH + f'results_fewrel/{(args.model).replace("/","-")}-{args.shots}shots_{args.dataset}_query-{args.query}_is-summ-{args.is_summ}_ep-start-{args.ep_start}_ep-end-{args.ep_end}_{formatted_date}.txt', 'w') as f:
    for dd, ev in zip(full_data, ep_votes):
        f.write('@@@@@@@@@@@@@@@@@@@@@@@\n')
        f.write(str(dd[0]) + '\n')
        f.write('$$$$$ query relation\n')
        f.write(dd[1] + '\n')
        f.write('$$$$$ relation list\n')
        f.write(str(dd[2]) + '\n')
        f.write('$$$$$ votes\n')
        f.write(str(ev) + '\n')

print('\n************Ended***************')