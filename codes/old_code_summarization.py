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
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from functools import partial

transformers.logging.set_verbosity_error()

####### Logging

access_token = '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'
login(token=access_token)


model_id = "Qwen/Qwen3-14B"
# model_id = "microsoft/Phi-4-mini-instruct"
# model_id = "google/gemma-3-4b-it"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"

DEVICE_MAP = "cuda:2"

print(model_id)
print(f"Device: {DEVICE_MAP}")

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE_MAP,
    # attn_implementation="eager" # **************************** For Titan RTX i.e. Cuda:1
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = 'left'

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id


def get_sentence_with_tags(meta_data):
    token_with_tags = meta_data['token'].copy()

    token_with_tags[meta_data['subj_start']] = '<subject>' + token_with_tags[meta_data['subj_start']]
    token_with_tags[meta_data['subj_end']] = token_with_tags[meta_data['subj_end']] + '</subject>'
    token_with_tags[meta_data['obj_start']] = '<object>' + token_with_tags[meta_data['obj_start']]
    token_with_tags[meta_data['obj_end']] = token_with_tags[meta_data['obj_end']] + '</object>'

    return ' '.join(token_with_tags)

def get_subject(meta_data):
    token_with_tags = meta_data['token'].copy()
    return ' '.join(token_with_tags[meta_data['subj_start']: 1 + meta_data['subj_end']])

def get_object(meta_data):
    token_with_tags = meta_data['token'].copy()
    return ' '.join(token_with_tags[meta_data['obj_start']: 1 + meta_data['obj_end']])


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
    
file_ = "fs_fewrel_shots_details.pkl"
NEW_ROOT_DIR = '/storage2/home/aunabilchakma/data/relation_extraction/'
test_data = read_pickle_file(os.path.join(NEW_ROOT_DIR, file_))

print(f"Running for Summarization model: {model_id}")
print(f"Processing core test sentences file: {file_}")

# summ_map = read_pickle_file(NEW_ROOT_DIR + f"fs_tacred_summ_map_umbcshots_{(model_id).replace('/','-')}.pkl")
# print(f"Cur size: {len(summ_map)}")
summ_map = dict()
umbc_ids = set(read_pickle_file(NEW_ROOT_DIR + "fs_fewrel_umbc_ids_to_summarize_v2.pkl"))
print(f"To summarize: {len(umbc_ids)}")

base_prompt = '''Summarize the relations between “#SUBJECT#” (Subject) and “#OBJECT#” (Object) from context. They are marked by subject and object tags repectively in the context.
Context: #SUPPORT_SENTENCE#

You must retain the respective <subject> and <object> tags wrapping the subject and the object in the summarized version. Just output the summarized relation between the subject and the object and nothing else.
'''

import hashlib
def sentence_to_id(sentence):
    normalized = sentence.lower().strip()
    hash_object = hashlib.sha256(normalized.encode('utf-8'))
    return hash_object.hexdigest()

time_start = time.time()
# for ei, (id_, shot) in enumerate(test_data['shots'].items()):

cnt = 0
for ei, (id_, shot) in enumerate(test_data['umbc_shots'].items()):

    # ******** If umbc_shots
    if id_ not in umbc_ids: 
        continue

    cnt += 1


    sentence = get_sentence_with_tags(shot).strip()
    subb = get_subject(shot)
    objj = get_object(shot)

    id256 = sentence_to_id(sentence) 
    if id256 in summ_map:
        continue

    new_prompt = base_prompt.replace("#SUPPORT_SENTENCE#", sentence)
    new_prompt = new_prompt.replace("#SUBJECT#", subb)
    new_prompt = new_prompt.replace("#OBJECT#", objj)

    # print(new_prompt)
    # break

    correct = False
    tried = 0
    while correct == False:
        if tried == 3:
            break
        tried += 1

        messages = [{"role": "user", "content": new_prompt}]
        formated_inputs = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking = False)
        model_inputs = tokenizer(formated_inputs, return_tensors="pt", padding = True, truncation = True).to(model.device)
        generated_ids = model.generate(**model_inputs,max_new_tokens=60)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        summ_map[id256] = response[0]

        v = response

        correct = True
        assert len(v) == 1
        if "<subject>" not in v[0].lower():
            correct = False
        if "</subject>" not in v[0].lower():
            correct = False
        if "<object>" not in v[0].lower():
            correct = False
        if "</object>" not in v[0].lower():
            correct = False 

    if ei % 5 == 0 and ei > 0:
        current_time = time.time()
        elapsed_time = current_time - time_start

        print(f"\rEpisode: {ei}, time passed: {elapsed_time:.2f} seconds, Len summ: {len(summ_map)}", end="")

print()
print(f"Final size: {len(summ_map)}")
print("Saving...")
save_dict(summ_map, os.path.join(NEW_ROOT_DIR, f"fs_fewrel_umbcshots_v2_summ_map_{(model_id).replace('/','-')}.pkl"))
print("Done finally")

print('\n************Ended***************')