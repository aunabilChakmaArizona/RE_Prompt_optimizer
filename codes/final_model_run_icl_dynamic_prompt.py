##### Arg parse and time

import argparse
import re
from pathlib import Path

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
parser.add_argument('--prompt', type=str, required=True, help='Optimized instruction prompt')
parser.add_argument('--code', type=str, required=True, help='Prompt/run code used in saved output filename')
parser.add_argument('--data_root', type=str, default=None, help='Directory containing relation extraction data files')
parser.add_argument('--shots_file', type=str, default=None, help='Optional shots details pickle path')
parser.add_argument('--entity_map_file', type=str, default=None, help='Optional entity resolution pickle path')
parser.add_argument('--output_dir', type=str, default=None, help='Directory for output vote txt files')

args = parser.parse_args()


from datetime import datetime
today = datetime.today()
formatted_date = today.strftime("%Y%m%d")  # %Y = 4-digit year, %m = 2-digit month, %d = 2-digit day

print(f"*********** Running Experiment on: {formatted_date}, Code:{args.code}, Model:{args.model}, Data: {args.dataset}, Shots: {args.shots}, Query: {args.query}, CUDA: {args.cuda} ********, IS_SUMM: {args.is_summ}")

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
from agents.agent_prompts import (
    INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1,
    INFERENCE_INPUT_PROMPT_V1,
    INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
    compose_inference_prompt,
)

transformers.logging.set_verbosity_error()

####### Logging

access_token = os.environ.get("HF_TOKEN")
if not access_token:
    raise RuntimeError("HF_TOKEN environment variable is required for Hugging Face login.")
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


INSTRUCTION_PROMPT = args.prompt.strip()
if not INSTRUCTION_PROMPT:
    raise ValueError("--prompt must not be empty")


def build_support_block(support_sentences):
    if not support_sentences:
        return ""
    if len(support_sentences) == 1:
        return f"Support Sentence: {support_sentences[0]}"
    return "\n".join(
        f"Support Sentence {idx}: {sentence}"
        for idx, sentence in enumerate(support_sentences, start=1)
    )


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
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(args.data_root).expanduser() if args.data_root else REPO_ROOT / "data"
OUTPUT_DIR = Path(args.output_dir).expanduser() if args.output_dir else REPO_ROOT / "outputs" / "opt_prompt"

if not DATA_ROOT.is_absolute():
    DATA_ROOT = (REPO_ROOT / DATA_ROOT).resolve()
if not OUTPUT_DIR.is_absolute():
    OUTPUT_DIR = (REPO_ROOT / OUTPUT_DIR).resolve()


def resolve_data_path(path_value):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return DATA_ROOT / path


def infer_shots_file(dataset_name, dataset_core):
    if args.shots_file:
        return resolve_data_path(args.shots_file)

    candidates = []
    if dataset_name.endswith("_episodes_1shots.pkl"):
        candidates.append(DATA_ROOT / dataset_name.replace("_episodes_1shots.pkl", "_episodes_shots_details.pkl"))
    if dataset_name.endswith("_episodes.pkl"):
        candidates.append(DATA_ROOT / dataset_name.replace("_episodes.pkl", "_episodes_shots_details.pkl"))
    candidates.extend([
        DATA_ROOT / f"fs_{dataset_core}_test_episodes_shots_details.pkl",
        DATA_ROOT / f"fs_{dataset_core}_shots_details.pkl",
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not infer shots details file. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


file_path = resolve_data_path(file_)
shots_file_path = infer_shots_file(file_, args.dataset_core)
episodes = read_pickle_file(file_path)[args.ep_start:args.ep_end]
shots = read_pickle_file(shots_file_path)

print("Format: " + prompt_type + prompt_details)
print("Running: " + file_)
print("Data root: " + str(DATA_ROOT))
print("Shots details: " + str(shots_file_path))

if args.is_summ == True:
    summ_map_path = DATA_ROOT / f"fs_{args.dataset_core}_v3_summ_map_{(model_id).replace('/','-')}.pkl"
    summ_map = read_pickle_file(summ_map_path)
else:
    summ_map = dict()

if args.entity_map_file:
    entity_map_path = resolve_data_path(args.entity_map_file)
elif (DATA_ROOT / f"fs_{args.dataset_core}_entity_resolution_{(model_id).replace('/','-')}.pkl").exists():
    entity_map_path = DATA_ROOT / f"fs_{args.dataset_core}_entity_resolution_{(model_id).replace('/','-')}.pkl"
else:
    entity_map_path = None

if entity_map_path is None:
    entity_map = None
    print("Entity filter: disabled (entity map file not found)")
else:
    entity_map = read_pickle_file(entity_map_path)
    print("Entity filter: " + str(entity_map_path))
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

    query_relation = query['relation']
    query_sentence = get_sentence_with_tags(query)
    relation_list = []
    

    for idx_way, way in enumerate(ways):
        way_ = []
        way_.append(shots["shots"][way[0]])
        for w in way[1:]:
            if w in shots["shots"]:
                way_.append(shots["shots"][w])
            else:
                way_.append(shots.get("umbc_shots", {})[w])
        way = way_


        idx_way = idx_way + 1

        relation = way[0]['relation']
        relation_list.append(relation)

        support_sentences = []
        for shot in way:
            assert shot['relation'] == relation

            support_sentence = get_sentence_with_tags(shot).strip()
            id256 = sentence_to_id(support_sentence)

            if args.is_summ == True:
                assert id256 in summ_map
                support_sentence = summ_map[id256]

            support_sentences.append(support_sentence)

            if '1_shots' in prompt_type:
                break

        support_block = build_support_block(support_sentences)
        prompt_ = compose_inference_prompt(
            inference_mode=INFERENCE_MODE_SEPARATE_NO_EXAMPLES,
            inference_prompt="",
            inference_instruction_prompt=INSTRUCTION_PROMPT,
            inference_answer_instruction_prompt=INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1,
            inference_example_prompt="",
            inference_input_prompt=INFERENCE_INPUT_PROMPT_V1,
            relation=relation,
            relation_description=prompts.get_relation_description(relation, dt=args.dataset_core),
            support_block=support_block,
            query_sentence=query_sentence,
            example_query_sentence=support_sentences[0] if support_sentences else query_sentence,
        )

        messages = [{"role": "user", "content": prompt_}]
        prompts_.append(messages)
        if len(prompts_) == 1:
            print(prompt_)

    #     print(prompt_)
    #     break
    # break

        if entity_map is not None and entity_map.get(qr, {}).get(relation) == False:
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

def safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
safe_code = safe_filename(args.code)
safe_model = safe_filename(args.model.replace("/", "-"))
safe_dataset = safe_filename(Path(args.dataset).name)
output_path = OUTPUT_DIR / f'{safe_code}_{safe_model}-{args.shots}shots_{safe_dataset}_query-{args.query}_is-summ-{args.is_summ}_ep-start-{args.ep_start}_ep-end-{args.ep_end}_{formatted_date}.txt'

with open(output_path, 'w') as f:
    for dd, ev in zip(full_data, ep_votes):
        f.write('@@@@@@@@@@@@@@@@@@@@@@@\n')
        f.write(str(dd[0]) + '\n')
        f.write('$$$$$ query relation\n')
        f.write(dd[1] + '\n')
        f.write('$$$$$ relation list\n')
        f.write(str(dd[2]) + '\n')
        f.write('$$$$$ votes\n')
        f.write(str(ev) + '\n')

print("Saved: " + str(output_path))
print('\n************Ended***************')
