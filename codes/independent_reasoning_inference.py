from __future__ import annotations

import json
import os
import pickle
import random
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available


# =========================
# Config (no CLI args)
# =========================
MODEL_ID = "Qwen/Qwen3-4B"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))
DATASET_TYPE = "fs_tacred"
DEV_SPLIT = "dev"  # validation split
DEV_EP_START = 0
DEV_EP_END = None  # None for full split

DEVICE_MAP = "cuda:2"  # None -> auto (cuda if available else cpu)
USE_CHAT_TEMPLATE = True
ADD_GENERATION_PROMPT = True
ENABLE_THINKING = True  # passed to apply_chat_template when supported

BATCH_SIZE = 50
MAX_NEW_TOKENS = 1000
MAX_INPUT_TOKENS = None  # e.g. 4096 or None

N_CHUNKS = 3  # for mean/std computation
SEED = 42

SAVE_RAW_RESPONSES = True
RAW_OUTPUT_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "qwen4_raw_responses_wf.json")
)


INFERENCE_PROMPT_GEMMA4_WF = """
You are an expert at recognizing connections between entities in text. You will be given a relationship definition, a clear example sentence demonstrating that relationship, and a new sentence. Your task is to decide if the same relationship exists between the subject and object entities in the new sentence.

Here's how to approach this task:

1.  Carefully analyze the provided relationship definition and example sentence to fully understand the nature of the connection.
2.  Identify the subject and object entities within the new sentence.
3.  Determine if the relationship described in the example sentence also applies to the subject and object in the new sentence.
4.  Analyze and answer with a concise "yes" or "no" – just one word.

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)

#SUPPORT_SENTENCE_BLOCK#

Query Sentence: #QUERY_SENTENCE#

Do reasoning for analysing. Then output the final answer here using tags: [d]yes[/d] or [d]no[/d].
"""

INFERENCE_PROMPT_GEMMA4_CMF = """You are given a relation name, a description of the relation in brackets, a support sentence that exemplifies the relation, and a query sentence. Your task is to determine if the query sentence demonstrates the specified relation between the Subject and Object entities, as indicated by the support sentence.

The Subject and Object entities are clearly marked within the sentence. You need to analyze the query sentence to see if it exhibits the same connection as described in the support sentence.  Focus on whether the subject and object share a direct and defined relationship, as illustrated by the example.  If the query sentence suggests the specified relation, respond with "yes"; otherwise, respond with "no".

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)

#SUPPORT_SENTENCE_BLOCK#

Query Sentence: #QUERY_SENTENCE#

Consider the context and specific wording of the query sentence carefully to determine if it aligns with the relationship described in the support instance. Do reasoning for analysing. Then output the final answer here using tags: [d]yes[/d] or [d]no[/d]."""

INFERENCE_PROMPT_QWEN4_CMF = """You are given a relation name, a description of the relation in brackets, a support sentence that exemplifies the relation, and a query sentence.

A relation connects the Subject and the Object entities. The Subject and the Object entities are indicated with subject and object tags, respectively. You need to decide whether the relation holds between the Subject and the Object in the query sentence.

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)

#SUPPORT_SENTENCE_BLOCK#

Query Sentence: #QUERY_SENTENCE#

When evaluating whether the relation holds, carefully consider the following:
- The relation must directly connect the Subject and Object entities as defined by the relation description.
- The Subject must be the entity described in the relation (e.g., a person for "per:spouse").
- The Object must be the specific entity that represents the relation's value (e.g., a city for "per:city_of_birth").
- Pay attention to explicit connections such as "born in," "died of," or "spouse of" that establish the relation.
- Be cautious of ambiguous phrases or geopolitical terms that may not directly indicate the relation.
- Ensure that the subject and object are correctly identified and that the relation is explicitly or implicitly expressed in the query.

If the relation holds between the Subject and Object in the query sentence, say using tags [d]yes[/d]; otherwise, [d]no[/d] and nothing else.
"""

INFERENCE_PROMPT_QWEN4_WF = """You are given a relation name, a description of the relation in brackets, a support sentence that exemplifies the relation, and a query sentence. Your task is to determine whether the relation described holds between the Subject and Object entities in the query sentence.

A relation connects two entities: the Subject and the Object. These entities are marked in the text with "subject" and "object" tags, respectively. You must analyze the query sentence to see if the relation described in the relation name and description is accurately represented between the Subject and Object entities.

Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)

#SUPPORT_SENTENCE_BLOCK#

Query Sentence: #QUERY_SENTENCE#

Based on the query sentence, determine if the relation holds between the Subject and Object. If it does, respond using tags [d]yes[/d]; otherwise, [d]no[/d] and nothing else.
"""

RELATION_DESCRIPTION = {
    "org:alternate_names": "an organization's alternate names",
    "org:city_of_headquarters": "an organization's city of headquarters",
    "org:country_of_headquarters": "an organization's country of headquarters",
    "org:dissolved": "an organization's date of dissolution",
    "org:founded": "an organization's date of founding",
    "org:founded_by": "an organization's founder",
    "org:member_of": "an organization's membership of another entity",
    "org:members": "an organization's members",
    "org:number_of_employees/members": "an organization's number of employees or members",
    "org:parents": "an organization's parents",
    "org:political/religious_affiliation": "an organization's political or religious affiliation",
    "org:shareholders": "an organization's shareholders",
    "org:stateorprovince_of_headquarters": "an organization's state or province of headquarters",
    "org:subsidiaries": "an organization's subsidiaries",
    "org:top_members/employees": "an organization's top members or employees",
    "org:website": "an organization's website",
    "per:age": "a person's age",
    "per:alternate_names": "a person's alternate names",
    "per:cause_of_death": "a person's cause of death",
    "per:charges": "a person's criminal charges",
    "per:children": "a person's children",
    "per:cities_of_residence": "a person's cities of residence",
    "per:city_of_birth": "a person's city of birth",
    "per:city_of_death": "a person's city of death",
    "per:countries_of_residence": "a person's countries of residence",
    "per:country_of_birth": "a person's country of birth",
    "per:country_of_death": "a person's country of death",
    "per:date_of_birth": "a person's date of birth",
    "per:date_of_death": "a person's date of death",
    "per:employee_of": "a person's employer",
    "per:origin": "a person's city or country of origin",
    "per:other_family": "a person's other family",
    "per:parents": "a person's parents",
    "per:religion": "a person's religion",
    "per:schools_attended": "schools attended by a person",
    "per:siblings": "a person's siblings",
    "per:spouse": "a person's spouse",
    "per:stateorprovince_of_birth": "a person's state or province of birth",
    "per:stateorprovince_of_death": "a person's state or province of death",
    "per:stateorprovinces_of_residence": "a person's state or province of residence",
    "per:title": "a person's title",
}


NO_RELATION = "no_relation"


def _default_device_map(device_map: Optional[str]) -> str:
    if device_map:
        return device_map
    if torch.cuda.is_available():
        return "cuda:3"
    return "cpu"


def _default_dtype(device_map: str) -> torch.dtype:
    return torch.bfloat16


def _default_attn_implementation(device_map: str) -> str | None:
    if device_map == "cpu":
        return None
    if torch.cuda.is_available() and is_flash_attn_2_available():
        print("[independent] flash_attention_2 found")
        return "flash_attention_2"
    return None


def load_model_and_tokenizer(model_id: str, device_map: Optional[str] = None):
    print(f"[independent] loading model: {model_id}")

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    device_map = _default_device_map(device_map)
    attn_implementation = _default_attn_implementation(device_map)
    model_kwargs = {
        "torch_dtype": _default_dtype(device_map),
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    print(f"[independent] model loading done: {model_id}")
    return model, tokenizer


def read_pickle_file(file_path: str) -> Any:
    with open(file_path, "rb") as handle:
        return pickle.load(handle)


def load_split_episodes(
    *,
    split: str,
    data_dir: str,
    dataset_type: str,
    ep_start: int = 0,
    ep_end: Optional[int] = None,
) -> Dict[str, Any]:
    episodes_path = os.path.join(
        data_dir, f"{dataset_type}_{split}_episodes_1shots.pkl"
    )
    details_path = os.path.join(
        data_dir, f"{dataset_type}_{split}_episodes_shots_details.pkl"
    )

    episodes: List[Dict[str, Any]] = read_pickle_file(episodes_path)
    if ep_end is None:
        episodes = episodes[ep_start:]
    else:
        episodes = episodes[ep_start:ep_end]

    details = read_pickle_file(details_path)
    return {
        "episodes": episodes,
        "shots": details.get("shots", {}),
        "queries": details.get("queries", {}),
        "umbc_shots": details.get("umbc_shots", {}),
    }


def get_sentence_with_tags(meta_data: Dict) -> str:
    token_with_tags = list(meta_data["token"])
    token_with_tags[meta_data["subj_start"]] = (
        "<subject>" + token_with_tags[meta_data["subj_start"]]
    )
    token_with_tags[meta_data["subj_end"]] = (
        token_with_tags[meta_data["subj_end"]] + "</subject>"
    )
    token_with_tags[meta_data["obj_start"]] = (
        "<object>" + token_with_tags[meta_data["obj_start"]]
    )
    token_with_tags[meta_data["obj_end"]] = (
        token_with_tags[meta_data["obj_end"]] + "</object>"
    )
    return " ".join(token_with_tags)


def build_support_block(support_sentences: Sequence[str]) -> str:
    if not support_sentences:
        return ""
    if len(support_sentences) == 1:
        return f"Support Sentence: {support_sentences[0]}"
    lines = []
    for idx, sent in enumerate(support_sentences, start=1):
        lines.append(f"Support Sentence {idx}: {sent}")
    return "\n".join(lines)


def resolve_way_shots(way_ids: Sequence[int], shots: Dict) -> List[Dict]:
    resolved = []
    if not way_ids:
        return resolved
    resolved.append(shots["shots"][way_ids[0]])
    return resolved


def get_relation_description(relation: str, dataset_type: str) -> str:
    if dataset_type == "fs_tacred":
        return RELATION_DESCRIPTION.get(relation, "")
    return ""


def format_inference_prompt(
    base_prompt: str,
    relation: str,
    relation_description: str,
    support_sentences: Sequence[str],
    query_sentence: str,
) -> str:
    support_block = build_support_block(support_sentences)
    prompt = base_prompt
    prompt = prompt.replace("#RELATION#", relation)
    prompt = prompt.replace("#RELATION_DESCRIPTION#", relation_description)
    prompt = prompt.replace("#SUPPORT_SENTENCE_BLOCK#", support_block)
    prompt = prompt.replace("#QUERY_SENTENCE#", query_sentence)
    return prompt


def _batched(items: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _extract_decision(text: str) -> str:
    lower = text.lower()
    match = re.search(r"\[d\]\s*(yes|no)\s*\[/d\]", lower)
    if match:
        return match.group(1)
    tokens = re.findall(r"\b(yes|no)\b", lower)
    if tokens:
        return tokens[-1]
    return "no"


def run_reasoned_inference(
    prompts: Sequence[str],
    *,
    model,
    tokenizer,
    batch_size: int,
    max_new_tokens: int,
    use_chat_template: bool,
    add_generation_prompt: bool,
    enable_thinking: bool,
    max_input_tokens: Optional[int],
    save_raw: bool,
    raw_output_file: str,
) -> Tuple[List[str], List[str]]:
    if not prompts:
        return [], []

    start_time = time.perf_counter()
    predictions: List[str] = []
    raw_responses: List[str] = []
    target_device = getattr(model, "device", None)

    for batch_index, batch in enumerate(_batched(list(prompts), batch_size), start=1):
        if use_chat_template:
            formatted = []
            for prompt in batch:
                try:
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                        enable_thinking=enable_thinking,
                    )
                except TypeError:
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=add_generation_prompt,
                    )
                formatted.append(text)
        else:
            formatted = list(batch)

        token_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }
        if max_input_tokens is not None:
            token_kwargs["max_length"] = max_input_tokens

        model_inputs = tokenizer(formatted, **token_kwargs)
        if target_device is not None:
            model_inputs = model_inputs.to(target_device)

        with torch.inference_mode():
            output_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = model_inputs["input_ids"].shape[1]
        generated = output_ids[:, input_len:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for prompt, text in zip(batch, decoded):
            raw_responses.append(text)
            predictions.append(_extract_decision(text))
            # defer saving until all outputs are collected

        elapsed = time.perf_counter() - start_time
        print(f"[independent] processed {batch_index} batches in {elapsed:.2f}s", flush=True)
        if batch_index % 20 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        del model_inputs, output_ids, generated, decoded

    elapsed = time.perf_counter() - start_time
    print(f"[independent] inference done in {elapsed:.2f}s")
    if save_raw:
        payload = {
            str(idx + 1): {"prompt": p, "response": r}
            for idx, (p, r) in enumerate(zip(prompts, raw_responses))
        }
        with open(raw_output_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        print(f"[independent] raw responses saved to {raw_output_file}")

    return predictions, raw_responses


def _chunk_indices(total: int, n_chunks: int) -> List[Tuple[int, int]]:
    if n_chunks <= 0:
        raise ValueError("n_chunks must be >= 1")
    base = total // n_chunks
    remainder = total % n_chunks
    indices = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < remainder else 0)
        end = start + size
        indices.append((start, end))
        start = end
    return indices


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var**0.5


def _score_micro(labels: Sequence[str], predictions: Sequence[str]) -> Tuple[float, float, float]:
    correct = 0
    guessed = 0
    gold = 0
    for gold_label, pred_label in zip(labels, predictions):
        if gold_label == NO_RELATION and pred_label == NO_RELATION:
            continue
        if gold_label == NO_RELATION and pred_label != NO_RELATION:
            guessed += 1
        elif gold_label != NO_RELATION and pred_label == NO_RELATION:
            gold += 1
        else:
            guessed += 1
            gold += 1
            if gold_label == pred_label:
                correct += 1
    prec = float(correct) / float(guessed) if guessed > 0 else 1.0
    rec = float(correct) / float(gold) if gold > 0 else 0.0
    f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_prf_stats(
    labels: Sequence[str],
    predictions: Sequence[str],
    *,
    n_chunks: int,
) -> Dict[str, float]:
    if not labels:
        return {
            "precision_mean": 0.0,
            "precision_std": 0.0,
            "recall_mean": 0.0,
            "recall_std": 0.0,
            "f1_mean": 0.0,
            "f1_std": 0.0,
        }

    n_chunks = max(1, min(n_chunks, len(labels)))
    chunk_bounds = _chunk_indices(len(labels), n_chunks)

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for start, end in chunk_bounds:
        if start == end:
            continue
        p, r, f1 = _score_micro(labels[start:end], predictions[start:end])
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    p_mean, p_std = _mean_std(precisions)
    r_mean, r_std = _mean_std(recalls)
    f1_mean, f1_std = _mean_std(f1s)

    return {
        "precision_mean": p_mean,
        "precision_std": p_std,
        "recall_mean": r_mean,
        "recall_std": r_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
    }


def main() -> None:
    random.seed(SEED)
    print("[independent] loading dataset")
    dev_data = load_split_episodes(
        split=DEV_SPLIT,
        data_dir=DATA_DIR,
        dataset_type=DATASET_TYPE,
        ep_start=DEV_EP_START,
        ep_end=DEV_EP_END,
    )
    episodes = dev_data["episodes"]
    shots = dev_data["shots"]
    queries = dev_data["queries"]

    print(f"[independent] dev episodes={len(episodes)}")

    model, tokenizer = load_model_and_tokenizer(MODEL_ID, device_map=DEVICE_MAP)

    prompts: List[str] = []
    episode_relations: List[List[str]] = []
    episode_labels: List[str] = []

    for episode in episodes:
        ways = episode["meta_train"]
        query_id = episode["meta_test"][0]
        query = queries[query_id]
        query_relation = query["relation"]
        query_sentence = get_sentence_with_tags(query).strip()

        relations: List[str] = []
        for way in ways:
            way_shots = resolve_way_shots(way, dev_data)
            relation = way_shots[0]["relation"]
            relations.append(relation)
            support_sentences = [get_sentence_with_tags(s).strip() for s in way_shots]
            relation_description = get_relation_description(relation, DATASET_TYPE)

            prompts.append(
                format_inference_prompt(
                    base_prompt=INFERENCE_PROMPT_QWEN4_WF,
                    relation=relation,
                    relation_description=relation_description,
                    support_sentences=support_sentences,
                    query_sentence=query_sentence,
                )
            )

        episode_relations.append(relations)
        episode_labels.append(query_relation if query_relation in relations else NO_RELATION)

    print(f"[independent] total prompts={len(prompts)} batch_size={BATCH_SIZE}")

    pair_predictions, _raw_responses = run_reasoned_inference(
        prompts,
        model=model,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        use_chat_template=USE_CHAT_TEMPLATE,
        add_generation_prompt=ADD_GENERATION_PROMPT,
        enable_thinking=ENABLE_THINKING,
        max_input_tokens=MAX_INPUT_TOKENS,
        save_raw=SAVE_RAW_RESPONSES,
        raw_output_file=RAW_OUTPUT_FILE,
    )

    episode_predictions: List[str] = []
    offset = 0
    for relations in episode_relations:
        chunk = pair_predictions[offset : offset + len(relations)]
        offset += len(relations)
        predicted_relation = NO_RELATION
        for relation, pred in zip(relations, chunk):
            if pred == "yes":
                predicted_relation = relation
                break
        episode_predictions.append(predicted_relation)

    metrics = compute_prf_stats(episode_labels, episode_predictions, n_chunks=N_CHUNKS)
    print(
        "[independent] validation score: "
        f"precision={metrics['precision_mean'] * 100:.2f}±{metrics['precision_std'] * 100:.2f}, "
        f"recall={metrics['recall_mean'] * 100:.2f}±{metrics['recall_std'] * 100:.2f}, "
        f"f1={metrics['f1_mean'] * 100:.2f}±{metrics['f1_std'] * 100:.2f}"
    )


if __name__ == "__main__":
    main()
