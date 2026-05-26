#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

DEFAULT_TOKENIZERS = {
    "qwen": "Qwen/Qwen3-4B",
    "gemma": "google/gemma-3-4b-it",
}


def parse_prompt_records(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8")
    records: Dict[str, str] = {}

    chunks = re.split(r"(?=^###### CODE: )", text, flags=re.MULTILINE)
    for chunk in chunks:
        code_match = re.search(r"^###### CODE:\s*(.*?)\s*$", chunk, flags=re.MULTILINE)
        if not code_match:
            continue
        code = code_match.group(1).strip()
        status_match = re.search(r"^###### STATUS:.*?$", chunk, flags=re.MULTILINE)
        if not status_match:
            continue
        body = chunk[status_match.end() :].strip()
        if body:
            records[code] = body
    return records


def iter_node_gradpo_pairs(
    records: Dict[str, str],
    dataset_prefixes: Sequence[str],
) -> Iterable[Tuple[str, str]]:
    node_suffixes = ["_node_x", "_node_y"]

    for code in records:
        if not any(code.startswith(prefix) for prefix in dataset_prefixes):
            continue
        for base_suffix in node_suffixes:
            if not code.endswith(base_suffix):
                continue
            extension_code = code[: -len(base_suffix)] + f"{base_suffix}_gradpo-gen"
            if extension_code in records:
                yield code, extension_code


def token_len(tokenizer, text: str, *, add_special_tokens: bool) -> int:
    return len(tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])


def print_table(rows: List[Dict[str, object]]) -> None:
    if not rows:
        print("No matching _node_x / _node_x_gradpo-gen pairs found.")
        return

    headers = [
        "pair",
        "qwen_base",
        "qwen_gradpo",
        "qwen_delta",
        "gemma_base",
        "gemma_gradpo",
        "gemma_delta",
    ]
    widths = {
        header: max(len(header), *(len(str(row[header])) for row in rows))
        for header in headers
    }
    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(str(row[header]).ljust(widths[header]) for header in headers))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute token lengths for _node_x/_node_y prompts and matching "
            "_node_x_gradpo-gen/_node_y_gradpo-gen prompt extensions using "
            "tokenizer-only loading."
        )
    )
    parser.add_argument("--prompts-file", type=Path, default=Path("all_the_prompts.txt"))
    parser.add_argument(
        "--dataset-prefix",
        action="append",
        dest="dataset_prefixes",
        default=None,
        help=(
            "Dataset code prefix to include. Can be passed multiple times. "
            "Defaults to tacred_ and fewrel_."
        ),
    )
    parser.add_argument("--qwen-tokenizer", default=DEFAULT_TOKENIZERS["qwen"])
    parser.add_argument("--gemma-tokenizer", default=DEFAULT_TOKENIZERS["gemma"])
    parser.add_argument(
        "--add-special-tokens",
        action="store_true",
        help="Include tokenizer special tokens in the length counts.",
    )
    args = parser.parse_args()
    dataset_prefixes = args.dataset_prefixes or ["tacred_", "fewrel_"]

    records = parse_prompt_records(args.prompts_file)
    pairs = list(iter_node_gradpo_pairs(records, dataset_prefixes))

    print(f"Loaded prompt records: {len(records)}")
    print(f"Dataset prefixes: {', '.join(dataset_prefixes)}")
    print(f"Matched pairs: {len(pairs)}")
    print(f"add_special_tokens={args.add_special_tokens}")
    print(f"Qwen tokenizer: {args.qwen_tokenizer}")
    print(f"Gemma tokenizer: {args.gemma_tokenizer}")
    print()

    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install it in the environment where "
            "you run this script, e.g. `pip install transformers`."
        ) from exc

    qwen_tok = AutoTokenizer.from_pretrained(args.qwen_tokenizer, trust_remote_code=True)
    gemma_tok = AutoTokenizer.from_pretrained(args.gemma_tokenizer, trust_remote_code=True)

    rows: List[Dict[str, object]] = []
    for base_code, gradpo_code in pairs:
        qwen_base = token_len(qwen_tok, records[base_code], add_special_tokens=args.add_special_tokens)
        qwen_gradpo = token_len(qwen_tok, records[gradpo_code], add_special_tokens=args.add_special_tokens)
        gemma_base = token_len(gemma_tok, records[base_code], add_special_tokens=args.add_special_tokens)
        gemma_gradpo = token_len(gemma_tok, records[gradpo_code], add_special_tokens=args.add_special_tokens)
        rows.append(
            {
                "pair": f"{base_code} -> {gradpo_code}",
                "qwen_base": qwen_base,
                "qwen_gradpo": qwen_gradpo,
                "qwen_delta": qwen_gradpo - qwen_base,
                "gemma_base": gemma_base,
                "gemma_gradpo": gemma_gradpo,
                "gemma_delta": gemma_gradpo - gemma_base,
            }
        )

    print_table(rows)
    print()

    if rows:
        print("Averages across matched pairs")
        for key in ["qwen_base", "qwen_gradpo", "qwen_delta", "gemma_base", "gemma_gradpo", "gemma_delta"]:
            print(f"{key}: {mean(float(row[key]) for row in rows):.2f}")


if __name__ == "__main__":
    main()

# pair                                                                       qwen_base  qwen_gradpo  qwen_delta  gemma_base  gemma_gradpo  gemma_delta
# -------------------------------------------------------------------------  ---------  -----------  ----------  ----------  ------------  -----------
# tacred_qwen_rpo_node_x -> tacred_qwen_rpo_node_x_gradpo-gen                397        401          4           419         423           4          
# tacred_qwen_rpo_node_y -> tacred_qwen_rpo_node_y_gradpo-gen                524        516          -8          553         545           -8         
# tacred_qwen_evoprompt_node_x -> tacred_qwen_evoprompt_node_x_gradpo-gen    167        167          0           166         167           1          
# tacred_qwen_evoprompt_node_y -> tacred_qwen_evoprompt_node_y_gradpo-gen    149        149          0           147         147           0          
# tacred_qwen_etgpo_node_x -> tacred_qwen_etgpo_node_x_gradpo-gen            295        296          1           305         306           1          
# tacred_gemma_rpo_node_x -> tacred_gemma_rpo_node_x_gradpo-gen              246        243          -3          251         248           -3         
# tacred_gemma_rpo_node_y -> tacred_gemma_rpo_node_y_gradpo-gen              550        534          -16         549         540           -9         
# tacred_gemma_evoprompt_node_y -> tacred_gemma_evoprompt_node_y_gradpo-gen  203        206          3           200         201           1          
# tacred_gemma_etgpo_node_x -> tacred_gemma_etgpo_node_x_gradpo-gen          244        241          -3          247         244           -3         
# fewrel_qwen_rpo_node_x -> fewrel_qwen_rpo_node_x_gradpo-gen                301        301          0           307         307           0          
# fewrel_qwen_rpo_node_y -> fewrel_qwen_rpo_node_y_gradpo-gen                634        635          1           661         662           1          
# fewrel_qwen_etgpo_node_x -> fewrel_qwen_etgpo_node_x_gradpo-gen            230        231          1           229         230           1          
# fewrel_gemma_rpo_node_x -> fewrel_gemma_rpo_node_x_gradpo-gen              166        162          -4          168         164           -4         
# fewrel_gemma_rpo_node_y -> fewrel_gemma_rpo_node_y_gradpo-gen              210        210          0           218         218           0          
# fewrel_gemma_evoprompt_node_x -> fewrel_gemma_evoprompt_node_x_gradpo-gen  115        113          -2          116         114           -2         
# fewrel_gemma_evoprompt_node_y -> fewrel_gemma_evoprompt_node_y_gradpo-gen  103        101          -2          104         102           -2         

# Averages across matched pairs
# qwen_base: 283.38
# qwen_gradpo: 281.62
# qwen_delta: -1.75
# gemma_base: 290.00
# gemma_gradpo: 288.62
# gemma_delta: -1.38