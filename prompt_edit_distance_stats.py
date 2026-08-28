#!/usr/bin/env python3
"""Measure stage-2 prompt edits relative to their stage-1 prompts.

Distances are unit-cost Levenshtein distances over three sequence types:

* characters: Unicode code points, including internal whitespace/newlines;
* words: non-whitespace strings (``re.findall(r"\\S+", prompt)``);
* tokens: token IDs from the target model's tokenizer, without special tokens.

An empty stage-2 prompt means that the optimizer retained its stage-1 input, so
it receives distance zero. A case with an empty stage-1 prompt is not a valid
comparison and is excluded from every method's denominator.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterable, List, Sequence, TypeVar


DEFAULT_TOKENIZERS = {
    "qwen": "Qwen/Qwen3-4B",
    "gemma": "google/gemma-3-4b-it",
}
SECOND_STAGE_METHODS = (
    "lpo",
    "gradpo-gen",
    "gradpo-prob",
    "greater",
    "greater-tg",
)

FIRST_STAGE_RE = re.compile(
    r"^(?P<dataset>tacred|fewrel)_"
    r"(?P<model>qwen|gemma)_"
    r"(?P<first_stage>rpo|evoprompt|etgpo)_"
    r"node_(?P<iteration>x|y)$"
)
SECOND_STAGE_RE = re.compile(
    r"^(?P<base>"
    r"(?P<dataset>tacred|fewrel)_"
    r"(?P<model>qwen|gemma)_"
    r"(?P<first_stage>rpo|evoprompt|etgpo)_"
    r"node_(?P<iteration>x|y))_"
    r"(?P<second_stage>lpo|gradpo-gen|gradpo-prob|greater|greater-tg)$"
)

T = TypeVar("T")


@dataclass(frozen=True)
class Comparison:
    base_code: str
    stage2_code: str
    second_stage: str
    model: str
    selected_change: bool
    char_distance: int
    word_distance: int
    token_distance: int


def parse_prompt_records(path: Path) -> Dict[str, str]:
    """Return every CODE and the text after STATUS, including empty bodies."""
    text = path.read_text(encoding="utf-8")
    records: Dict[str, str] = {}
    chunks = re.split(r"(?=^###### CODE:)", text, flags=re.MULTILINE)

    for chunk in chunks:
        code_match = re.search(
            r"^###### CODE:\s*(.*?)\s*$", chunk, flags=re.MULTILINE
        )
        if not code_match:
            continue
        status_match = re.search(r"^###### STATUS:.*?$", chunk, flags=re.MULTILINE)
        if not status_match:
            raise ValueError(f"Record has no STATUS line: {code_match.group(1)!r}")
        code = code_match.group(1).strip()
        if code in records:
            raise ValueError(f"Duplicate CODE: {code}")
        records[code] = chunk[status_match.end() :].strip()

    return records


def levenshtein_distance(left: Sequence[T], right: Sequence[T]) -> int:
    """Return unit-cost insertion/deletion/substitution edit distance."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def validate_layout(records: Dict[str, str]) -> List[str]:
    """Validate the canonical first/second-stage grid and return base codes."""
    base_codes = sorted(code for code in records if FIRST_STAGE_RE.fullmatch(code))
    second_codes = [code for code in records if SECOND_STAGE_RE.fullmatch(code)]
    if len(base_codes) != 20:
        raise ValueError(f"Expected 20 canonical first-stage records; found {len(base_codes)}")
    if len(second_codes) != 100:
        raise ValueError(
            f"Expected 100 canonical second-stage records; found {len(second_codes)}"
        )

    missing = [
        f"{base}_{method}"
        for base in base_codes
        for method in SECOND_STAGE_METHODS
        if f"{base}_{method}" not in records
    ]
    if missing:
        raise ValueError("Missing canonical second-stage records: " + ", ".join(missing))
    return base_codes


def load_tokenizers(
    qwen_name: str, gemma_name: str, *, local_files_only: bool
) -> Dict[str, object]:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "transformers is required for token distances. Run this script in an "
            "environment that provides it."
        ) from exc

    def resolve_local_snapshot(name_or_path: str) -> str:
        """Turn a cached Hub model ID into a local path for strict offline use.

        Transformers 5 may perform a Hub metadata request for some tokenizer
        classes even when ``local_files_only=True``. Passing the snapshot path
        avoids that request entirely.
        """
        path = Path(name_or_path).expanduser()
        if path.exists() or not local_files_only:
            return str(path) if path.exists() else name_or_path
        try:
            from huggingface_hub import try_to_load_from_cache
        except ModuleNotFoundError:
            return name_or_path
        cached_config = try_to_load_from_cache(name_or_path, "tokenizer_config.json")
        if isinstance(cached_config, str):
            # Keep the snapshot symlink path. Resolving it would point at the
            # individual file's blob directory rather than the full snapshot.
            return str(Path(cached_config).parent)
        return name_or_path

    qwen_source = resolve_local_snapshot(qwen_name)
    gemma_source = resolve_local_snapshot(gemma_name)
    return {
        "qwen": AutoTokenizer.from_pretrained(
            qwen_source,
            trust_remote_code=True,
            local_files_only=local_files_only,
        ),
        "gemma": AutoTokenizer.from_pretrained(
            gemma_source,
            trust_remote_code=True,
            local_files_only=local_files_only,
        ),
    }


def build_comparisons(
    records: Dict[str, str],
    base_codes: Iterable[str],
    tokenizers: Dict[str, object],
) -> tuple[List[Comparison], List[str]]:
    comparisons: List[Comparison] = []
    skipped_bases: List[str] = []

    @lru_cache(maxsize=None)
    def token_ids(model: str, prompt: str) -> tuple[int, ...]:
        tokenizer = tokenizers[model]
        return tuple(tokenizer.encode(prompt, add_special_tokens=False))

    for base_code in base_codes:
        base_prompt = records[base_code]
        if not base_prompt:
            skipped_bases.append(base_code)
            continue
        base_match = FIRST_STAGE_RE.fullmatch(base_code)
        assert base_match is not None
        model = base_match.group("model")

        for method in SECOND_STAGE_METHODS:
            stage2_code = f"{base_code}_{method}"
            stored_stage2_prompt = records[stage2_code]
            selected_change = bool(stored_stage2_prompt)
            # Empty means no improvement was selected, hence the effective final
            # prompt is identical to the stage-1 input.
            stage2_prompt = stored_stage2_prompt or base_prompt
            comparisons.append(
                Comparison(
                    base_code=base_code,
                    stage2_code=stage2_code,
                    second_stage=method,
                    model=model,
                    selected_change=selected_change,
                    char_distance=levenshtein_distance(base_prompt, stage2_prompt),
                    word_distance=levenshtein_distance(
                        re.findall(r"\S+", base_prompt),
                        re.findall(r"\S+", stage2_prompt),
                    ),
                    token_distance=levenshtein_distance(
                        token_ids(model, base_prompt),
                        token_ids(model, stage2_prompt),
                    ),
                )
            )
    return comparisons, skipped_bases


def write_details(path: Path, comparisons: Sequence[Comparison]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(Comparison.__dataclass_fields__))
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow(comparison.__dict__)


def format_summary(
    comparisons: Sequence[Comparison], *, changed_only: bool = False
) -> str:
    grouped: Dict[str, List[Comparison]] = defaultdict(list)
    for comparison in comparisons:
        grouped[comparison.second_stage].append(comparison)

    headers = (
        "method",
        "n",
        "changed",
        "char_total",
        "char_avg",
        "word_total",
        "word_avg",
        "token_total",
        "token_avg",
    )
    rows: List[List[str]] = []
    for method in SECOND_STAGE_METHODS:
        values = grouped[method]
        if changed_only:
            values = [value for value in values if value.selected_change]
        rows.append(
            [
                method,
                str(len(values)),
                str(sum(value.selected_change for value in values)),
                str(sum(value.char_distance for value in values)),
                f"{mean(value.char_distance for value in values):.2f}",
                str(sum(value.word_distance for value in values)),
                f"{mean(value.word_distance for value in values):.2f}",
                str(sum(value.token_distance for value in values)),
                f"{mean(value.token_distance for value in values):.2f}",
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-file", type=Path, default=Path("all_the_prompts.txt"))
    parser.add_argument(
        "--details-csv", type=Path, default=Path("prompt_edit_distance_details.csv")
    )
    parser.add_argument("--qwen-tokenizer", default=DEFAULT_TOKENIZERS["qwen"])
    parser.add_argument("--gemma-tokenizer", default=DEFAULT_TOKENIZERS["gemma"])
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require both tokenizer configurations to already be cached.",
    )
    args = parser.parse_args()

    records = parse_prompt_records(args.prompts_file)
    base_codes = validate_layout(records)
    tokenizers = load_tokenizers(
        args.qwen_tokenizer,
        args.gemma_tokenizer,
        local_files_only=args.local_files_only,
    )
    comparisons, skipped_bases = build_comparisons(records, base_codes, tokenizers)
    write_details(args.details_csv, comparisons)

    print(f"Canonical first-stage records: {len(base_codes)}")
    print(f"Excluded empty first-stage records: {len(skipped_bases)}")
    for code in skipped_bases:
        print(f"  {code}")
    print(f"Comparisons per method: {len(comparisons) // len(SECOND_STAGE_METHODS)}")
    print("Empty second-stage bodies are counted as unchanged (distance 0).")
    print("Token IDs exclude tokenizer-added special tokens.")
    print()
    print(format_summary(comparisons))
    print()
    print("Changed prompts only")
    print(format_summary(comparisons, changed_only=True))
    print()
    print(f"Per-comparison details: {args.details_csv}")


if __name__ == "__main__":
    main()
