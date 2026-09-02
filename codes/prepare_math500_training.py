"""Prepare leakage-free MATH training and validation data for MATH-500."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


SOURCE_DATASET = "DigitalLearningGmbH/MATH-lighteval"
SOURCE_REVISION = "0530c78699ea5e8eb5530600900e1f328b48acad"
SOURCE_FILE = "data/train-00000-of-00001.parquet"
SOURCE_URL = (
    f"https://huggingface.co/datasets/{SOURCE_DATASET}/resolve/"
    f"{SOURCE_REVISION}/{SOURCE_FILE}"
)
EXPECTED_SHA256 = "eca6e667f4305dd5e5ba09b4fd55e7f3174a0fbe361cdfd4c44758b593a76933"
VALIDATION_SIZE = 900
VALIDATION_FOLD_COUNT = 3
SPLIT_SEED = 42

EMPTY_BOX_ANSWER_OVERRIDES = {
    "for any integer $n>1$, the number of prime numbers greater than $n!+1$ and less than $n!+n$ is: "
    "$\\text{(a) } 0\\quad\\qquad \\text{(b) } 1\\quad\\\\ \\text{(c) } \\frac": "0",
    "you are given a sequence of $58$ terms; each term has the form $p+n$ where $p$ stands for the product "
    "$2 \\times 3 \\times 5 \\times\\ldots \\times 61$ of all prime": "0",
}


def find_repo_root() -> Path:
    """Find the repository root from the script location."""
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "AGENTS.md").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not find the repository root.")


def file_sha256(path: Path) -> str:
    """Calculate the SHA-256 checksum of a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_question(text: str) -> str:
    """Normalize a question for exact duplicate and overlap checks."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON objects from a JSONL file."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def extract_braced_value(text: str, opening_index: int) -> str | None:
    """Extract a possibly nested braced value beginning at an opening brace."""
    depth = 0
    for index in range(opening_index, len(text)):
        character = text[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[opening_index + 1 : index].strip()
    return None


def extract_last_boxed_answer(solution: str) -> str | None:
    """Extract the final braced or unbraced boxed answer from a MATH solution."""
    markers = list(re.finditer(r"\\(?:boxed|fbox)\b", solution))
    if not markers:
        return None

    marker = markers[-1]
    remainder = solution[marker.end() :].lstrip()
    if remainder.startswith("{"):
        opening_index = len(solution) - len(remainder)
        answer = extract_braced_value(solution, opening_index)
        return answer or None

    unbraced_match = re.match(r"([^\s$.,;]+)", remainder)
    return unbraced_match.group(1).strip() if unbraced_match else None


def parse_level(value: str) -> int | None:
    """Convert a MATH level label to an integer when one is available."""
    match = re.fullmatch(r"Level\s+([1-5])", value.strip())
    return int(match.group(1)) if match else None


def answer_override(question: str) -> str | None:
    """Return a documented answer for one of the two empty-box source rows."""
    normalized = normalize_question(question)
    for prefix, answer in EMPTY_BOX_ANSWER_OVERRIDES.items():
        if normalized.startswith(prefix):
            return answer
    return None


def make_training_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert official MATH training rows to the shared project schema."""
    records = []
    for source_index, row in frame.iterrows():
        question = str(row["problem"]).strip()
        solution = str(row["solution"]).strip()
        answer = extract_last_boxed_answer(solution) or answer_override(question)
        if not answer:
            raise ValueError(f"Could not extract answer for source row {source_index}.")
        records.append(
            {
                "id": f"math-train-{source_index:05d}",
                "dataset": "math500",
                "task_type": "math_symbolic_answer",
                "split": "train",
                "question": question,
                "answer": answer,
                "solution": solution,
                "subject": str(row["type"]).strip(),
                "level": parse_level(str(row["level"])),
                "source_id": f"train/{source_index:05d}",
            }
        )
    return records


def remove_duplicate_questions(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the first record for each normalized question and report removals."""
    unique_records = []
    removed_records = []
    seen_questions: set[str] = set()
    for record in records:
        question_key = normalize_question(record["question"])
        if question_key in seen_questions:
            removed_records.append(record)
            continue
        seen_questions.add(question_key)
        unique_records.append(record)
    return unique_records, removed_records


def validation_group_sizes(
    groups: dict[tuple[str, int | None], list[dict[str, Any]]],
    validation_size: int,
) -> dict[tuple[str, int | None], int]:
    """Allocate an exact validation size with proportional largest remainders."""
    total = sum(len(records) for records in groups.values())
    exact_quotas = {
        key: len(records) * validation_size / total for key, records in groups.items()
    }
    sizes = {key: math.floor(quota) for key, quota in exact_quotas.items()}
    remaining = validation_size - sum(sizes.values())
    ranked_keys = sorted(
        groups,
        key=lambda key: (exact_quotas[key] - sizes[key], str(key)),
        reverse=True,
    )
    for key in ranked_keys[:remaining]:
        sizes[key] += 1
    return sizes


def stratified_split(
    records: list[dict[str, Any]], validation_size: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records deterministically while preserving subject and level ratios."""
    groups: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["subject"], record["level"])].append(record)

    group_sizes = validation_group_sizes(groups, validation_size)
    random_generator = random.Random(seed)
    training_records = []
    validation_records = []
    for key in sorted(groups, key=str):
        group_records = list(groups[key])
        random_generator.shuffle(group_records)
        group_validation_size = group_sizes[key]
        validation_records.extend(group_records[:group_validation_size])
        training_records.extend(group_records[group_validation_size:])

    for record in validation_records:
        record["split"] = "validation"
    training_records.sort(key=lambda record: record["id"])
    validation_records.sort(key=lambda record: record["id"])
    return training_records, validation_records


def assign_validation_folds(
    records: list[dict[str, Any]], fold_count: int, seed: int
) -> list[list[dict[str, Any]]]:
    """Assign equal validation folds while balancing subject and level."""
    if len(records) % fold_count != 0:
        raise ValueError("Validation records must divide evenly across folds.")

    groups: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["subject"], record["level"])].append(record)

    random_generator = random.Random(seed)
    folds: list[list[dict[str, Any]]] = [[] for _ in range(fold_count)]
    next_tie_fold = 0
    for key in sorted(groups, key=str):
        group_records = list(groups[key])
        random_generator.shuffle(group_records)
        base_size, remainder = divmod(len(group_records), fold_count)
        group_fold_sizes = [base_size] * fold_count
        ranked_folds = sorted(
            range(fold_count),
            key=lambda index: (
                len(folds[index]),
                (index - next_tie_fold) % fold_count,
            ),
        )
        for fold_index in ranked_folds[:remainder]:
            group_fold_sizes[fold_index] += 1
        next_tie_fold = (next_tie_fold + remainder) % fold_count

        offset = 0
        for fold_index, fold_size in enumerate(group_fold_sizes):
            selected = group_records[offset : offset + fold_size]
            for record in selected:
                record["validation_fold"] = fold_index + 1
            folds[fold_index].extend(selected)
            offset += fold_size

    expected_fold_size = len(records) // fold_count
    if any(len(fold) != expected_fold_size for fold in folds):
        raise ValueError("Could not create equal stratified validation folds.")
    for fold in folds:
        fold.sort(key=lambda record: record["id"])
    return folds


def distribution(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    """Count records by a named metadata field using JSON-friendly keys."""
    counts = Counter("unknown" if record[field] is None else str(record[field]) for record in records)
    return dict(sorted(counts.items()))


def validate_splits(
    training_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> dict[str, int]:
    """Validate required fields, uniqueness, split isolation, and test leakage."""
    split_records = {
        "train": training_records,
        "validation": validation_records,
        "test": test_records,
    }
    question_sets = {}
    for split, records in split_records.items():
        assert all(record["split"] == split for record in records)
        assert len({record["id"] for record in records}) == len(records)
        assert all(record["question"] and record["answer"] for record in records)
        question_keys = {normalize_question(record["question"]) for record in records}
        assert len(question_keys) == len(records)
        question_sets[split] = question_keys

    assert question_sets["train"].isdisjoint(question_sets["validation"])
    assert question_sets["train"].isdisjoint(question_sets["test"])
    assert question_sets["validation"].isdisjoint(question_sets["test"])
    expected_fold_size = len(validation_records) // VALIDATION_FOLD_COUNT
    validation_fold_counts = Counter(
        record.get("validation_fold") for record in validation_records
    )
    assert validation_fold_counts == Counter(
        {fold: expected_fold_size for fold in range(1, VALIDATION_FOLD_COUNT + 1)}
    )
    assert all("validation_fold" not in record for record in training_records)
    assert all("validation_fold" not in record for record in test_records)
    return {split: len(records) for split, records in split_records.items()}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as one JSON object per line."""
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    """Write a JSON value with readable indentation."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Prepare MATH train/validation files and update MATH-500 metadata."""
    repo_root = find_repo_root()
    raw_train_path = repo_root / "data/math500/original/math_train.parquet"
    processed_dir = repo_root / "data/processed/math500"
    test_path = processed_dir / "test.jsonl"

    raw_sha256 = file_sha256(raw_train_path)
    if raw_sha256 != EXPECTED_SHA256:
        raise ValueError(
            f"Unexpected MATH training checksum: {raw_sha256}; expected {EXPECTED_SHA256}."
        )

    raw_frame = pd.read_parquet(raw_train_path)
    if len(raw_frame) != 7500:
        raise ValueError(f"Expected 7,500 raw rows, found {len(raw_frame):,}.")

    all_training_records = make_training_records(raw_frame)
    unique_records, duplicate_records = remove_duplicate_questions(all_training_records)
    training_records, validation_records = stratified_split(
        unique_records, VALIDATION_SIZE, SPLIT_SEED
    )
    validation_folds = assign_validation_folds(
        validation_records, VALIDATION_FOLD_COUNT, SPLIT_SEED
    )
    test_records = read_jsonl(test_path)
    split_sizes = validate_splits(training_records, validation_records, test_records)

    processed_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(processed_dir / "train.jsonl", training_records)
    write_jsonl(processed_dir / "validation.jsonl", validation_records)
    for fold_index, fold_records in enumerate(validation_folds, start=1):
        write_jsonl(
            processed_dir / f"validation_fold_{fold_index}.jsonl",
            fold_records,
        )
    write_json(
        processed_dir / "all.json",
        training_records + validation_records + test_records,
    )

    test_info = json.loads((processed_dir / "dataset_info.json").read_text())
    if "sources" in test_info:
        test_source = test_info["sources"]["test"]
    else:
        test_source = test_info["source"]
    write_json(
        processed_dir / "dataset_info.json",
        {
            "dataset": "math500",
            "task_type": "math_symbolic_answer",
            "splits": split_sizes,
            "split_subject_counts": {
                "train": distribution(training_records, "subject"),
                "validation": distribution(validation_records, "subject"),
                "test": distribution(test_records, "subject"),
            },
            "split_level_counts": {
                "train": distribution(training_records, "level"),
                "validation": distribution(validation_records, "level"),
                "test": distribution(test_records, "level"),
            },
            "preparation": {
                "validation_size": VALIDATION_SIZE,
                "validation_fold_count": VALIDATION_FOLD_COUNT,
                "validation_fold_size": VALIDATION_SIZE // VALIDATION_FOLD_COUNT,
                "split_seed": SPLIT_SEED,
                "stratified_by": ["subject", "level"],
                "raw_training_rows": len(all_training_records),
                "duplicate_training_rows_removed": len(duplicate_records),
                "answer_overrides": len(EMPTY_BOX_ANSWER_OVERRIDES),
                "normalized_question_overlap_between_splits": 0,
            },
            "sources": {
                "training_and_validation": {
                    "dataset": SOURCE_DATASET,
                    "revision": SOURCE_REVISION,
                    "url": SOURCE_URL,
                    "raw_file": str(raw_train_path.relative_to(repo_root)),
                    "sha256": raw_sha256,
                },
                "test": test_source,
            },
            "files": [
                "train.jsonl",
                "validation.jsonl",
                "validation_fold_1.jsonl",
                "validation_fold_2.jsonl",
                "validation_fold_3.jsonl",
                "test.jsonl",
                "all.json",
            ],
        },
    )

    print(f"Raw MATH training rows: {len(all_training_records):,}")
    print(f"Duplicate rows removed: {len(duplicate_records):,}")
    print(f"Processed training rows: {len(training_records):,}")
    print(f"Processed validation rows: {len(validation_records):,}")
    print(f"MATH-500 test rows: {len(test_records):,}")
    print("Normalized-question overlap between splits: 0")


if __name__ == "__main__":
    main()
