"""Prepare disjoint HotpotQA train, validation, and local-test splits."""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "hotpotqa" / "original"
TRAIN_SOURCE_PATHS = [
    RAW_DIR / "train-00000-of-00002.parquet",
    RAW_DIR / "train-00001-of-00002.parquet",
]
VALIDATION_SOURCE_PATH = RAW_DIR / "validation.parquet"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "hotpotqa"
TRAIN_SIZE = 10_000
VALIDATION_SIZE = 1_500
TEST_SIZE = 1_500
VALIDATION_FOLD_COUNT = 3
SPLIT_SEED = 42


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records as one JSON object per line."""
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one readable JSON document."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def clean_context(source_context: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a columnar HotpotQA context into titled paragraph records."""
    titles = source_context["title"]
    sentence_groups = source_context["sentences"]
    if len(titles) != len(sentence_groups):
        raise ValueError("HotpotQA context titles and paragraphs have different lengths.")
    return [
        {
            "title": str(title).strip(),
            "sentences": [str(sentence).strip() for sentence in sentences],
        }
        for title, sentences in zip(titles, sentence_groups)
    ]


def clean_supporting_facts(source_facts: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert columnar supporting facts into title and sentence-index records."""
    titles = source_facts["title"]
    sentence_ids = source_facts["sent_id"]
    if len(titles) != len(sentence_ids):
        raise ValueError("HotpotQA supporting-fact columns have different lengths.")
    return [
        {"title": str(title).strip(), "sentence_id": int(sentence_id)}
        for title, sentence_id in zip(titles, sentence_ids)
    ]


def make_hotpotqa_records(
    frame: pd.DataFrame,
    *,
    split: str,
    source_split: str,
) -> list[dict[str, Any]]:
    """Convert HotpotQA rows to the shared context-open-QA schema."""
    records = []
    for row in frame.to_dict(orient="records"):
        answer = str(row["answer"]).strip()
        records.append(
            {
                "id": str(row["id"]).strip(),
                "dataset": "hotpotqa",
                "task_type": "hotpotqa_open_qa",
                "split": split,
                "question": str(row["question"]).strip(),
                "context": clean_context(row["context"]),
                "answer": answer,
                "answers": [answer],
                "question_type": str(row["type"]).strip(),
                "level": str(row["level"]).strip(),
                "supporting_facts": clean_supporting_facts(row["supporting_facts"]),
                "source_split": source_split,
                "source_config": "distractor",
            }
        )
    return records


def shuffled_indices(size: int, seed: int) -> list[int]:
    """Create one deterministic random ordering of source-row indices."""
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return indices


def sample_source_frame(
    frame: pd.DataFrame,
    size: int,
    seed: int,
) -> pd.DataFrame:
    """Draw a deterministic sample without replacement from one source frame."""
    if size > len(frame):
        raise ValueError(f"Cannot sample {size} rows from a frame of size {len(frame)}.")
    selected_indices = shuffled_indices(len(frame), seed)[:size]
    return frame.iloc[selected_indices].reset_index(drop=True)


def split_training_frame(
    frame: pd.DataFrame,
    train_size: int,
    validation_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw non-overlapping training and validation subsets from source training."""
    required_size = train_size + validation_size
    if required_size > len(frame):
        raise ValueError(
            f"Requested {required_size} rows from a training frame of size {len(frame)}."
        )
    selected_indices = shuffled_indices(len(frame), seed)[:required_size]
    training_indices = selected_indices[:train_size]
    validation_indices = selected_indices[train_size:]
    return (
        frame.iloc[training_indices].reset_index(drop=True),
        frame.iloc[validation_indices].reset_index(drop=True),
    )


def assign_validation_folds(
    records: list[dict[str, Any]],
    fold_count: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Assign equal folds balanced by difficulty, question type, and answer type."""
    if len(records) % fold_count != 0:
        raise ValueError("Validation size must divide evenly across folds.")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        answer_type = (
            "yes_no" if record["answer"].casefold() in {"yes", "no"} else "text"
        )
        key = (record["level"], record["question_type"], answer_type)
        groups[key].append(record)

    random_generator = random.Random(seed)
    folds = [[] for _ in range(fold_count)]
    running_offset = 0
    for key in sorted(groups):
        group_records = groups[key]
        random_generator.shuffle(group_records)
        for local_index, record in enumerate(group_records):
            fold_index = (running_offset + local_index) % fold_count
            record["validation_fold"] = fold_index + 1
            folds[fold_index].append(record)
        running_offset += len(group_records)
    for fold in folds:
        fold.sort(key=lambda record: record["id"])
    return folds


def validate_record_content(records: list[dict[str, Any]]) -> None:
    """Check identifiers, questions, answers, and contexts in one split."""
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("A HotpotQA split contains duplicate identifiers.")
    for record in records:
        if not record["question"] or not record["answer"]:
            raise ValueError(f"HotpotQA record {record['id']} has an empty QA field.")
        if not record["context"]:
            raise ValueError(f"HotpotQA record {record['id']} has no context.")
        if any(not paragraph["sentences"] for paragraph in record["context"]):
            raise ValueError(f"HotpotQA record {record['id']} has an empty paragraph.")


def normalized_question(record: dict[str, Any]) -> str:
    """Normalize one question for duplicate and split-overlap checks."""
    return re.sub(r"\s+", " ", record["question"]).strip().casefold()


def validate_splits(
    training_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    validation_folds: list[list[dict[str, Any]]],
) -> None:
    """Check requested sizes, source isolation, folds, and split non-overlap."""
    expected_sizes = {
        "train": (training_records, TRAIN_SIZE),
        "validation": (validation_records, VALIDATION_SIZE),
        "test": (test_records, TEST_SIZE),
    }
    for split_name, (records, expected_size) in expected_sizes.items():
        if len(records) != expected_size:
            raise ValueError(
                f"Expected {expected_size} {split_name} records, found {len(records)}."
            )
        validate_record_content(records)

    split_ids = {
        split_name: {record["id"] for record in records}
        for split_name, (records, _) in expected_sizes.items()
    }
    if split_ids["train"] & split_ids["validation"]:
        raise ValueError("Training and validation identifiers overlap.")
    if split_ids["test"] & (split_ids["train"] | split_ids["validation"]):
        raise ValueError("Test identifiers overlap with prompt-optimization data.")
    split_questions = {
        split_name: {normalized_question(record) for record in records}
        for split_name, (records, _) in expected_sizes.items()
    }
    for split_name, (records, _) in expected_sizes.items():
        if len(split_questions[split_name]) != len(records):
            raise ValueError(f"The {split_name} split contains duplicate questions.")
    if split_questions["train"] & split_questions["validation"]:
        raise ValueError("Training and validation questions overlap.")
    if split_questions["test"] & (
        split_questions["train"] | split_questions["validation"]
    ):
        raise ValueError("Test questions overlap with prompt-optimization data.")
    if any(record["source_split"] != "train" for record in training_records):
        raise ValueError("Training records must come only from source training.")
    if any(record["source_split"] != "train" for record in validation_records):
        raise ValueError("Validation records must come only from source training.")
    if any(record["source_split"] != "validation" for record in test_records):
        raise ValueError("Test records must come only from source validation.")

    expected_fold_size = VALIDATION_SIZE // VALIDATION_FOLD_COUNT
    if len(validation_folds) != VALIDATION_FOLD_COUNT:
        raise ValueError("Unexpected number of validation folds.")
    if any(len(fold) != expected_fold_size for fold in validation_folds):
        raise ValueError("Validation folds do not contain 500 examples each.")
    fold_ids = [record["id"] for fold in validation_folds for record in fold]
    if set(fold_ids) != split_ids["validation"] or len(fold_ids) != len(set(fold_ids)):
        raise ValueError("Validation folds do not partition validation exactly once.")


def split_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize question type, difficulty, answers, and context sizes."""
    context_size_counts = Counter(len(record["context"]) for record in records)
    answer_type_counts = Counter(
        "yes_no" if record["answer"].casefold() in {"yes", "no"} else "text"
        for record in records
    )
    return {
        "total": len(records),
        "source_split": records[0]["source_split"],
        "question_type_counts": dict(
            sorted(Counter(record["question_type"] for record in records).items())
        ),
        "level_counts": dict(
            sorted(Counter(record["level"] for record in records).items())
        ),
        "answer_type_counts": dict(sorted(answer_type_counts.items())),
        "context_paragraph_counts": {
            str(size): count for size, count in sorted(context_size_counts.items())
        },
    }


def prepare_hotpotqa() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Create processed HotpotQA train, validation, folds, test, and summary files."""
    source_training = pd.concat(
        [pd.read_parquet(path) for path in TRAIN_SOURCE_PATHS],
        ignore_index=True,
    )
    source_validation = pd.read_parquet(VALIDATION_SOURCE_PATH)
    training_frame, validation_frame = split_training_frame(
        source_training,
        TRAIN_SIZE,
        VALIDATION_SIZE,
        SPLIT_SEED,
    )
    test_frame = sample_source_frame(source_validation, TEST_SIZE, SPLIT_SEED)

    training_records = make_hotpotqa_records(
        training_frame,
        split="train",
        source_split="train",
    )
    validation_records = make_hotpotqa_records(
        validation_frame,
        split="validation",
        source_split="train",
    )
    test_records = make_hotpotqa_records(
        test_frame,
        split="test",
        source_split="validation",
    )
    validation_folds = assign_validation_folds(
        validation_records,
        VALIDATION_FOLD_COUNT,
        SPLIT_SEED + 1,
    )
    training_records.sort(key=lambda record: record["id"])
    validation_records.sort(key=lambda record: record["id"])
    test_records.sort(key=lambda record: record["id"])
    validate_splits(
        training_records,
        validation_records,
        test_records,
        validation_folds,
    )

    split_records = {
        "train": training_records,
        "validation": validation_records,
        "test": test_records,
    }
    summary = {
        "dataset": "HotpotQA",
        "source_config": "distractor",
        "official_test_labels_public": False,
        "split_seed": SPLIT_SEED,
        "sampling": "deterministic random sampling without replacement",
        "split_reference": "https://aclanthology.org/2024.emnlp-main.597.pdf",
        "source_sizes": {
            "train": len(source_training),
            "validation": len(source_validation),
        },
        "splits": {
            split_name: split_summary(records)
            for split_name, records in split_records.items()
        },
        "validation_folds": {
            str(index): split_summary(fold)
            for index, fold in enumerate(validation_folds, start=1)
        },
        "source_files": [
            str(path.relative_to(REPO_ROOT))
            for path in [*TRAIN_SOURCE_PATHS, VALIDATION_SOURCE_PATH]
        ],
        "processed_files": {
            "train": "data/processed/hotpotqa/train.jsonl",
            "validation": "data/processed/hotpotqa/validation.jsonl",
            "test": "data/processed/hotpotqa/test.jsonl",
        },
        "test_source_note": (
            "The 1,500-example final test set is sampled only from the official "
            "validation split. The remaining source-validation examples are unused."
        ),
        "reported_metrics": ["answer_exact_match", "answer_token_f1"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, records in split_records.items():
        write_jsonl(OUTPUT_DIR / f"{split_name}.jsonl", records)
    for index, fold in enumerate(validation_folds, start=1):
        write_jsonl(OUTPUT_DIR / f"validation_fold_{index}.jsonl", fold)
    write_json(OUTPUT_DIR / "dataset_info.json", summary)
    return split_records, summary


def main() -> None:
    """Prepare HotpotQA and print the resulting dataset summary."""
    _, summary = prepare_hotpotqa()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
