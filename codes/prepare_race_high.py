"""Download and normalize the official labeled RACE high-school test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "2fec9fd81f1dc971569a9b729c43f2f0e6436637"
SOURCE_URL = (
    "https://huggingface.co/datasets/ehovy/race/resolve/"
    f"{SOURCE_REVISION}/high/test-00000-of-00001.parquet"
)
RAW_PATH = REPO_ROOT / "data" / "race_high" / "original" / "test.parquet"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "race_high"
EXPECTED_TEST_SIZE = 3_498
OPTION_LABELS = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    """Read RACE-H preparation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download the source again and replace processed files.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path, overwrite: bool) -> None:
    """Download one source file unless it already exists."""
    if destination.exists() and not overwrite:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as stream:
        shutil.copyfileobj(response, stream)


def file_sha256(path: Path) -> str:
    """Calculate the SHA-256 checksum of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Write records as one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one readable JSON metadata file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def make_choices(options: Sequence[Any]) -> list[dict[str, str]]:
    """Pair the four RACE answer texts with labels A through D."""
    if len(options) != len(OPTION_LABELS):
        raise ValueError(f"Expected four RACE options, found {len(options)}.")
    return [
        {"label": label, "text": str(option).strip()}
        for label, option in zip(OPTION_LABELS, options)
    ]


def make_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert source rows into the shared passage-based MCQ schema."""
    records = []
    question_number_by_passage: Counter[str] = Counter()
    for row in frame.to_dict(orient="records"):
        source_passage_id = str(row["example_id"]).strip()
        question_number_by_passage[source_passage_id] += 1
        question_number = question_number_by_passage[source_passage_id]
        choices = make_choices(row["options"])
        answer = str(row["answer"]).strip().upper()
        answer_text = next(
            choice["text"] for choice in choices if choice["label"] == answer
        )
        records.append(
            {
                "id": f"{source_passage_id}#question-{question_number}",
                "dataset": "race_high",
                "task_type": "context_multiple_choice_qa",
                "split": "test",
                "context": str(row["article"]).strip(),
                "question": str(row["question"]).strip(),
                "choices": choices,
                "answer": answer,
                "answer_text": answer_text,
                "source_passage_id": source_passage_id,
                "source_dataset": "ehovy/race",
                "source_config": "high",
                "source_split": "test",
            }
        )
    return records


def validate_records(records: Sequence[dict[str, Any]]) -> None:
    """Check RACE-H size, identifiers, passages, choices, and answers."""
    if len(records) != EXPECTED_TEST_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SIZE} RACE-H test records, found {len(records)}."
        )
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("RACE-H test records contain duplicate identifiers.")
    for record in records:
        if not record["context"] or not record["question"]:
            raise ValueError(f"RACE-H record {record['id']} has empty text.")
        labels = [choice["label"] for choice in record["choices"]]
        if labels != list(OPTION_LABELS):
            raise ValueError(f"RACE-H record {record['id']} has invalid choices.")
        if any(not choice["text"] for choice in record["choices"]):
            raise ValueError(f"RACE-H record {record['id']} has an empty choice.")
        if record["answer"] not in OPTION_LABELS:
            raise ValueError(f"RACE-H record {record['id']} has an invalid answer.")


def duplicate_question_count(records: Sequence[dict[str, Any]]) -> int:
    """Count repeated context-question pairs after basic normalization."""
    normalized = [
        (
            " ".join(record["context"].casefold().split()),
            " ".join(record["question"].casefold().split()),
        )
        for record in records
    ]
    return len(normalized) - len(set(normalized))


def prepare_race_high(overwrite: bool) -> dict[str, Any]:
    """Download RACE-H and create its processed local-test files."""
    download_file(SOURCE_URL, RAW_PATH, overwrite)
    frame = pd.read_parquet(RAW_PATH)
    records = make_records(frame)
    validate_records(records)
    summary = {
        "dataset": "RACE",
        "source_dataset": "ehovy/race",
        "source_revision": SOURCE_REVISION,
        "source_config": "high",
        "source_split": "test",
        "usage": "local_test",
        "task_type": "context_multiple_choice_qa",
        "primary_metric": "accuracy",
        "test_size": len(records),
        "label_counts": dict(
            sorted(Counter(record["answer"] for record in records).items())
        ),
        "duplicate_context_question_pairs": duplicate_question_count(records),
        "raw_file": {
            "path": str(RAW_PATH.relative_to(REPO_ROOT)),
            "sha256": file_sha256(RAW_PATH),
            "url": SOURCE_URL,
        },
        "processed_file": str((OUTPUT_DIR / "test.jsonl").relative_to(REPO_ROOT)),
    }
    write_jsonl(OUTPUT_DIR / "test.jsonl", records)
    write_json(OUTPUT_DIR / "dataset_info.json", summary)
    return summary


def main() -> None:
    """Prepare RACE-H and print the resulting dataset summary."""
    args = parse_args()
    summary = prepare_race_high(args.overwrite)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
