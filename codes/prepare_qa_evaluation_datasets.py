"""Download and normalize WebQuestions and CommonsenseQA evaluation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
WEBQUESTIONS_REVISION = "0e473cbe21d1e91ec18da343644498be6a3f5454"
COMMONSENSEQA_REVISION = "94630fe30dad47192a8546eb75f094926d47e155"
WEBQUESTIONS_TEST_URL = (
    "https://huggingface.co/datasets/stanfordnlp/web_questions/resolve/"
    f"{WEBQUESTIONS_REVISION}/data/test-00000-of-00001.parquet"
)
COMMONSENSEQA_URLS = {
    split: (
        "https://huggingface.co/datasets/tau/commonsense_qa/resolve/"
        f"{COMMONSENSEQA_REVISION}/data/{split}-00000-of-00001.parquet"
    )
    for split in ("validation", "test")
}


def parse_args() -> argparse.Namespace:
    """Read dataset preparation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", "webquestions", "commonsenseqa"),
        default="all",
        help="Dataset to prepare.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download raw files again and replace processed files.",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path, overwrite: bool) -> None:
    """Download one source file unless it is already available."""
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
    """Write a readable JSON metadata file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def duplicate_question_count(records: Sequence[dict[str, Any]]) -> int:
    """Count repeated questions after simple case and whitespace normalization."""
    normalized = [" ".join(record["question"].lower().split()) for record in records]
    return len(normalized) - len(set(normalized))


def prepare_webquestions(overwrite: bool) -> None:
    """Prepare the labeled WebQuestions test split for open-answer evaluation."""
    raw_path = REPO_ROOT / "data/webquestions/original/test.parquet"
    output_dir = REPO_ROOT / "data/processed/webquestions"
    download_file(WEBQUESTIONS_TEST_URL, raw_path, overwrite)
    frame = pd.read_parquet(raw_path)
    records = []
    for index, row in frame.iterrows():
        answers = [str(answer).strip() for answer in row["answers"] if str(answer).strip()]
        records.append(
            {
                "id": f"webquestions-test-{index:04d}",
                "dataset": "webquestions",
                "split": "test",
                "task_type": "open_qa",
                "question": str(row["question"]).strip(),
                "answers": answers,
                "source_url": str(row["url"]).strip(),
            }
        )
    write_jsonl(output_dir / "test.jsonl", records)
    write_json(
        output_dir / "dataset_info.json",
        {
            "dataset": "stanfordnlp/web_questions",
            "revision": WEBQUESTIONS_REVISION,
            "task_type": "open_qa",
            "primary_metric": "official_macro_answer_set_f1",
            "secondary_metric": "normalized_macro_answer_set_f1",
            "splits": {"test": len(records)},
            "duplicate_questions": duplicate_question_count(records),
            "raw_files": {
                str(raw_path.relative_to(REPO_ROOT)): file_sha256(raw_path),
            },
        },
    )
    print(f"Prepared WebQuestions test: {len(records)} examples")


def commonsenseqa_choices(raw_choices: dict[str, Any]) -> list[dict[str, str]]:
    """Convert parallel CommonsenseQA label and text arrays into choice objects."""
    labels = list(raw_choices["label"])
    texts = list(raw_choices["text"])
    return [
        {"label": str(label).strip().upper(), "text": str(text).strip()}
        for label, text in zip(labels, texts)
    ]


def commonsenseqa_records(frame: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    """Normalize one CommonsenseQA source split."""
    records = []
    for _, row in frame.iterrows():
        choices = commonsenseqa_choices(row["choices"])
        answer = str(row["answerKey"]).strip().upper()
        answer_text = next(
            (choice["text"] for choice in choices if choice["label"] == answer),
            None,
        )
        records.append(
            {
                "id": str(row["id"]).strip(),
                "dataset": "commonsenseqa",
                "split": split,
                "task_type": "multiple_choice_qa",
                "question": str(row["question"]).strip(),
                "choices": choices,
                "answer": answer or None,
                "answer_text": answer_text,
                "question_concept": str(row["question_concept"]).strip(),
            }
        )
    return records


def prepare_commonsenseqa(overwrite: bool) -> None:
    """Prepare labeled and hidden-label CommonsenseQA evaluation splits."""
    raw_dir = REPO_ROOT / "data/commonsenseqa/original"
    output_dir = REPO_ROOT / "data/processed/commonsenseqa"
    raw_paths = {
        split: raw_dir / f"{split}.parquet" for split in COMMONSENSEQA_URLS
    }
    for split, url in COMMONSENSEQA_URLS.items():
        download_file(url, raw_paths[split], overwrite)
    validation_records = commonsenseqa_records(
        pd.read_parquet(raw_paths["validation"]),
        "validation",
    )
    test_records = commonsenseqa_records(
        pd.read_parquet(raw_paths["test"]),
        "test",
    )
    write_jsonl(output_dir / "validation.jsonl", validation_records)
    write_jsonl(output_dir / "local_test.jsonl", validation_records)
    write_jsonl(output_dir / "test_unlabeled.jsonl", test_records)
    write_json(
        output_dir / "dataset_info.json",
        {
            "dataset": "tau/commonsense_qa",
            "revision": COMMONSENSEQA_REVISION,
            "task_type": "multiple_choice_qa",
            "primary_metric": "accuracy",
            "splits": {
                "validation": len(validation_records),
                "local_test": len(validation_records),
                "test_unlabeled": len(test_records),
            },
            "notes": {
                "local_test": "A byte-for-byte record copy of the labeled official validation split for local scoring.",
                "test_unlabeled": "The official test split has empty answerKey values and cannot be scored locally.",
            },
            "duplicate_questions": {
                "validation": duplicate_question_count(validation_records),
                "test": duplicate_question_count(test_records),
            },
            "raw_files": {
                str(path.relative_to(REPO_ROOT)): file_sha256(path)
                for path in raw_paths.values()
            },
        },
    )
    print(
        "Prepared CommonsenseQA: "
        f"{len(validation_records)} labeled validation/local-test examples and "
        f"{len(test_records)} unlabeled official-test examples"
    )


def main() -> None:
    """Prepare the requested evaluation datasets."""
    args = parse_args()
    if args.dataset in ("all", "webquestions"):
        prepare_webquestions(args.overwrite)
    if args.dataset in ("all", "commonsenseqa"):
        prepare_commonsenseqa(args.overwrite)


if __name__ == "__main__":
    main()
