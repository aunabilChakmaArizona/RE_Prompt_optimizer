"""Prepare the labeled FOLIO v2 validation split for local test inference."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "data" / "folio" / "original" / "folio_v2_validation.jsonl"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "folio"
LABEL_TO_OPTION = {"True": "A", "False": "B", "Uncertain": "C"}
CHOICES = [
    {"label": "A", "text": "True"},
    {"label": "B", "text": "False"},
    {"label": "C", "text": "Uncertain"},
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON objects from a JSONL file."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


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


def split_premises(value: str) -> list[str]:
    """Split the source premise block into clean non-empty premises."""
    return [line.strip() for line in value.splitlines() if line.strip()]


def build_question(premises: list[str], conclusion: str) -> str:
    """Format premises and a conclusion as one logical-inference question."""
    numbered_premises = "\n".join(
        f"{index}. {premise}" for index, premise in enumerate(premises, start=1)
    )
    return f"Premises:\n{numbered_premises}\n\nConclusion:\n{conclusion.strip()}"


def make_folio_records(source_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert FOLIO rows to the shared multiple-choice QA schema."""
    records = []
    for source_record in source_records:
        label = str(source_record["label"]).strip()
        if label not in LABEL_TO_OPTION:
            raise ValueError(f"Unsupported FOLIO label: {label!r}")
        premises = split_premises(str(source_record["premises"]))
        conclusion = str(source_record["conclusion"]).strip()
        example_id = int(source_record["example_id"])
        records.append(
            {
                "id": f"folio-v2-validation-{example_id}",
                "dataset": "folio",
                "task_type": "multiple_choice_qa",
                "split": "test",
                "question": build_question(premises, conclusion),
                "choices": CHOICES,
                "answer": LABEL_TO_OPTION[label],
                "answer_text": label,
                "premises": premises,
                "conclusion": conclusion,
                "story_id": int(source_record["story_id"]),
                "example_id": example_id,
                "source_split": "validation",
                "source_revision": "v2",
            }
        )
    return records


def validate_records(records: list[dict[str, Any]]) -> None:
    """Check record count, identifiers, questions, and answer labels."""
    if len(records) != 203:
        raise ValueError(f"Expected 203 FOLIO validation examples, found {len(records)}.")
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("FOLIO records contain duplicate identifiers.")
    if any(not record["question"].strip() for record in records):
        raise ValueError("FOLIO records contain an empty question.")
    if any(record["answer"] not in {"A", "B", "C"} for record in records):
        raise ValueError("FOLIO records contain an invalid answer option.")


def prepare_folio() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create the processed FOLIO local-test JSONL and its summary."""
    source_records = read_jsonl(SOURCE_PATH)
    records = make_folio_records(source_records)
    validate_records(records)
    label_counts = Counter(record["answer_text"] for record in records)
    summary = {
        "dataset": "FOLIO v2",
        "usage": "local_test",
        "source_split": "validation",
        "official_test_labels_public": False,
        "total": len(records),
        "task_type": "multiple_choice_qa",
        "labels": LABEL_TO_OPTION,
        "label_counts": dict(sorted(label_counts.items())),
        "source_path": str(SOURCE_PATH.relative_to(REPO_ROOT)),
        "processed_path": str((OUTPUT_DIR / "test.jsonl").relative_to(REPO_ROOT)),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / "test.jsonl", records)
    write_json(OUTPUT_DIR / "dataset_info.json", summary)
    return records, summary


def main() -> None:
    """Prepare FOLIO and print the resulting dataset summary."""
    _, summary = prepare_folio()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
