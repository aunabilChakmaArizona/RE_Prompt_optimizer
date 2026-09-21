"""Download and normalize all official ANLI rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from anli_task_common import (
    ANLI_CHOICES,
    ANLI_LABEL_NAMES,
    ANLI_OPTION_LABELS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = "facebook/anli"
SOURCE_CONFIG = "plain_text"
SOURCE_REVISION = "8e4813d81f46d313dac7892e1c28076917cfcdf9"
RAW_DIR = REPO_ROOT / "data" / "anli" / "original"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "anli"
ROUNDS = ("r1", "r2", "r3")
SOURCE_SPLITS = ("train", "dev", "test")
OUTPUT_SPLITS = {"train": "train", "dev": "validation", "test": "test"}
PROMPTOPT_VALIDATION_SEED = 42
PROMPTOPT_VALIDATION_FOLD_SIZE = 1_000
EXPECTED_SOURCE_SIZES = {
    "train_r1": 16_946,
    "dev_r1": 1_000,
    "test_r1": 1_000,
    "train_r2": 45_460,
    "dev_r2": 1_000,
    "test_r2": 1_000,
    "train_r3": 100_459,
    "dev_r3": 1_200,
    "test_r3": 1_200,
}
def parse_args() -> argparse.Namespace:
    """Read ANLI preparation settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download sources again and replace processed files.",
    )
    return parser.parse_args()


def source_url(source_split: str) -> str:
    """Build the pinned Hugging Face URL for one ANLI source split."""
    filename = f"{source_split}-00000-of-00001.parquet"
    return (
        f"https://huggingface.co/datasets/{SOURCE_DATASET}/resolve/"
        f"{SOURCE_REVISION}/{SOURCE_CONFIG}/{filename}"
    )


def download_file(url: str, destination: Path, overwrite: bool) -> None:
    """Download one source file unless a local copy should be reused."""
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
    """Write readable JSON metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def make_question(premise: str, hypothesis: str) -> str:
    """Format one premise-hypothesis pair as a classification question."""
    return f"Premise:\n{premise}\n\nHypothesis:\n{hypothesis}"


def make_records(
    frame: pd.DataFrame,
    source_split: str,
    output_split: str,
    round_name: str,
) -> list[dict[str, Any]]:
    """Convert one official ANLI split into the shared classification schema."""
    records = []
    for row in frame.to_dict(orient="records"):
        label_index = int(row["label"])
        if label_index not in ANLI_LABEL_NAMES:
            raise ValueError(f"Unexpected ANLI label index: {label_index}")
        premise = str(row["premise"]).strip()
        hypothesis = str(row["hypothesis"]).strip()
        label_name = ANLI_LABEL_NAMES[label_index]
        records.append(
            {
                "id": str(row["uid"]).strip(),
                "dataset": "anli",
                "task_type": "multiple_choice_qa",
                "split": output_split,
                "round": round_name.upper(),
                "premise": premise,
                "hypothesis": hypothesis,
                "question": make_question(premise, hypothesis),
                "choices": [dict(choice) for choice in ANLI_CHOICES],
                "answer": ANLI_OPTION_LABELS[label_index],
                "answer_text": label_name,
                "source_label": label_index,
                "annotator_reason": str(row.get("reason", "")).strip(),
                "source_dataset": SOURCE_DATASET,
                "source_config": SOURCE_CONFIG,
                "source_split": source_split,
            }
        )
    return records


def validate_records(
    records: Sequence[dict[str, Any]],
    output_split: str,
    expected_size: int,
) -> None:
    """Check the size, identifiers, text fields, and labels of one combined split."""
    if len(records) != expected_size:
        raise ValueError(
            f"Expected {expected_size} {output_split} records, found {len(records)}."
        )
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"ANLI {output_split} contains duplicate identifiers.")
    for record in records:
        if not record["premise"] or not record["hypothesis"]:
            raise ValueError(f"ANLI record {record['id']} has empty input text.")
        if record["answer"] not in {"A", "B", "C"}:
            raise ValueError(f"ANLI record {record['id']} has an invalid answer.")
        if record["round"] not in {"R1", "R2", "R3"}:
            raise ValueError(f"ANLI record {record['id']} has an invalid round.")


def normalized_pair(record: dict[str, Any]) -> tuple[str, str]:
    """Normalize an ANLI premise-hypothesis pair for overlap counting."""
    return (
        " ".join(record["premise"].casefold().split()),
        " ".join(record["hypothesis"].casefold().split()),
    )


def sample_label_quotas(
    records: Sequence[dict[str, Any]],
    quotas: dict[str, int],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Select deterministic label quotas while retaining source order."""
    grouped = {
        label: [record for record in records if record["answer"] == label]
        for label in ("A", "B", "C")
    }
    if set(quotas) != set(grouped):
        raise ValueError("ANLI label quotas must define A, B, and C.")
    if any(len(grouped[label]) < quotas[label] for label in grouped):
        raise ValueError("ANLI validation labels cannot satisfy stratified quotas.")
    retained_ids = {
        record["id"]
        for label in grouped
        for record in rng.sample(grouped[label], quotas[label])
    }
    return [record for record in records if record["id"] in retained_ids]


def make_promptopt_validation_folds(
    records: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Create three deterministic folds balanced across labels and ANLI rounds."""
    by_round = {
        round_name: [record for record in records if record["round"] == round_name]
        for round_name in ("R1", "R2", "R3")
    }
    if len(by_round["R1"]) != 1_000 or len(by_round["R2"]) != 1_000:
        raise ValueError("ANLI R1 and R2 validation must each contain 1,000 records.")

    selected_by_round = {
        "R1": by_round["R1"],
        "R2": by_round["R2"],
        "R3": sample_label_quotas(
            by_round["R3"],
            {"A": 332, "B": 334, "C": 334},
            random.Random(PROMPTOPT_VALIDATION_SEED + 3),
        ),
    }
    # These allocations give every fold 1,000 examples, mix all three rounds,
    # and rotate the single extra label and round example across the folds.
    fold_allocations = {
        ("R1", "A"): (112, 111, 111),
        ("R1", "B"): (111, 111, 111),
        ("R1", "C"): (111, 111, 111),
        ("R2", "A"): (111, 112, 111),
        ("R2", "B"): (111, 111, 111),
        ("R2", "C"): (111, 111, 111),
        ("R3", "A"): (111, 110, 111),
        ("R3", "B"): (111, 112, 111),
        ("R3", "C"): (111, 111, 112),
    }
    folds: list[list[dict[str, Any]]] = [[], [], []]
    for round_index, round_name in enumerate(("R1", "R2", "R3"), start=1):
        for label_index, label in enumerate(("A", "B", "C"), start=1):
            stratum = [
                record
                for record in selected_by_round[round_name]
                if record["answer"] == label
            ]
            random.Random(
                PROMPTOPT_VALIDATION_SEED + 10 * round_index + label_index
            ).shuffle(stratum)
            start = 0
            for fold_index, count in enumerate(
                fold_allocations[(round_name, label)],
                start=1,
            ):
                end = start + count
                folds[fold_index - 1].extend(
                    {**record, "validation_fold": fold_index}
                    for record in stratum[start:end]
                )
                start = end
            if start != len(stratum):
                raise ValueError(
                    f"ANLI allocation did not consume {round_name}/{label}."
                )
    for fold_index, fold in enumerate(folds, start=1):
        random.Random(PROMPTOPT_VALIDATION_SEED + 100 + fold_index).shuffle(fold)
    return folds


def validate_promptopt_validation_folds(
    folds: Sequence[Sequence[dict[str, Any]]],
) -> None:
    """Check prompt-optimization fold sizes, rounds, labels, and identifiers."""
    if len(folds) != 3:
        raise ValueError(f"Expected three ANLI validation folds, found {len(folds)}.")
    expected_label_counts = (
        {"A": 334, "B": 333, "C": 333},
        {"A": 333, "B": 334, "C": 333},
        {"A": 333, "B": 333, "C": 334},
    )
    expected_round_counts = (
        {"R1": 334, "R2": 333, "R3": 333},
        {"R1": 333, "R2": 334, "R3": 333},
        {"R1": 333, "R2": 333, "R3": 334},
    )
    all_ids = []
    for fold_index, fold in enumerate(folds, start=1):
        if len(fold) != PROMPTOPT_VALIDATION_FOLD_SIZE:
            raise ValueError(
                f"ANLI validation fold {fold_index} contains {len(fold)} records."
            )
        if {record["validation_fold"] for record in fold} != {fold_index}:
            raise ValueError(f"ANLI validation fold {fold_index} has invalid markers.")
        label_counts = dict(Counter(record["answer"] for record in fold))
        round_counts = dict(Counter(record["round"] for record in fold))
        if label_counts != expected_label_counts[fold_index - 1]:
            raise ValueError(
                f"ANLI fold {fold_index} has invalid label counts: {label_counts}."
            )
        if round_counts != expected_round_counts[fold_index - 1]:
            raise ValueError(
                f"ANLI fold {fold_index} has invalid round counts: {round_counts}."
            )
        all_ids.extend(record["id"] for record in fold)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("ANLI prompt-optimization validation folds overlap.")


def prepare_anli(overwrite: bool) -> dict[str, Any]:
    """Download all ANLI rounds and create combined train, validation, and test files."""
    combined_records = {split: [] for split in OUTPUT_SPLITS.values()}
    raw_files = []
    source_counts = {}

    for round_name in ROUNDS:
        for split_name in SOURCE_SPLITS:
            source_split = f"{split_name}_{round_name}"
            raw_path = RAW_DIR / f"{source_split}.parquet"
            url = source_url(source_split)
            download_file(url, raw_path, overwrite)
            frame = pd.read_parquet(raw_path)
            expected_size = EXPECTED_SOURCE_SIZES[source_split]
            if len(frame) != expected_size:
                raise ValueError(
                    f"Expected {expected_size} rows in {source_split}, found {len(frame)}."
                )
            output_split = OUTPUT_SPLITS[split_name]
            records = make_records(
                frame,
                source_split=source_split,
                output_split=output_split,
                round_name=round_name,
            )
            combined_records[output_split].extend(records)
            source_counts[source_split] = len(records)
            raw_files.append(
                {
                    "source_split": source_split,
                    "path": str(raw_path.relative_to(REPO_ROOT)),
                    "url": url,
                    "sha256": file_sha256(raw_path),
                    "rows": len(records),
                }
            )

    expected_combined_sizes = {
        output_split: sum(
            EXPECTED_SOURCE_SIZES[f"{source_split}_{round_name}"]
            for round_name in ROUNDS
        )
        for source_split, output_split in OUTPUT_SPLITS.items()
    }
    for output_split, records in combined_records.items():
        validate_records(
            records,
            output_split=output_split,
            expected_size=expected_combined_sizes[output_split],
        )
        write_jsonl(OUTPUT_DIR / f"{output_split}.jsonl", records)

    validation_folds = make_promptopt_validation_folds(
        combined_records["validation"]
    )
    validate_promptopt_validation_folds(validation_folds)
    promptopt_validation = [record for fold in validation_folds for record in fold]
    write_jsonl(OUTPUT_DIR / "validation_promptopt.jsonl", promptopt_validation)
    for fold_index, fold in enumerate(validation_folds, start=1):
        write_jsonl(OUTPUT_DIR / f"validation_fold_{fold_index}.jsonl", fold)

    id_sets = {
        split_name: {record["id"] for record in records}
        for split_name, records in combined_records.items()
    }
    pair_sets = {
        split_name: {normalized_pair(record) for record in records}
        for split_name, records in combined_records.items()
    }
    split_pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    summary = {
        "dataset": "ANLI",
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "source_config": SOURCE_CONFIG,
        "task_type": "three_way_natural_language_inference",
        "primary_metric": "accuracy",
        "label_mapping": {
            "A": "entailment",
            "B": "neutral",
            "C": "contradiction",
        },
        "source_split_sizes": source_counts,
        "processed_splits": {
            split_name: {
                "size": len(records),
                "round_counts": dict(
                    sorted(Counter(record["round"] for record in records).items())
                ),
                "label_counts": dict(
                    sorted(Counter(record["answer"] for record in records).items())
                ),
                "file": str(
                    (OUTPUT_DIR / f"{split_name}.jsonl").relative_to(REPO_ROOT)
                ),
            }
            for split_name, records in combined_records.items()
        },
        "prompt_optimization_validation": {
            "size": len(promptopt_validation),
            "seed": PROMPTOPT_VALIDATION_SEED,
            "fold_definition": (
                "Three equally sized folds, each balanced across labels and mixed "
                "across ANLI rounds."
            ),
            "file": str(
                (OUTPUT_DIR / "validation_promptopt.jsonl").relative_to(REPO_ROOT)
            ),
            "folds": {
                str(fold_index): {
                    "size": len(fold),
                    "round_counts": dict(
                        sorted(Counter(record["round"] for record in fold).items())
                    ),
                    "label_counts": dict(
                        sorted(Counter(record["answer"] for record in fold).items())
                    ),
                    "file": str(
                        (OUTPUT_DIR / f"validation_fold_{fold_index}.jsonl").relative_to(
                            REPO_ROOT
                        )
                    ),
                }
                for fold_index, fold in enumerate(validation_folds, start=1)
            },
        },
        "cross_split_duplicate_ids": {
            f"{left}_vs_{right}": len(id_sets[left] & id_sets[right])
            for left, right in split_pairs
        },
        "cross_split_duplicate_text_pairs": {
            f"{left}_vs_{right}": len(pair_sets[left] & pair_sets[right])
            for left, right in split_pairs
        },
        "raw_files": raw_files,
    }
    write_json(OUTPUT_DIR / "dataset_info.json", summary)
    return summary


def main() -> None:
    """Prepare ANLI and print the resulting dataset summary."""
    args = parse_args()
    summary = prepare_anli(args.overwrite)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
