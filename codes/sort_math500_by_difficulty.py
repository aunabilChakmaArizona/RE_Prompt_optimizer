"""Sort MATH-500 problem IDs using the dataset's official difficulty levels."""

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "processed" / "math500" / "test.jsonl"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "math500"


def parse_args():
    """Read input and output paths from the command line."""
    parser = argparse.ArgumentParser(
        description="Sort MATH-500 IDs from easy to hard using Level 1-5."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Processed MATH-500 JSONL file.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for sorted files.")
    return parser.parse_args()


def load_records(path):
    """Load all non-empty JSONL records from a file."""
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip():
                record = json.loads(line)
                record["_original_index"] = len(records)
                record["_line_number"] = line_number
                records.append(record)
    return records


def validate_records(records):
    """Check that every record has a unique ID and a valid Level 1-5 value."""
    seen_ids = set()
    for record in records:
        problem_id = record.get("id")
        level = record.get("level")
        if not problem_id:
            raise ValueError(f"Missing ID on input line {record['_line_number']}.")
        if problem_id in seen_ids:
            raise ValueError(f"Duplicate problem ID: {problem_id}")
        if not isinstance(level, int) or level not in range(1, 6):
            raise ValueError(f"Invalid difficulty level for {problem_id}: {level!r}")
        seen_ids.add(problem_id)


def sort_records(records):
    """Sort by official difficulty and preserve source order within each level."""
    return sorted(records, key=lambda record: (record["level"], record["_original_index"]))


def write_id_list(records, path):
    """Write one problem ID per line in easy-to-hard order."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(f"{record['id']}\n")


def write_difficulty_order(records, path):
    """Write the sorted IDs with difficulty and source metadata as JSONL."""
    with path.open("w", encoding="utf-8") as file:
        for rank, record in enumerate(records, start=1):
            output = {
                "rank_easy_to_hard": rank,
                "id": record["id"],
                "difficulty_level": record["level"],
                "difficulty_label": f"Level {record['level']}",
                "subject": record.get("subject"),
                "original_index": record["_original_index"],
            }
            file.write(json.dumps(output, ensure_ascii=False) + "\n")


def write_summary(records, path):
    """Write total and per-level problem counts as JSON."""
    level_counts = Counter(record["level"] for record in records)
    summary = {
        "total_problems": len(records),
        "sorting_rule": "Official MATH difficulty level ascending; original order within each level.",
        "counts_by_level": {f"Level {level}": level_counts[level] for level in range(1, 6)},
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main():
    """Create easy-to-hard MATH-500 ID and metadata files."""
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    validate_records(records)
    sorted_records = sort_records(records)

    write_id_list(sorted_records, output_dir / "ids_easy_to_hard.txt")
    write_difficulty_order(sorted_records, output_dir / "difficulty_order.jsonl")
    write_summary(sorted_records, output_dir / "difficulty_summary.json")
    print(f"Sorted {len(sorted_records)} MATH-500 problems from Level 1 to Level 5.")
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
