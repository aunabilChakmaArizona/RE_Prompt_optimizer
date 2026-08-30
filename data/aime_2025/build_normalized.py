#!/usr/bin/env python3
"""Build one validated AIME 2025 JSONL file from the upstream exam files."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXAMS = (
    ("AIME I", ROOT / "aime2025-I.jsonl"),
    ("AIME II", ROOT / "aime2025-II.jsonl"),
)
OUTPUT = ROOT / "aime2025.jsonl"

# The upstream value includes the unit for an arc measured in degrees. AIME
# answers are integers, and independent AIME 2025 sources give answer 336.
KNOWN_ANSWER_NORMALIZATIONS = {
    ("AIME II", 5, r"336^\circ"): "336",
}


def normalize_answer(exam: str, problem_number: int, answer: object) -> str:
    value = str(answer).strip()
    value = KNOWN_ANSWER_NORMALIZATIONS.get(
        (exam, problem_number, value), value
    )
    if not re.fullmatch(r"\d{1,3}", value):
        raise ValueError(
            f"Unexpected answer for {exam} problem {problem_number}: {answer!r}"
        )
    return value


def main() -> None:
    records: list[dict[str, object]] = []
    for exam, path in EXAMS:
        with path.open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
        if len(rows) != 15:
            raise ValueError(f"Expected 15 problems in {path}, found {len(rows)}")

        exam_slug = exam.lower().replace(" ", "-")
        for problem_number, row in enumerate(rows, start=1):
            if set(row) != {"question", "answer"}:
                raise ValueError(f"Unexpected fields in {path}: {set(row)}")
            question = row["question"]
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Missing question in {exam} problem {problem_number}"
                )
            records.append(
                {
                    "id": f"aime-2025-{exam_slug}-{problem_number:02d}",
                    "year": 2025,
                    "exam": exam,
                    "problem_number": problem_number,
                    "split": "test",
                    "question": question,
                    "answer": normalize_answer(
                        exam, problem_number, row["answer"]
                    ),
                }
            )

    if len({record["question"] for record in records}) != 30:
        raise ValueError("Questions are not unique across the two exams")

    OUTPUT.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} validated records to {OUTPUT}")


if __name__ == "__main__":
    main()
