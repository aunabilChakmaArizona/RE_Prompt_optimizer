"""Prompt rendering and answer extraction shared by math inference and optimization."""

from __future__ import annotations

import re
from typing import Any, Sequence

from math_grading.graders import normalize_numeric_answer


DEFAULT_INSTRUCTION_PROMPT = (
    "Solve the following math problem. Think step by step carefully before answering."
)
ANSWER_INSTRUCTION_PROMPT = (
    "After you finish reasoning, output the final answer exactly once between the "
    "tags <answer> and </answer>. Put only the final answer inside the tags, using "
    "LaTeX notation when needed."
)
ANSWER_PATTERN = re.compile(
    r"<answer\s*>(.*?)</answer\s*>",
    re.IGNORECASE | re.DOTALL,
)


def build_math_prompt(instruction_prompt: str, question: str) -> str:
    """Place the editable instruction, fixed answer instruction, and question in order."""
    return (
        f"{instruction_prompt.strip()}\n\n"
        f"{ANSWER_INSTRUCTION_PROMPT}\n\n"
        f"Question:\n{question.strip()}"
    )


def extract_tagged_answer(response: str) -> str | None:
    """Extract the last complete answer enclosed by answer tags."""
    matches = ANSWER_PATTERN.findall(response)
    if not matches:
        return None
    return matches[-1].strip()


def extract_last_boxed_answer(response: str) -> str | None:
    """Extract the content of the last balanced LaTeX boxed expression."""
    box_starts = list(re.finditer(r"\\boxed\s*\{", response))
    for box_start in reversed(box_starts):
        opening_brace = response.find("{", box_start.start())
        depth = 0
        for index in range(opening_brace, len(response)):
            character = response[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    answer = response[opening_brace + 1 : index].strip()
                    if answer:
                        return answer
                    break
    return None


def extract_math_answer(response: str) -> tuple[str | None, str]:
    """Use an answer tag first and the last boxed answer as a fallback."""
    tagged_answer = extract_tagged_answer(response)
    if tagged_answer:
        return tagged_answer, "answer_tag"

    boxed_answer = extract_last_boxed_answer(response)
    if boxed_answer:
        return boxed_answer, "boxed_fallback"
    return None, "missing"


def validate_math_records(records: Sequence[dict[str, Any]]) -> str: 
    """Check selected records and return their shared mathematical task type."""
    if not records:
        raise ValueError("The selected math record set is empty.")

    task_types = {str(record.get("task_type", "")).strip() for record in records}
    if len(task_types) != 1 or "" in task_types:
        raise ValueError("All selected math records must have one non-empty task_type.")
    task_type = next(iter(task_types))

    for record in records:
        if not str(record.get("question", "")).strip():
            raise ValueError(f"Record {record.get('id', '<unknown>')} has no question.")
        answer = str(record.get("answer", "")).strip()
        if not answer:
            raise ValueError(f"Record {record.get('id', '<unknown>')} has no answer.")
        if task_type == "math_short_answer" and normalize_numeric_answer(answer) is None:
            raise ValueError(
                f"Record {record.get('id', '<unknown>')} has an invalid numeric answer."
            )
    return task_type
