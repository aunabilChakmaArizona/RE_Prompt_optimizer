#!/usr/bin/env python3
"""Write the consolidated changed-only prompt edit statistics table."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from gradpo_random_results_comparison import format_comparison, load_results_grid


METHODS = ("lpo", "gradpo-gen", "gradpo-prob", "greater", "greater-tg")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def model_from_code(code: str) -> str:
    if "_qwen_" in code:
        return "qwen"
    if "_gemma_" in code:
        return "gemma"
    raise ValueError(f"Cannot identify model from code: {code}")


def format_table(
    details: list[dict[str, str]],
    lpo_audit: list[dict[str, str]],
    gradpo_audit: list[dict[str, str]],
    *,
    model: str | None,
) -> list[str]:
    changed = defaultdict(list)
    for row in details:
        if row["selected_change"] != "True":
            continue
        if model is not None and row["model"] != model:
            continue
        changed[row["second_stage"]].append(row)

    def include_code(code: str) -> bool:
        return model is None or model_from_code(code) == model

    lpo_spans = sum(
        int(row["spans_changed"])
        for row in lpo_audit
        if row["method"] == "lpo" and include_code(row["code"])
    )
    gradpo_spans = {
        method: sum(
            int(row["changed_spans"])
            for row in gradpo_audit
            if row["method"] == method and include_code(row["code"])
        )
        for method in ("gradpo-gen", "gradpo-prob")
    }
    span_totals = {
        "lpo": lpo_spans,
        **gradpo_spans,
        "greater": len(changed["greater"]),
        "greater-tg": len(changed["greater-tg"]),
    }

    headers = (
        "method",
        "changed",
        "char_total",
        "char_avg",
        "word_total",
        "word_avg",
        "token_total",
        "token_avg",
        "span_total",
        "span_avg",
    )
    rows = []
    for method in METHODS:
        items = changed[method]
        n = len(items)
        char_total = sum(int(row["char_distance"]) for row in items)
        word_total = sum(int(row["word_distance"]) for row in items)
        token_total = sum(int(row["token_distance"]) for row in items)
        span_total = span_totals[method]
        rows.append(
            (
                method,
                str(n),
                str(char_total),
                f"{char_total / n:.2f}",
                str(word_total),
                f"{word_total / n:.2f}",
                str(token_total),
                f"{token_total / n:.2f}",
                str(span_total),
                f"{span_total / n:.2f}",
            )
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    table = [
        "  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    table.extend(
        "  ".join(value.ljust(widths[i]) for i, value in enumerate(row))
        for row in rows
    )
    return table


def main() -> None:
    details = read_csv(Path("prompt_edit_distance_details.csv"))
    lpo_audit = read_csv(Path("lpo_greater_changed_source_audit.csv"))
    gradpo_audit = read_csv(Path("gradpo_changed_span_audit.csv"))

    random_comparison = load_results_grid(
        Path("random_mode_results.txt"),
        Path("all_the_prompts.txt"),
        Path("codes/final_check_results2.ipynb"),
    )

    combined_table = format_table(details, lpo_audit, gradpo_audit, model=None)
    qwen_table = format_table(details, lpo_audit, gradpo_audit, model="qwen")
    gemma_table = format_table(details, lpo_audit, gradpo_audit, model="gemma")

    gradpo_not_selected = [
        row["code"] for row in gradpo_audit if row["selected_by_optimizer"] == "False"
    ]
    greater_not_selected = [
        row["code"]
        for row in lpo_audit
        if row["method"].startswith("greater")
        and row["accepted_by_optimizer"] == "False"
    ]
    report = "\n".join(
        [
            "CHANGED-ONLY SECOND-STAGE PROMPT EDIT STATISTICS",
            "",
            "Changed = a non-empty second-stage prompt in all_the_prompts.txt.",
            "Averages use only changed prompts for that method.",
            "Character/word/token values are unit-cost Levenshtein distances.",
            "Token distances use the target model tokenizer without special tokens.",
            "LPO spans come from source-log location counts; GradPO spans come from",
            "matching beam nodes' num_changed_spans; GreaTer changes one location.",
            "",
            "COMBINED (QWEN + GEMMA)",
            *combined_table,
            "",
            "QWEN ONLY",
            *qwen_table,
            "",
            "GEMMA ONLY",
            *gemma_table,
            "",
            "Stored nonempty GradPO candidates that were not the optimizer's final selection:",
            *(f"  - {code}" for code in gradpo_not_selected),
            "",
            "Stored nonempty GreaTer candidate not accepted by its optimizer:",
            *(f"  - {code}" for code in greater_not_selected),
            "",
            "These flagged records remain included because this report follows the",
            "file-defined changed set requested above.",
            "",
            *format_comparison(random_comparison),
        ]
    )
    output = Path("changed_prompt_stats.txt")
    output.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
