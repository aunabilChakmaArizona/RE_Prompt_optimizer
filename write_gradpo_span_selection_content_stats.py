#!/usr/bin/env python3
"""Audit the content of top-gradient spans selected in all GradPO-Gen runs."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from gradpo_span_audit import parse_records, resolve_log
from prompt_edit_distance_stats import FIRST_STAGE_RE


PROMPTS_PATH = Path("all_the_prompts.txt")
OUTPUT_PATH = Path("gradpo_span_selection_content_stats.txt")
GROUPED_OUTPUT_PATH = Path("gradpo_selected_spans_four_categories.txt")
SIMPLE_GROUPED_OUTPUT_PATH = Path(
    "gradpo_selected_spans_four_categories_simple.txt"
)

CATEGORY_ORDER = (
    "Shared task-content wording",
    "Generic instruction fragments",
    "Decision/output/formatting wording",
    "Relation-specific rule fragments",
)

ENTITY_DECISION_SPANS = {
    "holds between the",
    "subject and",
    "entities in",
    "marked subject",
    "explicitly expresses",
    "subject",
    "instantiated between",
}
ANSWER_DECISION_SPANS = {"answer", "only answer", "determine if", "decide"}
GENERIC_SPANS = {
    "you are",
    "the",
    "of",
    "you",
    "are given",
    "a",
    "to",
    "given a",
    "itness",
}
OTHER_TASK_SPANS = {
    "instance",
    "exemplifying",
    "specified",
    "on the information",
    "exemplifies",
}
RELATION_SPECIFIC_SPANS = {"province", "hier"}


@dataclass(frozen=True)
class RunSpans:
    code: str
    model: str
    improved_on_dev: bool
    spans: tuple[str, ...]
    source_log: str


def norm(text: str) -> str:
    return " ".join(str(text).split())


def extract_tagged_prompt_blocks(log_text: str) -> list[tuple[str, tuple[str, ...]]]:
    blocks: list[tuple[str, tuple[str, ...]]] = []
    pattern = re.compile(
        r"(?:Input Prompt with editable spans:|"
        r"current instruction prompt with targeted spans below:)"
        r"\s*```(?:\w+)?\s*\n?(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    for match in pattern.finditer(log_text):
        tagged = match.group(1).strip()
        clean_prompt = re.sub(r"</?span_\d+>", "", tagged)
        spans = tuple(
            norm(span_match.group(2))
            for span_match in re.finditer(
                r"<span_(\d+)>(.*?)</span_\1>", tagged, re.DOTALL
            )
        )
        if spans:
            blocks.append((clean_prompt, spans))
    return blocks


def classify_span(span: str) -> str:
    value = span.casefold()
    if "sentence" in value:
        return "Shared task-content wording"
    if value in {"name", "description", "description of"}:
        return "Shared task-content wording"
    if value in ENTITY_DECISION_SPANS:
        return "Shared task-content wording"
    if value in GENERIC_SPANS or value in OTHER_TASK_SPANS:
        return "Generic instruction fragments"
    if value in ANSWER_DECISION_SPANS or value == "brackets":
        return "Decision/output/formatting wording"
    if value in RELATION_SPECIFIC_SPANS:
        return "Relation-specific rule fragments"
    raise ValueError(f"Unclassified selected span: {span!r}")


def collect_runs() -> list[RunSpans]:
    root = Path.cwd().resolve()
    records = parse_records(PROMPTS_PATH)
    valid_bases = sorted(
        code
        for code, record in records.items()
        if FIRST_STAGE_RE.fullmatch(code) and record.prompt
    )
    if len(valid_bases) != 19:
        raise ValueError(f"Expected 19 valid first-stage prompts; found {len(valid_bases)}")

    runs: list[RunSpans] = []
    for base_code in valid_bases:
        gradpo_code = f"{base_code}_gradpo-gen"
        gradpo_record = records[gradpo_code]
        log_path = resolve_log(root, gradpo_record.source)
        if log_path is None:
            raise FileNotFoundError(
                f"Could not resolve source log for {gradpo_code}: {gradpo_record.source}"
            )
        blocks = extract_tagged_prompt_blocks(
            log_path.read_text(encoding="utf-8", errors="replace")
        )
        if not blocks:
            raise ValueError(f"No tagged span blocks found for {gradpo_code}")

        base_prompt = records[base_code].prompt
        ranked_blocks = [
            (
                SequenceMatcher(
                    None,
                    norm(base_prompt),
                    norm(clean_prompt),
                ).ratio(),
                spans,
            )
            for clean_prompt, spans in blocks
        ]
        match_ratio, spans = max(ranked_blocks, key=lambda item: item[0])
        if match_ratio < 0.999:
            raise ValueError(
                f"Could not exactly match input prompt for {gradpo_code}: "
                f"similarity={match_ratio:.4f}"
            )

        base_match = FIRST_STAGE_RE.fullmatch(base_code)
        assert base_match is not None
        runs.append(
            RunSpans(
                code=gradpo_code,
                model=base_match.group("model"),
                improved_on_dev=bool(gradpo_record.prompt),
                spans=spans,
                source_log=str(log_path.relative_to(root)),
            )
        )
    return runs


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(":---" for _ in headers) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def main() -> None:
    runs = collect_runs()
    all_spans = [span for run in runs for span in run.spans]
    exact_counts = Counter(span.casefold() for span in all_spans)
    category_counts = Counter(classify_span(span) for span in all_spans)

    qwen_runs = [run for run in runs if run.model == "qwen"]
    gemma_runs = [run for run in runs if run.model == "gemma"]
    improved_runs = [run for run in runs if run.improved_on_dev]
    unchanged_runs = [run for run in runs if not run.improved_on_dev]
    repeated_occurrences = sum(
        count for count in exact_counts.values() if count >= 2
    )
    singleton_types = sum(count == 1 for count in exact_counts.values())

    if len(all_spans) != 75:
        raise ValueError(f"Expected 75 selected regions; found {len(all_spans)}")
    if len(qwen_runs) != 9 or sum(len(run.spans) for run in qwen_runs) != 45:
        raise ValueError("Unexpected Qwen run/span count")
    if len(gemma_runs) != 10 or sum(len(run.spans) for run in gemma_runs) != 30:
        raise ValueError("Unexpected Gemma run/span count")
    if sum(category_counts.values()) != 75:
        raise ValueError("Category counts do not sum to 75")
    expected_category_counts = {
        "Shared task-content wording": 38,
        "Generic instruction fragments": 23,
        "Decision/output/formatting wording": 12,
        "Relation-specific rule fragments": 2,
    }
    if dict(category_counts) != expected_category_counts:
        raise ValueError(
            f"Unexpected consolidated category counts: {dict(category_counts)}"
        )
    if category_counts["Relation-specific rule fragments"] != 2:
        raise ValueError("Expected exactly two relation-specific fragments")

    category_examples = {
        category: sorted(
            {
                span.casefold()
                for span in all_spans
                if classify_span(span) == category
            },
        )
        for category in CATEGORY_ORDER
    }
    category_rows = [
        [
            category,
            str(category_counts[category]),
            f"{100 * category_counts[category] / len(all_spans):.1f}%",
            ", ".join(f"`{value}`" for value in category_examples[category]),
        ]
        for category in CATEGORY_ORDER
    ]

    recurring_rows = [
        [f"`{span}`", str(count), f"{100 * count / len(all_spans):.1f}%"]
        for span, count in exact_counts.most_common()
        if count >= 2
    ]
    run_rows = [
        [
            run.code,
            "yes" if run.improved_on_dev else "no",
            str(len(run.spans)),
            ", ".join(f"`{span}`" for span in run.spans),
        ]
        for run in runs
    ]

    report = "\n".join(
        [
            "GRADPO-GEN TOP-GRADIENT SPAN CONTENT ANALYSIS",
            "",
            "Purpose",
            "This report summarizes what kinds of prompt regions GradPO-Gen selected",
            "using gradient scores. These are the top-gradient regions proposed for",
            "editing, not necessarily the spans ultimately changed in the retained prompt.",
            "",
            "Scope and extraction",
            "- 19 valid first-stage prompts were analyzed; the invalid",
            "  fewrel_qwen_evoprompt_node_y case was not refined.",
            "- Every GradPO-Gen attempt is included, including development improvements",
            "  and failures.",
            "- Selected regions were recovered from each run's logged tagged input prompt",
            "  and matched exactly to its first-stage prompt.",
            "- Categories below are a manual, mutually exclusive content taxonomy.",
            "",
            "Main statistics",
            f"- Total runs: {len(runs)}",
            f"- Total selected regions: {len(all_spans)}",
            "- Qwen: 9 runs × 5 regions = 45 regions",
            "- Gemma: 10 runs × 3 regions = 30 regions",
            f"- Runs with a non-empty development improvement: {len(improved_runs)}",
            f"- Runs with no development improvement: {len(unchanged_runs)}",
            f"- Regions from development-improvement runs: {sum(len(r.spans) for r in improved_runs)}",
            f"- Regions from development-failure runs: {sum(len(r.spans) for r in unchanged_runs)}",
            f"- Unique exact span strings after case folding: {len(exact_counts)}",
            f"- Occurrences belonging to a repeated exact span: {repeated_occurrences}/75 "
            f"({100 * repeated_occurrences / 75:.1f}%)",
            f"- Exact span types occurring only once: {singleton_types}",
            "",
            "Content categories",
            *markdown_table(
                ["Category", "Count", "Percent", "Exact span strings"],
                category_rows,
            ),
            "",
            "Most important interpretation",
            "- Shared task-content wording accounts for 38/75 regions (50.7%).",
            "- Generic instruction fragments account for 23/75 regions (30.7%).",
            "- Decision, output, and formatting wording accounts for 12/75 regions",
            "  (16.0%).",
            "- Only 2/75 selected regions (2.7%) came from relation-specific rules:",
            "  `province` and the partial span `hier`.",
            "- `province` occurred inside a rule illustrating",
            "  `org:stateorprovince_of_headquarters`.",
            "- `hier` came from `hierarchies` in a rule contrasting `has part` with",
            "  `produced by`.",
            "- Neither is a complete relation label or named entity.",
            "- No selected region contains a complete dataset relation label or a named",
            "  instance entity.",
            "- The other 73/75 regions (97.3%) are shared instruction-level, structural,",
            "  decision, formatting, generic, or partial-word fragments.",
            "- Therefore, the selected gradient regions are overwhelmingly associated",
            "  with shared task instructions rather than individual relations or examples.",
            "",
            "Recurring exact spans",
            *markdown_table(["Exact span", "Count", "Percent"], recurring_rows),
            "",
            "Six prominent recurring task-level spans",
            "- `name` (9), `sentence` (9), `sentence exemplifying` (6),",
            "  `brackets` (6), `You are` (4), and `Answer` (3).",
            "- Together these account for 37/75 selected regions (49.3%).",
            "- The remaining regions are mostly less-frequent instruction fragments and",
            "  generic words; this does not mean that the remaining 50.7% are",
            "  relation-specific.",
            "",
            "Suggested paper text",
            "Across all 19 GradPO-Gen runs, we recovered 75 selected high-gradient",
            "regions. Shared task-content wording accounts for 50.7%, generic",
            "instruction fragments for 30.7%, and decision, output, or formatting",
            "wording for 16.0%. Only two selections (2.7%), `province` and the partial",
            "span `hier`, came from relation-specific rules, and neither is a complete",
            "relation label or named entity. Thus, 73/75 selections (97.3%) concern",
            "shared task instructions rather than individual relations or examples.",
            "",
            "Selected regions by run",
            *markdown_table(
                ["GradPO-Gen code", "Dev improvement", "Regions", "Selected region text"],
                run_rows,
            ),
        ]
    )

    OUTPUT_PATH.write_text(report + "\n", encoding="utf-8")

    category_descriptions = {
        "Shared task-content wording": (
            "General input, sentence, relation-description, subject/entity, and "
            "relation-decision wording shared across the task."
        ),
        "Generic instruction fragments": (
            "Function words, partial words, grammatical framing, and other general "
            "instruction fragments."
        ),
        "Decision/output/formatting wording": (
            "Wording that controls the decision, answer, or relation-description "
            "format."
        ),
        "Relation-specific rule fragments": (
            "Fragments selected from illustrative rules concerning particular "
            "relations."
        ),
    }
    grouped_lines = [
        "GRADPO-GEN SELECTED SPANS: FOUR-CATEGORY COMPLETE LIST",
        "",
        "Scope",
        "- 75 top-gradient selected-region occurrences from all 19 valid",
        "  relation-extraction GradPO-Gen runs.",
        "- Duplicate span strings are retained because each occurrence is a distinct",
        "  gradient selection in a source run.",
        "- These are regions proposed for editing, not necessarily regions ultimately",
        "  changed in the retained prompt.",
        "",
        "Summary",
    ]
    for category in CATEGORY_ORDER:
        count = category_counts[category]
        grouped_lines.append(
            f"- {category}: {count}/75 ({100 * count / len(all_spans):.1f}%)"
        )

    occurrence_total = 0
    for category in CATEGORY_ORDER:
        occurrences = [
            (run.code, span_index, span)
            for run in runs
            for span_index, span in enumerate(run.spans, start=1)
            if classify_span(span) == category
        ]
        occurrence_total += len(occurrences)
        grouped_lines.extend(
            [
                "",
                f"{category.upper()} — {len(occurrences)}/75 "
                f"({100 * len(occurrences) / len(all_spans):.1f}%)",
                category_descriptions[category],
                "",
                "All occurrences (duplicates retained)",
            ]
        )
        for item_index, (code, span_index, span) in enumerate(occurrences, start=1):
            grouped_lines.append(
                f"{item_index:02d}. `{span}` — {code}, selected-region {span_index}"
            )
        category_exact_counts = Counter(
            span.casefold() for _code, _span_index, span in occurrences
        )
        grouped_lines.extend(["", "Exact-string frequency summary"])
        for span, count in category_exact_counts.most_common():
            grouped_lines.append(f"- `{span}`: {count}")

    if occurrence_total != 75:
        raise ValueError(
            f"Expected 75 grouped selected-span occurrences; found {occurrence_total}"
        )
    GROUPED_OUTPUT_PATH.write_text(
        "\n".join(grouped_lines) + "\n",
        encoding="utf-8",
    )

    simple_grouped_lines = []
    simple_occurrence_total = 0
    for category in CATEGORY_ORDER:
        category_spans = [
            span
            for run in runs
            for span in run.spans
            if classify_span(span) == category
        ]
        simple_occurrence_total += len(category_spans)
        simple_grouped_lines.append(f"{category} ({len(category_spans)})")
        simple_grouped_lines.extend(category_spans)
        simple_grouped_lines.append("")
    if simple_occurrence_total != 75:
        raise ValueError(
            "Expected 75 simple grouped selected-span occurrences; "
            f"found {simple_occurrence_total}"
        )
    SIMPLE_GROUPED_OUTPUT_PATH.write_text(
        "\n".join(simple_grouped_lines).rstrip() + "\n",
        encoding="utf-8",
    )
    print(report)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"Saved: {GROUPED_OUTPUT_PATH}")
    print(f"Saved: {SIMPLE_GROUPED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
