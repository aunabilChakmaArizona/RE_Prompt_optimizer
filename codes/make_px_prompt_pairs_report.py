#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import re
import textwrap
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Sequence


DEFAULT_ROOT = Path("gradients_experiments")
DEFAULT_PATTERN = "*_llm_cand_sugg_beam5real_Q5_px_*_r5_20260301_030139_node11_qwen4_lambda_tune"
DEFAULT_OUTPUT_STEM = "px_last_iteration_prompt_pairs"
INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1 = """The answer instruction prompt is:
```
Output only "yes" or "no" as answer, with no explanation or additional text.
```
""".strip()
INFERENCE_INPUT_PROMPT_V1 = """The input block looks like:
```
Relation name: "#RELATION#" (#RELATION_DESCRIPTION#)

#SUPPORT_SENTENCE_BLOCK#

Query Sentence: #QUERY_SENTENCE#

Answer:
```
""".strip()


@dataclass(frozen=True)
class PromptRecord:
    px: float
    folder: str
    summary_path: Path
    prompt: str
    validation_f1: float | None
    dev_f1: float | None
    perplexity: float | None
    loss: float | None
    combined_score: float | None


@dataclass(frozen=True)
class PromptPair:
    left: PromptRecord
    right: PromptRecord


@dataclass(frozen=True)
class DemoPromptExample:
    relation: str
    relation_description: str
    support_sentence: str
    query_sentence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the best prompt from the last iteration for each px run, sample "
            "random cross-px pairs, and export a PDF plus a metrics text file."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing the experiment folders.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern for the px experiment folders under --root.",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=10,
        help="Number of random prompt pairs to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling prompt pairs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "reports",
        help="Directory where the PDF and TXT files will be written.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Filename stem for the generated outputs.",
    )
    return parser.parse_args()


def extract_px(folder_name: str) -> float:
    match = re.search(r"px_(\d+\.\d+)", folder_name)
    if not match:
        raise ValueError(f"Could not parse px value from {folder_name}")
    return float(match.group(1))


def load_prompt_record(summary_path: Path) -> PromptRecord:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    selected = payload.get("final_selected_prompt")
    if selected is None:
        iterations = payload.get("iterations") or []
        if not iterations:
            raise ValueError(f"No iterations or final_selected_prompt in {summary_path}")
        selected = iterations[-1].get("selected_prompt")
    if not selected or not selected.get("prompt"):
        raise ValueError(f"No selected prompt found in {summary_path}")

    metrics = selected.get("selection_metrics") or {}
    balanced_train_prf = selected.get("balanced_train_prf") or {}
    dev_prf = selected.get("dev_prf") or {}
    folder_name = summary_path.parent.name
    return PromptRecord(
        px=extract_px(folder_name),
        folder=folder_name,
        summary_path=summary_path,
        prompt=selected["prompt"],
        validation_f1=balanced_train_prf.get("f1"),
        dev_f1=dev_prf.get("f1"),
        perplexity=metrics.get("mean_perplexity"),
        loss=metrics.get("mean_cross_entropy"),
        combined_score=metrics.get("combined_score"),
    )


def collect_prompt_records(root: Path, pattern: str) -> List[PromptRecord]:
    records = [load_prompt_record(path / "summary.json") for path in sorted(root.glob(pattern))]
    if len(records) < 2:
        raise ValueError("Need at least two px runs to build prompt pairs.")
    return sorted(records, key=lambda item: item.px)


def sample_pairs(records: Sequence[PromptRecord], num_pairs: int, seed: int) -> List[PromptPair]:
    all_pairs = [PromptPair(left=a, right=b) for a, b in combinations(records, 2)]
    if num_pairs > len(all_pairs):
        raise ValueError(
            f"Requested {num_pairs} pairs but only {len(all_pairs)} unique cross-px pairs exist."
        )
    rng = random.Random(seed)
    shuffled_records = list(records)
    rng.shuffle(shuffled_records)

    selected_pairs: List[PromptPair] = []
    seen_pair_keys = set()

    for left, right in zip(shuffled_records, shuffled_records[1:]):
        pair = PromptPair(left=left, right=right)
        pair_key = tuple(sorted((left.px, right.px)))
        if pair_key in seen_pair_keys:
            continue
        selected_pairs.append(pair)
        seen_pair_keys.add(pair_key)
        if len(selected_pairs) == num_pairs:
            return selected_pairs

    remaining_pairs = [pair for pair in all_pairs if tuple(sorted((pair.left.px, pair.right.px))) not in seen_pair_keys]
    rng.shuffle(remaining_pairs)
    for pair in remaining_pairs:
        selected_pairs.append(pair)
        if len(selected_pairs) == num_pairs:
            break

    return selected_pairs


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_prompt_lines(text: str, width: int) -> List[str]:
    lines: List[str] = []
    for paragraph in text.splitlines():
        stripped = paragraph.rstrip()
        if not stripped:
            lines.append("")
            continue
        wrapped = textwrap.wrap(
            stripped,
            width=width,
            break_long_words=True,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        lines.extend(wrapped or [""])
    return lines or [""]


def build_pdf_pages(pairs: Sequence[PromptPair]) -> List[str]:
    page_width = 612
    page_height = 792
    margin = 36
    gutter = 18
    top_margin = 36
    bottom_margin = 36
    font_size = 8
    label_font_size = 10
    line_height = 10
    padding = 6
    label_row_height = 20
    column_width = (page_width - (2 * margin) - gutter) / 2
    chars_per_line = max(20, int(column_width / (font_size * 0.6)))

    pages: List[str] = []
    commands: List[str] = []
    y_top = page_height - top_margin
    y_cursor = y_top

    def start_page() -> None:
        nonlocal commands, y_cursor
        commands = ["0.5 w"]
        y_cursor = y_top

    def flush_page() -> None:
        if commands:
            pages.append("\n".join(commands))

    start_page()
    x_left = margin
    x_mid = margin + column_width
    x_right = page_width - margin

    for pair_index, pair in enumerate(pairs, start=1):
        left_lines = wrap_prompt_lines(pair.left.prompt, chars_per_line)
        right_lines = wrap_prompt_lines(pair.right.prompt, chars_per_line)
        row_line_count = max(len(left_lines), len(right_lines))
        row_height = (2 * padding) + (row_line_count * line_height)
        total_height = label_row_height + row_height
        if y_cursor - total_height < bottom_margin:
            flush_page()
            start_page()

        label_bottom = y_cursor - label_row_height
        commands.extend(
            [
                f"{x_left:.2f} {y_cursor:.2f} m {x_right:.2f} {y_cursor:.2f} l S",
                f"{x_left:.2f} {label_bottom:.2f} m {x_right:.2f} {label_bottom:.2f} l S",
                f"{x_left:.2f} {y_cursor:.2f} m {x_left:.2f} {label_bottom:.2f} l S",
                f"{x_right:.2f} {y_cursor:.2f} m {x_right:.2f} {label_bottom:.2f} l S",
            ]
        )
        label_text = f"Pair {pair_index}"
        label_x = (page_width / 2) - ((len(label_text) * label_font_size * 0.6) / 2)
        label_y = y_cursor - 14
        commands.append(
            f"BT /F1 {label_font_size} Tf 1 0 0 1 {label_x:.2f} {label_y:.2f} Tm ({pdf_escape(label_text)}) Tj ET"
        )

        y_cursor = label_bottom
        y_bottom = y_cursor - row_height
        commands.extend(
            [
                f"{x_left:.2f} {y_cursor:.2f} m {x_right:.2f} {y_cursor:.2f} l S",
                f"{x_left:.2f} {y_bottom:.2f} m {x_right:.2f} {y_bottom:.2f} l S",
                f"{x_left:.2f} {y_cursor:.2f} m {x_left:.2f} {y_bottom:.2f} l S",
                f"{x_mid + (gutter / 2):.2f} {y_cursor:.2f} m {x_mid + (gutter / 2):.2f} {y_bottom:.2f} l S",
                f"{x_right:.2f} {y_cursor:.2f} m {x_right:.2f} {y_bottom:.2f} l S",
            ]
        )

        left_text_x = x_left + padding
        right_text_x = x_mid + (gutter / 2) + padding
        text_y = y_cursor - padding - font_size

        for index, line in enumerate(left_lines):
            line_y = text_y - (index * line_height)
            commands.append(
                f"BT /F1 {font_size} Tf 1 0 0 1 {left_text_x:.2f} {line_y:.2f} Tm ({pdf_escape(line)}) Tj ET"
            )
        for index, line in enumerate(right_lines):
            line_y = text_y - (index * line_height)
            commands.append(
                f"BT /F1 {font_size} Tf 1 0 0 1 {right_text_x:.2f} {line_y:.2f} Tm ({pdf_escape(line)}) Tj ET"
            )

        y_cursor = y_bottom

    flush_page()
    return pages


def write_simple_pdf(path: Path, page_contents: Sequence[str]) -> None:
    font_object_id = 3
    page_object_ids = [4 + (index * 2) for index in range(len(page_contents))]
    content_object_ids = [page_id + 1 for page_id in page_object_ids]
    object_count = 3 + (2 * len(page_contents))

    objects: dict[int, bytes] = {}
    page_refs = " ".join(f"{page_id} 0 R" for page_id in page_object_ids)
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = f"<< /Type /Pages /Kids [{page_refs}] /Count {len(page_object_ids)} >>".encode(
        "ascii"
    )
    objects[font_object_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"

    for page_id, content_id, content in zip(page_object_ids, content_object_ids, page_contents):
        encoded_content = content.encode("latin-1", errors="replace")
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_object_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("ascii")
        objects[content_id] = (
            f"<< /Length {len(encoded_content)} >>\nstream\n".encode("ascii")
            + encoded_content
            + b"\nendstream"
        )

    pdf_parts: List[bytes] = [b"%PDF-1.4\n"]
    offsets = [0] * (object_count + 1)
    for object_id in range(1, object_count + 1):
        offsets[object_id] = sum(len(part) for part in pdf_parts)
        pdf_parts.append(f"{object_id} 0 obj\n".encode("ascii"))
        pdf_parts.append(objects[object_id])
        pdf_parts.append(b"\nendobj\n")

    xref_offset = sum(len(part) for part in pdf_parts)
    pdf_parts.append(f"xref\n0 {object_count + 1}\n".encode("ascii"))
    pdf_parts.append(b"0000000000 65535 f \n")
    for object_id in range(1, object_count + 1):
        pdf_parts.append(f"{offsets[object_id]:010d} 00000 n \n".encode("ascii"))
    pdf_parts.append(
        (
            f"trailer\n<< /Size {object_count + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(pdf_parts))


def format_float(value: float | None) -> str:
    return "NA" if value is None else f"{value:.2f}"


def build_support_block(support_sentences: Sequence[str]) -> str:
    if not support_sentences:
        return ""
    if len(support_sentences) == 1:
        return f"Support Sentence: {support_sentences[0]}"
    return "\n".join(
        f"Support Sentence {index}: {sentence}"
        for index, sentence in enumerate(support_sentences, start=1)
    )


def load_demo_prompt_example(root: Path, pattern: str) -> DemoPromptExample:
    for run_dir in sorted(root.glob(pattern)):
        sample_path = run_dir / "sampled_data_and_predictions.json"
        if not sample_path.exists():
            continue
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        example = None
        iterations = payload.get("iterations") or []
        if iterations:
            sampled_examples = iterations[0].get("sampled_examples") or []
            if sampled_examples:
                example = sampled_examples[0]
        if example is None:
            sampled_examples = payload.get("sampled_examples") or []
            if sampled_examples:
                example = sampled_examples[0]
        if example is None:
            continue
        return DemoPromptExample(
            relation=example["support"]["relation"],
            relation_description=example["relation_description"],
            support_sentence=example["support_sentence"],
            query_sentence=example["query_sentence"],
        )
    raise ValueError("Unable to locate a sampled example for the PDF cover page.")


def render_demo_input_prompt(example: DemoPromptExample) -> str:
    rendered = INFERENCE_INPUT_PROMPT_V1
    rendered = rendered.replace("#RELATION#", example.relation)
    rendered = rendered.replace("#RELATION_DESCRIPTION#", example.relation_description)
    rendered = rendered.replace(
        "#SUPPORT_SENTENCE_BLOCK#",
        build_support_block([example.support_sentence]),
    )
    rendered = rendered.replace("#QUERY_SENTENCE#", example.query_sentence)
    return rendered


def load_prompt_lineage(summary_path: Path) -> List[tuple[int, str]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    iterations = payload.get("iterations") or []
    lineage: List[tuple[int, str]] = []
    for iteration in iterations:
        selected_prompt = (iteration.get("selected_prompt") or {}).get("prompt")
        iteration_index = iteration.get("iteration_index")
        if selected_prompt and iteration_index is not None:
            lineage.append((int(iteration_index), selected_prompt))
    return lineage


def build_cover_page(example: DemoPromptExample) -> str:
    page_width = 612
    page_height = 792
    margin = 36
    top_margin = 40
    bottom_margin = 36
    body_font_size = 9
    heading_font_size = 14
    section_font_size = 11
    line_height = 12
    usable_width = page_width - (2 * margin)
    chars_per_line = max(40, int(usable_width / (body_font_size * 0.6)))

    sections = [
        ("", [
            "The following pages contain instruction prompts only.",
            "During inference, each instruction prompt is followed by a fixed answer instruction prompt and a fixed input prompt template.",
        ]),
        ("", INFERENCE_ANSWER_INSTRUCTION_PROMPT_V1.splitlines()),
        ("", INFERENCE_INPUT_PROMPT_V1.splitlines()),
        # ("Demo Input Prompt", render_demo_input_prompt(example).splitlines()),
    ]

    commands: List[str] = ["0.5 w"]
    y_cursor = page_height - top_margin

    def add_text_block(lines: Iterable[str], *, font_size: int, gap_after: int = 8) -> None:
        nonlocal y_cursor
        wrapped_lines: List[str] = []
        for raw_line in lines:
            if raw_line == "":
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(
                textwrap.wrap(
                    raw_line,
                    width=chars_per_line,
                    break_long_words=True,
                    replace_whitespace=False,
                    drop_whitespace=False,
                )
                or [""]
            )
        for line in wrapped_lines:
            if y_cursor < bottom_margin:
                break
            if line:
                commands.append(
                    f"BT /F1 {font_size} Tf 1 0 0 1 {margin:.2f} {y_cursor:.2f} Tm ({pdf_escape(line)}) Tj ET"
                )
            y_cursor -= line_height
        y_cursor -= gap_after

    # add_text_block(["Prompt Pair Report"], font_size=heading_font_size, gap_after=12)
    for title, content_lines in sections:
        add_text_block([title], font_size=section_font_size, gap_after=4)
        add_text_block(content_lines, font_size=body_font_size, gap_after=10)

    return "\n".join(commands)


def write_metrics_report(path: Path, pairs: Sequence[PromptPair], seed: int) -> None:
    headers = [
        "pair_id",
        "side",
        "λ",
        "dev_f1",
        "perplexity",
        "loss",
        "objective (= loss + λ·perplexity)",
    ]
    rows: List[List[str]] = []
    for pair_index, pair in enumerate(pairs, start=1):
        for column_name, record in (("left", pair.left), ("right", pair.right)):
            rows.append(
                [
                    str(pair_index),
                    column_name,
                    format_float(record.px),
                    format_float(record.dev_f1),
                    format_float(record.perplexity),
                    format_float(record.loss),
                    format_float(record.combined_score),
                ]
            )

    column_widths = [
        max(len(header), max((len(row[index]) for row in rows), default=0))
        for index, header in enumerate(headers)
    ]
    justified_rows = [
        "  ".join(value.ljust(column_widths[index]) for index, value in enumerate(headers))
    ]
    justified_rows.extend(
        "  ".join(value.ljust(column_widths[index]) for index, value in enumerate(row))
        for row in rows
    )

    lines = [
        "dev_f1 = dev_prf.f1 from summary.json",
        "λ = perplexity weight from the run",
        "objective = loss + λ·perplexity",
        f"random_seed = {seed}",
        "",
        *justified_rows,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_parent_prompt_report(path: Path, pairs: Sequence[PromptPair]) -> None:
    pair_occurrences: dict[tuple[float, str], List[str]] = {}
    unique_records: dict[tuple[float, str], PromptRecord] = {}
    for pair_index, pair in enumerate(pairs, start=1):
        for side, record in (("left", pair.left), ("right", pair.right)):
            key = (record.px, record.folder)
            pair_occurrences.setdefault(key, []).append(f"pair_{pair_index}_{side}")
            unique_records[key] = record

    lines: List[str] = []
    for key in sorted(unique_records, key=lambda item: item[0]):
        record = unique_records[key]
        lineage = load_prompt_lineage(record.summary_path)
        lines.append(f"lambda = {record.px:.2f}")
        lines.append(f"used_in = {', '.join(pair_occurrences[key])}")
        lines.append(f"final_selected_prompt_folder = {record.folder}")
        lines.append("")
        for iteration_index, prompt_text in lineage:
            lines.append(f"Iteration {iteration_index}")
            lines.append(prompt_text)
            lines.append("")
        lines.append("=" * 80)
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = collect_prompt_records(args.root, args.pattern)
    pairs = sample_pairs(records, args.num_pairs, args.seed)
    demo_example = load_demo_prompt_example(args.root, args.pattern)
    pdf_path = args.output_dir / f"{args.output_stem}.pdf"
    txt_path = args.output_dir / f"{args.output_stem}_metrics.txt"
    parent_txt_path = args.output_dir / f"{args.output_stem}_parent_prompts.txt"

    pair_pages = build_pdf_pages(pairs)
    write_simple_pdf(pdf_path, [build_cover_page(demo_example), *pair_pages])
    write_metrics_report(txt_path, pairs, args.seed)
    write_parent_prompt_report(parent_txt_path, pairs)

    print(f"selected_px_runs={len(records)}")
    print(f"pairs_written={len(pairs)}")
    print(f"pdf={pdf_path}")
    print(f"metrics_txt={txt_path}")
    print(f"parent_prompts_txt={parent_txt_path}")


if __name__ == "__main__":
    main()
