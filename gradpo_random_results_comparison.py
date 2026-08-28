#!/usr/bin/env python3
"""Report the full GradPO-Gen and GradPO-Gen-Random result grid.

Random-mode scores are read only from ``random_mode_results.txt``.  GradPO-Gen
scores are read from the cached original evaluation table in
``codes/final_check_results2.ipynb``, which used the same five-chunk scoring
procedure as the paper.  Missing results remain blank; first-stage scores are
never substituted for a missing second-stage result.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from prompt_edit_distance_stats import FIRST_STAGE_RE, parse_prompt_records


SCORE_BLOCK_RE = re.compile(
    r"(?m)^(?P<filename>[^\n]+\.txt)\n"
    r"(?P<count>\d+)\n"
    r"(?P<p>[0-9.]+)\s*±\s*(?P<p_std>[0-9.]+)\s*\t"
    r"(?P<r>[0-9.]+)\s*±\s*(?P<r_std>[0-9.]+)\s*\t"
    r"(?P<f1>[0-9.]+)\s*±\s*(?P<f1_std>[0-9.]+)$"
)
RANDOM_CODE_RE = re.compile(
    r"^(?P<code>"
    r"(?P<dataset>tacred|fewrel)_"
    r"(?P<model>qwen|gemma)_"
    r"(?P<first_stage>rpo|evoprompt|etgpo)_"
    r"node_(?P<iteration>x|y)_gradpo-gen_random)_"
)
RESULT_CODE_RE = re.compile(
    r"^(?P<code>.+?)_(?:google-gemma-3-4b-it|Qwen-Qwen3-4B)-"
)
FIRST_STAGE_ORDER = ("rpo", "evoprompt", "etgpo")
FIRST_STAGE_LABEL = {
    "rpo": "RPO",
    "evoprompt": "EvoPrompt",
    "etgpo": "ETGPO",
}


@dataclass(frozen=True)
class Score:
    precision: float
    precision_std: float
    recall: float
    recall_std: float
    f1: float
    f1_std: float
    examples: int


@dataclass(frozen=True)
class GridResult:
    base_code: str
    dataset: str
    model: str
    first_stage: str
    iteration: str
    normal: Score | None
    random: Score | None


def parse_score_blocks(text: str) -> dict[str, Score]:
    """Parse notebook-style filename/count/P-R-F1 blocks by result code."""
    scores: dict[str, Score] = {}
    for match in SCORE_BLOCK_RE.finditer(text):
        filename = Path(match.group("filename")).name
        code_match = RESULT_CODE_RE.match(filename)
        if not code_match:
            continue
        code = code_match.group("code")
        if code in scores:
            raise ValueError(f"Duplicate score block for {code}")
        scores[code] = Score(
            precision=float(match.group("p")),
            precision_std=float(match.group("p_std")),
            recall=float(match.group("r")),
            recall_std=float(match.group("r_std")),
            f1=float(match.group("f1")),
            f1_std=float(match.group("f1_std")),
            examples=int(match.group("count")),
        )
    return scores


def notebook_saved_results_text(path: Path) -> str:
    """Return saved score blocks from both cell sources and cell outputs.

    The original full GradPO evaluation table is preserved in a notebook cell,
    while the latest random-mode run is in a cell output.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    pieces: list[str] = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        pieces.append(source if isinstance(source, str) else "".join(source))
        for output in cell.get("outputs", []):
            value = output.get("text", [])
            pieces.append(value if isinstance(value, str) else "".join(value))
    return "\n".join(pieces)


def load_results_grid(
    random_results_path: Path,
    prompts_path: Path,
    notebook_path: Path,
) -> list[GridResult]:
    random_scores = parse_score_blocks(random_results_path.read_text(encoding="utf-8"))
    saved_scores = parse_score_blocks(notebook_saved_results_text(notebook_path))
    prompts = parse_prompt_records(prompts_path)
    all_bases = sorted(code for code in prompts if FIRST_STAGE_RE.fullmatch(code))
    valid_bases = [code for code in all_bases if prompts[code]]
    rows: list[GridResult] = []

    if len(all_bases) != 20 or len(valid_bases) != 19:
        raise ValueError(
            f"Expected 20 canonical and 19 valid first-stage prompts; found "
            f"{len(all_bases)} and {len(valid_bases)}"
        )

    expected_random_codes = {f"{base}_gradpo-gen_random" for base in valid_bases}
    unexpected_random = set(random_scores) - expected_random_codes
    if unexpected_random:
        raise ValueError(f"Unexpected random result codes: {sorted(unexpected_random)}")

    for base_code in valid_bases:
        match = FIRST_STAGE_RE.fullmatch(base_code)
        assert match is not None
        normal_code = f"{base_code}_gradpo-gen"
        random_code = f"{normal_code}_random"
        normal = None
        if prompts.get(normal_code):
            normal = saved_scores.get(normal_code)
            if normal is None:
                raise ValueError(f"No cached GradPO-Gen score for {normal_code}")
        random = random_scores.get(random_code)
        if random is not None and not prompts.get(random_code):
            raise ValueError(f"Random score has no non-empty prompt: {random_code}")
        rows.append(
            GridResult(
                base_code=base_code,
                dataset=match.group("dataset"),
                model=match.group("model"),
                first_stage=match.group("first_stage"),
                iteration=match.group("iteration"),
                normal=normal,
                random=random,
            )
        )

    if sum(row.normal is not None for row in rows) != 16:
        raise ValueError("Expected 16 reported GradPO-Gen results")
    if sum(row.random is not None for row in rows) != len(random_scores):
        raise ValueError("Not every parsed random result was placed in the report grid")
    if any(
        score.examples != 150_000
        for row in rows
        for score in (row.normal, row.random)
        if score is not None
    ):
        raise ValueError("A reported result is not a completed 150,000-example run")
    return sorted(
        rows,
        key=lambda row: (
            row.dataset,
            row.model,
            FIRST_STAGE_ORDER.index(row.first_stage),
            row.iteration,
        ),
    )


def plain_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[i]) for i, value in enumerate(row))
        for row in rows
    )
    return lines


def format_score(score: Score | None, metric: str) -> str:
    if score is None:
        return ""
    return f"{getattr(score, metric):.1f} ± {getattr(score, metric + '_std'):.2f}"


def first_stage_case(row: GridResult) -> str:
    if row.first_stage == "etgpo":
        return "ETGPO-1 (x)"
    iteration = "5" if row.iteration == "x" else "10"
    return f"{FIRST_STAGE_LABEL[row.first_stage]}-{iteration} ({row.iteration})"


def format_comparison(rows: list[GridResult]) -> list[str]:
    lines = [
        "GRADPO-GEN VS GRADPO-GEN-RANDOM TEST RESULTS",
        "",
        "Full grid: 20 nominal first-stage cases, 19 valid and reported here.",
        "Excluded case: fewrel_qwen_evoprompt_node_y (empty first-stage prompt).",
        "GradPO-Gen values follow the paper's reported test results. Random values",
        "come only from random_mode_results.txt. Missing values remain blank because",
        "the second-stage method did not improve on dev or has no result in its",
        "designated source; no first-stage score is substituted.",
        "Scores are P/R/F1 mean ± population std over five 30,000-example chunks.",
        "Delta F1 = GradPO-Gen-Random minus GradPO-Gen, in percentage points, and is",
        "shown only when both results exist.",
        "",
    ]

    groups = (
        ("fewrel", "qwen", "FEWREL / QWEN"),
        ("fewrel", "gemma", "FEWREL / GEMMA"),
        ("tacred", "qwen", "TACRED / QWEN"),
        ("tacred", "gemma", "TACRED / GEMMA"),
    )
    headers = (
        "first_stage",
        "gen_P",
        "gen_R",
        "gen_F1",
        "random_P",
        "random_R",
        "random_F1",
        "delta_F1",
    )
    for dataset, model, label in groups:
        items = [row for row in rows if row.dataset == dataset and row.model == model]
        table_rows = []
        for row in items:
            delta = ""
            if row.normal is not None and row.random is not None:
                delta = f"{row.random.f1 - row.normal.f1:+.1f}"
            table_rows.append(
                (
                    first_stage_case(row),
                    format_score(row.normal, "precision"),
                    format_score(row.normal, "recall"),
                    format_score(row.normal, "f1"),
                    format_score(row.random, "precision"),
                    format_score(row.random, "recall"),
                    format_score(row.random, "f1"),
                    delta,
                )
            )
        lines.extend(
            [
                f"{label} ({len(items)} valid first-stage cases)",
                *plain_table(headers, table_rows),
                "",
            ]
        )

    lines.extend(
        [
            "COVERAGE",
            f"  GradPO-Gen results:        {sum(row.normal is not None for row in rows)}/19",
            f"  GradPO-Gen-Random results: {sum(row.random is not None for row in rows)}/19",
            f"  Directly matched results:  {sum(row.normal is not None and row.random is not None for row in rows)}/19",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-results", type=Path, default=Path("random_mode_results.txt"))
    parser.add_argument("--prompts", type=Path, default=Path("all_the_prompts.txt"))
    parser.add_argument(
        "--notebook", type=Path, default=Path("codes/final_check_results2.ipynb")
    )
    args = parser.parse_args()
    rows = load_results_grid(args.random_results, args.prompts, args.notebook)
    print("\n".join(format_comparison(rows)))


if __name__ == "__main__":
    main()
