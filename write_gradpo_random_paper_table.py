#!/usr/bin/env python3
"""Write the complete GradPO-Gen versus random-span test-results table."""

from __future__ import annotations

from pathlib import Path

from gradpo_random_results_comparison import (
    Score,
    notebook_saved_results_text,
    parse_score_blocks,
)
from prompt_edit_distance_stats import FIRST_STAGE_RE, parse_prompt_records


PROMPTS_PATH = Path("all_the_prompts.txt")
RANDOM_RESULTS_PATH = Path("random_mode_results.txt")
NOTEBOOK_PATH = Path("codes/final_check_results2.ipynb")
OUTPUT_PATH = Path("gradpo_gen_vs_random_all_20_results.txt")

DATASET_ORDER = {"tacred": 0, "fewrel": 1}
MODEL_ORDER = {"qwen": 0, "gemma": 1}
METHOD_ORDER = {"rpo": 0, "evoprompt": 1, "etgpo": 2}
METHOD_LABEL = {"rpo": "RPO", "evoprompt": "EvoPrompt-DE", "etgpo": "ETGPO"}


def format_score(score: Score | None, metric: str) -> str:
    if score is None:
        return "--"
    return f"{getattr(score, metric):.1f} ± {getattr(score, metric + '_std'):.2f}"


def reasoning_iterations(method: str, node: str) -> str:
    if method == "etgpo":
        return "1"
    return "5" if node == "x" else "10"


def main() -> None:
    prompts = parse_prompt_records(PROMPTS_PATH)
    saved_scores = parse_score_blocks(notebook_saved_results_text(NOTEBOOK_PATH))
    random_scores = parse_score_blocks(RANDOM_RESULTS_PATH.read_text(encoding="utf-8"))

    bases = [code for code in prompts if FIRST_STAGE_RE.fullmatch(code)]
    if len(bases) != 20:
        raise ValueError(f"Expected 20 first-stage configurations; found {len(bases)}")

    def sort_key(code: str) -> tuple[int, int, int, str]:
        match = FIRST_STAGE_RE.fullmatch(code)
        assert match is not None
        return (
            DATASET_ORDER[match.group("dataset")],
            MODEL_ORDER[match.group("model")],
            METHOD_ORDER[match.group("first_stage")],
            match.group("iteration"),
        )

    rows: list[list[str]] = []
    gen_count = 0
    random_count = 0
    for base in sorted(bases, key=sort_key):
        match = FIRST_STAGE_RE.fullmatch(base)
        assert match is not None
        gen_code = f"{base}_gradpo-gen"
        random_code = f"{gen_code}_random"

        # A non-empty stored prompt means that the second-stage method improved
        # on development and was evaluated on test. Otherwise its cells stay
        # missing, exactly as in the paper tables.
        gen_score = saved_scores.get(gen_code) if prompts.get(gen_code) else None
        random_score = random_scores.get(random_code)

        if gen_score is not None:
            gen_count += 1
        if random_score is not None:
            random_count += 1

        rows.append(
            [
                match.group("dataset").upper(),
                "Qwen3-4B" if match.group("model") == "qwen" else "Gemma3-4B",
                METHOD_LABEL[match.group("first_stage")],
                reasoning_iterations(
                    match.group("first_stage"), match.group("iteration")
                ),
                format_score(gen_score, "precision"),
                format_score(gen_score, "recall"),
                format_score(gen_score, "f1"),
                format_score(random_score, "precision"),
                format_score(random_score, "recall"),
                format_score(random_score, "f1"),
            ]
        )

    if gen_count != 16:
        raise ValueError(f"Expected 16 GradPO-Gen results; found {gen_count}")
    if random_count != len(random_scores):
        raise ValueError("Not every random result was placed in the 20-row table")

    headers = [
        "Dataset",
        "Model",
        "First stage",
        "R iters",
        "Gen P",
        "Gen R",
        "Gen F1",
        "Random P",
        "Random R",
        "Random F1",
    ]
    alignments = [":---", ":---", ":---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
    table = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(alignments) + "|",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]

    report = "\n".join(
        [
            "GRADPO-GEN VS. GRADPO-GEN-RANDOM: COMPLETE TEST RESULTS",
            "",
            "This table contains all 20 nominal first-stage configurations.",
            "Values are mean ± standard deviation over five 10,000-episode test sets.",
            "A missing value (--) means that the second-stage method did not improve",
            "over its first-stage prompt on development, so test evaluation was omitted.",
            "No first-stage test score is substituted into a missing second-stage cell.",
            "",
            *table,
            "",
            "Coverage:",
            "- Valid first-stage configurations: 19/20",
            f"- GradPO-Gen test results: {gen_count}/19 valid cases",
            f"- GradPO-Gen-Random test results: {random_count}/19 valid cases",
            "",
            "Special case:",
            "- FEWREL / Qwen3-4B / EvoPrompt-DE / 10 iterations has no valid",
            "  first-stage prompt because it did not improve over the 5-iteration",
            "  EvoPrompt-DE prompt; consequently, neither second-stage method was run.",
            "",
            "Source mapping:",
            "- GradPO-Gen: original paper results cached in codes/final_check_results2.ipynb",
            "- GradPO-Gen-Random: random_mode_results.txt",
        ]
    )
    OUTPUT_PATH.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
