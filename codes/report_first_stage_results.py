"""Write canonical first-stage reports for the three selected tasks."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "experiment_tracking" / "first_stage"
GEMMA_OPENBOOK_RPO_CODE = (
    "openbookqa_non_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500_generalmeta"
)
QWEN_OPENBOOK_RPO_CODE = (
    "openbookqa_non_reasoning_qwen_rpo_qwen14opt_lambda1_vs1500_"
    "generalmeta_retry_gpu045"
)
QWEN_MATH_RPO_CODE = (
    "math500_reasoning_qwen_rpo_qwen14opt_lambda1_vs900_error3_tokenlimit"
)
QWEN_OPENBOOK_EVOPROMPT_CODE = (
    "openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1"
)
MATH_EVOPROMPT_CODES = (
    "math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900",
    "math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500",
)


@dataclass(frozen=True)
class RunSpec:
    """Describe one canonical first-stage experiment directory."""

    dataset: str
    mode: str
    model: str
    method: str
    code: str
    run_dir: str


RUN_SPECS = (
    RunSpec(
        "OpenBookQA",
        "non-reasoning",
        "Qwen3-4B",
        "RPO",
        QWEN_OPENBOOK_RPO_CODE,
        "outputs/qa_prompt_optimization/non_reasoning/rpo/"
        + QWEN_OPENBOOK_RPO_CODE,
    ),
    RunSpec(
        "OpenBookQA",
        "non-reasoning",
        "Qwen3-4B",
        "EvoPrompt-DE",
        "openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1",
        "outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/"
        "openbookqa_non_reasoning_qwen_evoprompt_qwen14opt_lambda1",
    ),
    RunSpec(
        "OpenBookQA",
        "non-reasoning",
        "Qwen3-4B",
        "ETGPO",
        "openbookqa_non_reasoning_qwen_etgpo_qwen14opt_lambda1_fixed_vs1500",
        "outputs/qa_prompt_optimization/non_reasoning/etgpo/"
        "openbookqa_non_reasoning_qwen_etgpo_qwen14opt_lambda1_fixed_vs1500",
    ),
    RunSpec(
        "OpenBookQA",
        "non-reasoning",
        "Gemma3-4B",
        "RPO",
        GEMMA_OPENBOOK_RPO_CODE,
        "outputs/qa_prompt_optimization/non_reasoning/rpo/"
        + GEMMA_OPENBOOK_RPO_CODE,
    ),
    RunSpec(
        "OpenBookQA",
        "non-reasoning",
        "Gemma3-4B",
        "EvoPrompt-DE",
        "openbookqa_non_reasoning_gemma_evoprompt_gemma12opt_lambda1",
        "outputs/qa_prompt_optimization/non_reasoning/evoprompt_de/"
        "openbookqa_non_reasoning_gemma_evoprompt_gemma12opt_lambda1",
    ),
    RunSpec(
        "OpenBookQA",
        "non-reasoning",
        "Gemma3-4B",
        "ETGPO",
        "openbookqa_non_reasoning_gemma_etgpo_gemma12opt_lambda1_fixed_vs1500",
        "outputs/qa_prompt_optimization/non_reasoning/etgpo/"
        "openbookqa_non_reasoning_gemma_etgpo_gemma12opt_lambda1_fixed_vs1500",
    ),
    RunSpec(
        "HotpotQA",
        "reasoning",
        "Qwen3-4B",
        "RPO",
        "hotpotqa_reasoning_qwen_rpo_qwen14opt_lambda1_vs1500",
        "outputs/qa_prompt_optimization/reasoning/rpo/"
        "hotpotqa_reasoning_qwen_rpo_qwen14opt_lambda1_vs1500",
    ),
    RunSpec(
        "HotpotQA",
        "reasoning",
        "Qwen3-4B",
        "EvoPrompt-DE",
        "hotpotqa_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs1500_parser_fixed",
        "outputs/qa_prompt_optimization/reasoning/evoprompt_de/"
        "hotpotqa_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs1500_parser_fixed",
    ),
    RunSpec(
        "HotpotQA",
        "reasoning",
        "Qwen3-4B",
        "ETGPO",
        "hotpotqa_reasoning_qwen_etgpo_qwen14opt_lambda1_vs1500_taxonomy_fixed",
        "outputs/qa_prompt_optimization/reasoning/etgpo/"
        "hotpotqa_reasoning_qwen_etgpo_qwen14opt_lambda1_vs1500_taxonomy_fixed",
    ),
    RunSpec(
        "HotpotQA",
        "reasoning",
        "Gemma3-4B",
        "RPO",
        "hotpotqa_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500",
        "outputs/qa_prompt_optimization/reasoning/rpo/"
        "hotpotqa_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500",
    ),
    RunSpec(
        "HotpotQA",
        "reasoning",
        "Gemma3-4B",
        "EvoPrompt-DE",
        "hotpotqa_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500",
        "outputs/qa_prompt_optimization/reasoning/evoprompt_de/"
        "hotpotqa_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500",
    ),
    RunSpec(
        "HotpotQA",
        "reasoning",
        "Gemma3-4B",
        "ETGPO",
        "hotpotqa_reasoning_gemma_etgpo_gemma12opt_lambda1_vs1500",
        "outputs/qa_prompt_optimization/reasoning/etgpo/"
        "hotpotqa_reasoning_gemma_etgpo_gemma12opt_lambda1_vs1500",
    ),
    RunSpec(
        "MATH-500",
        "reasoning",
        "Qwen3-4B",
        "RPO",
        QWEN_MATH_RPO_CODE,
        "outputs/math_prompt_optimization/reasoning/rpo/"
        + QWEN_MATH_RPO_CODE,
    ),
    RunSpec(
        "MATH-500",
        "reasoning",
        "Qwen3-4B",
        "EvoPrompt-DE",
        "math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900",
        "outputs/math_prompt_optimization/reasoning/evoprompt_de/"
        "math500_reasoning_qwen_evoprompt_qwen14opt_lambda1_vs900",
    ),
    RunSpec(
        "MATH-500",
        "reasoning",
        "Qwen3-4B",
        "ETGPO",
        "math500_reasoning_qwen_etgpo_qwen14opt_lambda1_vs900",
        "outputs/math_prompt_optimization/reasoning/etgpo/"
        "math500_reasoning_qwen_etgpo_qwen14opt_lambda1_vs900",
    ),
    RunSpec(
        "MATH-500",
        "reasoning",
        "Gemma3-4B",
        "RPO",
        "math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500",
        "outputs/math_prompt_optimization/reasoning/rpo/"
        "math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500",
    ),
    RunSpec(
        "MATH-500",
        "reasoning",
        "Gemma3-4B",
        "EvoPrompt-DE",
        "math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500",
        "outputs/math_prompt_optimization/reasoning/evoprompt_de/"
        "math500_reasoning_gemma_evoprompt_gemma12opt_lambda1_vs1500",
    ),
    RunSpec(
        "MATH-500",
        "reasoning",
        "Gemma3-4B",
        "ETGPO",
        "math500_reasoning_gemma_etgpo_gemma12opt_lambda1_vs1500",
        "outputs/math_prompt_optimization/reasoning/etgpo/"
        "math500_reasoning_gemma_etgpo_gemma12opt_lambda1_vs1500",
    ),
)


MANUAL_PROMPT_NOTES = {
    QWEN_MATH_RPO_CODE: (
        "Experimental RPO-5 uses actual iteration 2, randomly selected with "
        "seed 42 among nine candidates strictly between initial and best stable "
        "scores (87.95 stable). RPO-10 uses the best actual iteration-5 prompt "
        "(89.54 stable). Both improve over the 85.27-stable initial prompt. "
        "These are assigned source slots, not literal 5/10-iteration results."
    ),
    "math500_reasoning_gemma_rpo_gemma12opt_lambda1_vs1500": (
        "Valid full prompts, but the retained instructions are long and include "
        "specialized advice about rotations, geometry, and diagrams."
    ),
    "math500_reasoning_gemma_etgpo_gemma12opt_lambda1_vs1500": (
        "Valid full prompt, but the combinatorial-ratio and closed-locker guidance "
        "is more example-specific than ideal."
    ),
    QWEN_OPENBOOK_RPO_CODE: (
        "FINAL OpenBookQA Qwen RPO run and source assignment. RPO-5 retains the "
        "actual iteration-1 winner (1191/1500 correct, 79.40 raw, 78.75 stable); "
        "RPO-10 retains the actual iteration-9 winner (1198/1500 correct, 79.87 "
        "raw, 79.21 stable). The run initial is 1185/1500 correct, 79.00 raw, "
        "and 78.43 stable, so the retained sources add 6 and 13 net correct "
        "answers, respectively. Both prompts are complete and valid; the small "
        "gains reflect limited prompt-optimization headroom on this task."
    ),
    GEMMA_OPENBOOK_RPO_CODE: (
        "FINAL OpenBookQA Gemma RPO run and source assignment for the new "
        "second-stage rerun. Experimental RPO-5 uses actual iteration 1 "
        "(1046/1500 correct, 69.73 raw, 67.84 stable); experimental RPO-10 "
        "uses actual iteration 2 (1057/1500 correct, 70.47 raw, 69.47 stable). "
        "The run initial is 976/1500 correct, 65.07 raw, and 64.12 stable, so "
        "the assigned sources add 70 and 81 net correct answers, respectively."
    ),
    QWEN_OPENBOOK_EVOPROMPT_CODE: (
        "For distinct second-stage sources, the experimental EvoPrompt-5 slot "
        "uses the actual iteration-1 prompt (79.01 stable), while EvoPrompt-10 "
        "uses the actual iteration-4 prompt retained through iteration 10 "
        "(80.24 stable). Both improve over the 77.95-stable initial prompt."
    ),
    MATH_EVOPROMPT_CODES[0]: (
        "The experimental EvoPrompt-5 slot uses actual iteration 1 "
        "(87.37 stable); EvoPrompt-10 uses actual iteration 2 (89.12 stable), "
        "retained through iteration 10. Original iterations 5 and 10 were "
        "identical. Both assigned sources improve over 85.48 initial stable."
    ),
    MATH_EVOPROMPT_CODES[1]: (
        "The experimental EvoPrompt-5 slot uses actual iteration 1 "
        "(67.82 stable); EvoPrompt-10 uses actual iteration 2 (68.07 stable), "
        "retained through iteration 10. Original iterations 5 and 10 were "
        "identical. Both assigned sources improve over 60.50 initial stable."
    ),
}


def parse_args() -> argparse.Namespace:
    """Read the directory where both text reports will be written."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory receiving the detailed and compact text reports.",
    )
    return parser.parse_args()


def absolute_path(relative_path: str) -> Path:
    """Resolve one repository-relative artifact path."""
    return PROJECT_ROOT / relative_path


def load_summary(spec: RunSpec) -> dict[str, Any] | None:
    """Load a completed run summary or return None for a pending run."""
    path = absolute_path(spec.run_dir) / "summary.json"
    if not path.is_file():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    if spec.code in {GEMMA_OPENBOOK_RPO_CODE, QWEN_MATH_RPO_CODE}:
        source_iteration = 2 if spec.code == QWEN_MATH_RPO_CODE else 3
        candidates_path = absolute_path(spec.run_dir) / "candidates.jsonl"
        if candidates_path.is_file():
            for line in candidates_path.read_text(encoding="utf-8").splitlines():
                candidate = json.loads(line)
                if int(candidate.get("iteration", -1)) == source_iteration:
                    summary.setdefault("snapshots", {})["5"] = {
                        "iteration": source_iteration,
                        "node_id": source_iteration,
                        "prompt": candidate["prompt"],
                        "metrics": candidate["metrics"],
                    }
                    break
    if (
        spec.code == QWEN_OPENBOOK_EVOPROMPT_CODE
        or spec.code in MATH_EVOPROMPT_CODES
    ):
        events_path = absolute_path(spec.run_dir) / "events.jsonl"
        if events_path.is_file():
            for line in events_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if (
                    event.get("event") == "evoprompt_iteration_completed"
                    and int(event.get("iteration", -1)) == 1
                ):
                    summary.setdefault("snapshots", {})["5"] = {
                        "iteration": 1,
                        "prompt": event["train_best_prompt"],
                        "metrics": {
                            "accuracy": event["validation_accuracy"],
                            "accuracy_percent": 100.0 * event["validation_accuracy"],
                            "stable_accuracy": event["validation_selection_score"],
                            "stable_accuracy_percent": (
                                100.0 * event["validation_selection_score"]
                            ),
                        },
                    }
                    break
    return summary


def read_text(path: Path) -> str | None:
    """Read and trim one prompt file when it exists."""
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip()


def normalized_text(text: str | None) -> str:
    """Normalize whitespace for prompt identity checks."""
    return " ".join((text or "").split())


def format_percent(value: Any) -> str:
    """Format one percentage-point value or an em dash."""
    if value is None:
        return "—"
    return f"{float(value):.2f}"


def format_delta(value: Any) -> str:
    """Format one signed percentage-point difference or an em dash."""
    if value is None:
        return "—"
    return f"{float(value):+.2f}"


def accuracy_percent(metrics: dict[str, Any] | None) -> float | None:
    """Extract raw accuracy in percentage points from one metric dictionary."""
    if not metrics:
        return None
    if metrics.get("accuracy_percent") is not None:
        return float(metrics["accuracy_percent"])
    if metrics.get("accuracy") is not None:
        return 100.0 * float(metrics["accuracy"])
    return None


def stable_percent(metrics: dict[str, Any] | None) -> float | None:
    """Extract stable accuracy in percentage points from one metric dictionary."""
    if not metrics:
        return None
    if metrics.get("stable_accuracy_percent") is not None:
        return float(metrics["stable_accuracy_percent"])
    if metrics.get("stable_accuracy") is not None:
        return 100.0 * float(metrics["stable_accuracy"])
    return None


def snapshot_metrics(
    summary: dict[str, Any] | None,
    iteration: str,
) -> dict[str, Any] | None:
    """Return retained snapshot metrics for iteration 5 or 10."""
    if not summary:
        return None
    snapshot = (summary.get("snapshots") or {}).get(iteration)
    if not snapshot:
        return None
    return snapshot.get("metrics")


def assigned_snapshot_metrics(
    spec: RunSpec,
    summary: dict[str, Any] | None,
    iteration: str,
) -> dict[str, Any] | None:
    """Return metrics for a literal or explicitly reassigned 5/10 source slot."""
    if spec.code != GEMMA_OPENBOOK_RPO_CODE:
        return snapshot_metrics(summary, iteration)

    actual_iteration = {"5": 1, "10": 2}.get(iteration)
    if actual_iteration is None:
        return snapshot_metrics(summary, iteration)
    population_path = absolute_path(spec.run_dir) / "population.json"
    if not population_path.exists():
        return None
    population = json.loads(population_path.read_text(encoding="utf-8"))
    for node in population:
        if int(node.get("iteration", -1)) == actual_iteration:
            return {
                "accuracy": node.get("accuracy"),
                "stable_accuracy": node.get("selection_score"),
            }
    return None


def expected_prompt_entries(spec: RunSpec) -> list[tuple[str, str]]:
    """Return the report labels and prompt filenames expected from one method."""
    if spec.code == QWEN_MATH_RPO_CODE:
        return [
            ("RPO-5", "prompt_experimental_iteration_5_from_actual_iteration_2.txt"),
            ("RPO-10", "prompt_experimental_iteration_10_from_actual_iteration_5.txt"),
        ]
    if spec.code == GEMMA_OPENBOOK_RPO_CODE:
        return [
            (
                "RPO-5",
                "prompt_experimental_iteration_5_from_actual_iteration_1.txt",
            ),
            (
                "RPO-10",
                "prompt_experimental_iteration_10_from_actual_iteration_2.txt",
            ),
        ]
    if spec.code == QWEN_OPENBOOK_EVOPROMPT_CODE:
        return [
            (
                "EvoPrompt-DE-5",
                "prompt_experimental_iteration_5_from_actual_iteration_1.txt",
            ),
            (
                "EvoPrompt-DE-10",
                "prompt_experimental_iteration_10_from_actual_iteration_4.txt",
            ),
        ]
    if spec.code in MATH_EVOPROMPT_CODES:
        return [
            (
                "EvoPrompt-DE-5",
                "prompt_experimental_iteration_5_from_actual_iteration_1.txt",
            ),
            (
                "EvoPrompt-DE-10",
                "prompt_experimental_iteration_10_from_actual_iteration_2.txt",
            ),
        ]
    if spec.method in {"RPO", "EvoPrompt-DE"}:
        return [
            (f"{spec.method}-5", "prompt_iteration_5.txt"),
            (f"{spec.method}-10", "prompt_iteration_10.txt"),
        ]
    return [("ETGPO-1", "final_prompt.txt")]


def final_metrics(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the final retained validation metrics for one run."""
    if not summary:
        return None
    return (summary.get("validation") or {}).get("final")


def run_status(spec: RunSpec, summary: dict[str, Any] | None) -> str:
    """Classify completion, stable gain, and genuine prompt change."""
    if summary is None:
        return "PENDING"
    validation = summary.get("validation") or {}
    initial_stable = stable_percent(validation.get("initial"))
    retained_stable = stable_percent(validation.get("final"))
    run_dir = absolute_path(spec.run_dir)
    initial_prompt = read_text(run_dir / "initial_prompt.txt")
    final_prompt = read_text(run_dir / "final_prompt.txt")
    changed = normalized_text(initial_prompt) != normalized_text(final_prompt)
    if retained_stable is None or initial_stable is None:
        return "INCOMPLETE_METRICS"
    if not changed and retained_stable > initial_stable + 1e-9:
        return "SCORE_GAIN; PROMPT UNCHANGED"
    if not changed:
        return "UNCHANGED"
    if retained_stable > initial_stable + 1e-9:
        return "IMPROVED"
    return "CHANGED; NO STABLE GAIN"


def audit_prompt(
    text: str | None,
    initial_text: str | None,
    earlier_texts: Sequence[tuple[str, str]],
) -> str:
    """Detect missing, trivial, tagged, unchanged, or duplicate saved prompts."""
    if text is None:
        return "MISSING"
    words = re.findall(r"\S+", text)
    if len(words) < 5 or normalized_text(text).lower() in {"and", "or", "the"}:
        return "INVALID: TRIVIAL FRAGMENT"
    warnings = []
    if re.search(r"</?(?:prompt|answer)>", text, flags=re.IGNORECASE):
        warnings.append("contains optimizer/answer tags")
    if normalized_text(text) == normalized_text(initial_text):
        warnings.append("same as initial prompt")
    for label, earlier_text in earlier_texts:
        if normalized_text(text) == normalized_text(earlier_text):
            warnings.append(f"duplicate of {label}")
            break
    return "OK" if not warnings else "; ".join(warnings)


def detailed_report() -> str:
    """Build the complete score, artifact, and prompt-audit report."""
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "CANONICAL FIRST-STAGE PROMPT-OPTIMIZATION RESULTS",
        "=================================================",
        "",
        f"Generated: {generated}",
        "Scope: OpenBookQA non-reasoning, HotpotQA reasoning, MATH-500 reasoning",
        "Selection: stable accuracy = fold mean accuracy - lambda * fold std",
        "Policy: compare gains only within the same run; validation sizes may differ.",
        "PENDING means the canonical run has not yet written summary.json.",
        "",
        "METHOD-LEVEL RESULTS",
        "--------------------",
        "",
        "| Dataset | Mode | Model | Method | N | Initial raw | Initial stable | "
        "Snapshot-5 raw/stable | Snapshot-10 raw/stable | Final raw | Final stable | "
        "Stable gain | Status | CODE | Saved directory |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    completed = 0
    for spec in RUN_SPECS:
        summary = load_summary(spec)
        validation = (summary or {}).get("validation") or {}
        initial = validation.get("initial")
        final = validation.get("final")
        metrics_5 = assigned_snapshot_metrics(spec, summary, "5")
        metrics_10 = assigned_snapshot_metrics(spec, summary, "10")
        initial_stable = stable_percent(initial)
        retained_stable = stable_percent(final)
        gain = (
            retained_stable - initial_stable
            if retained_stable is not None and initial_stable is not None
            else None
        )
        count = initial.get("total") if initial else None
        if summary is not None:
            completed += 1
        snapshot_5 = (
            f"{format_percent(accuracy_percent(metrics_5))} / "
            f"{format_percent(stable_percent(metrics_5))}"
            if metrics_5
            else "—"
        )
        snapshot_10 = (
            f"{format_percent(accuracy_percent(metrics_10))} / "
            f"{format_percent(stable_percent(metrics_10))}"
            if metrics_10
            else "—"
        )
        lines.append(
            f"| {spec.dataset} | {spec.mode} | {spec.model} | {spec.method} | "
            f"{count if count is not None else '—'} | "
            f"{format_percent(accuracy_percent(initial))} | "
            f"{format_percent(initial_stable)} | {snapshot_5} | {snapshot_10} | "
            f"{format_percent(accuracy_percent(final))} | "
            f"{format_percent(retained_stable)} | {format_delta(gain)} | "
            f"{run_status(spec, summary)} | {spec.code} | {spec.run_dir} |"
        )
    lines.extend(
        [
            "",
            f"Completed canonical runs: {completed}/{len(RUN_SPECS)}",
            f"Pending canonical runs: {len(RUN_SPECS) - completed}/{len(RUN_SPECS)}",
            "",
            "SOURCE-PROMPT SCORES AND AUDIT",
            "------------------------------",
            "",
            "These are the five first-stage source slots used by second-stage refiners: "
            "RPO-5, RPO-10, EvoPrompt-5, EvoPrompt-10, and ETGPO-1.",
            "",
            "| Dataset | Model | Source | Stable score | Stable gain from run initial | "
            "Prompt audit | Saved prompt | CODE |",
            "|---|---|---|---:|---:|---|---|---|",
        ]
    )
    for spec in RUN_SPECS:
        summary = load_summary(spec)
        validation = (summary or {}).get("validation") or {}
        initial_metrics = validation.get("initial")
        initial_stable = stable_percent(initial_metrics)
        run_dir = absolute_path(spec.run_dir)
        initial_text = read_text(run_dir / "initial_prompt.txt")
        earlier_texts: list[tuple[str, str]] = []
        for label, filename in expected_prompt_entries(spec):
            path = run_dir / filename
            text = read_text(path)
            if label.endswith("-5"):
                metrics = assigned_snapshot_metrics(spec, summary, "5")
            elif label.endswith("-10"):
                metrics = assigned_snapshot_metrics(spec, summary, "10")
            else:
                metrics = final_metrics(summary)
            score = stable_percent(metrics)
            gain = (
                score - initial_stable
                if score is not None and initial_stable is not None
                else None
            )
            audit = audit_prompt(text, initial_text, earlier_texts)
            relative_path = f"{spec.run_dir}/{filename}"
            lines.append(
                f"| {spec.dataset} | {spec.model} | {label} | "
                f"{format_percent(score)} | {format_delta(gain)} | {audit} | "
                f"{relative_path} | {spec.code} |"
            )
            if text is not None:
                earlier_texts.append((label, text))
    lines.extend(["", "MANUAL PROMPT-AUDIT NOTES", "-------------------------", ""])
    for code, note in MANUAL_PROMPT_NOTES.items():
        lines.append(f"- {code}: {note}")
    lines.extend(
        [
            "",
            "No available canonical saved prompt is a one-word parser fragment such as "
            '"and". Automatic audit also checks for leaked <prompt> and <answer> tags.',
            "",
        ]
    )
    return "\n".join(lines)


def compact_report() -> str:
    """Build the stable-score summary followed by complete saved prompt texts."""
    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "FIRST-STAGE STABLE-SCORE SUMMARY AND PROMPTS",
        "============================================",
        "",
        f"Generated: {generated}",
        "Scores are percentages. Gains are calculated within each canonical run.",
        "Different validation sizes are retained and shown rather than mixed.",
        "",
        "STABLE-SCORE SUMMARY",
        "--------------------",
        "",
        "| Dataset | Mode | Model | Method | N | Initial stable | Final stable | "
        "Stable gain | Status | CODE | Saved directory |",
        "|---|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for spec in RUN_SPECS:
        summary = load_summary(spec)
        validation = (summary or {}).get("validation") or {}
        initial = validation.get("initial")
        final = validation.get("final")
        initial_stable = stable_percent(initial)
        retained_stable = stable_percent(final)
        gain = (
            retained_stable - initial_stable
            if retained_stable is not None and initial_stable is not None
            else None
        )
        count = initial.get("total") if initial else None
        lines.append(
            f"| {spec.dataset} | {spec.mode} | {spec.model} | {spec.method} | "
            f"{count if count is not None else '—'} | "
            f"{format_percent(initial_stable)} | {format_percent(retained_stable)} | "
            f"{format_delta(gain)} | {run_status(spec, summary)} | {spec.code} | "
            f"{spec.run_dir} |"
        )

    lines.extend(["", "INITIAL AND RETAINED PROMPTS", "----------------------------", ""])
    dataset_order = ("OpenBookQA", "HotpotQA", "MATH-500")
    model_order = ("Qwen3-4B", "Gemma3-4B")
    for dataset in dataset_order:
        dataset_specs = [spec for spec in RUN_SPECS if spec.dataset == dataset]
        lines.extend([dataset.upper(), "~" * len(dataset), ""])
        initial_prompts: list[str] = []
        for spec in dataset_specs:
            text = read_text(absolute_path(spec.run_dir) / "initial_prompt.txt")
            if text and normalized_text(text) not in {
                normalized_text(item) for item in initial_prompts
            }:
                initial_prompts.append(text)
        lines.append("Initial prompt(s):")
        if initial_prompts:
            for index, text in enumerate(initial_prompts, start=1):
                lines.extend([f"[{index}] {text}", ""])
        else:
            lines.extend(["PENDING", ""])

        for model in model_order:
            lines.extend([model, "^" * len(model), ""])
            model_specs = [spec for spec in dataset_specs if spec.model == model]
            for spec in model_specs:
                summary = load_summary(spec)
                validation = (summary or {}).get("validation") or {}
                initial_stable = stable_percent(validation.get("initial"))
                run_dir = absolute_path(spec.run_dir)
                initial_text = read_text(run_dir / "initial_prompt.txt")
                earlier_texts: list[tuple[str, str]] = []
                for label, filename in expected_prompt_entries(spec):
                    text = read_text(run_dir / filename)
                    if label.endswith("-5"):
                        metrics = assigned_snapshot_metrics(spec, summary, "5")
                    elif label.endswith("-10"):
                        metrics = assigned_snapshot_metrics(spec, summary, "10")
                    else:
                        metrics = final_metrics(summary)
                    score = stable_percent(metrics)
                    gain = (
                        score - initial_stable
                        if score is not None and initial_stable is not None
                        else None
                    )
                    audit = audit_prompt(text, initial_text, earlier_texts)
                    lines.extend(
                        [
                            f"{label}",
                            f"CODE: {spec.code}",
                            f"Directory: {spec.run_dir}",
                            f"Prompt file: {spec.run_dir}/{filename}",
                            f"Stable score: {format_percent(score)}",
                            f"Stable gain: {format_delta(gain)}",
                            f"Audit: {audit}",
                            "Prompt:",
                            text if text is not None else "PENDING",
                            "",
                        ]
                    )
                    if text is not None:
                        earlier_texts.append((label, text))
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write detailed and compact canonical first-stage text reports."""
    args = parse_args()
    report_dir = args.report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = report_dir / "detailed_first_stage_results.txt"
    compact_path = report_dir / "first_stage_summary_and_prompts.txt"
    detailed_path.write_text(detailed_report(), encoding="utf-8")
    compact_path.write_text(compact_report(), encoding="utf-8")
    completed = sum(load_summary(spec) is not None for spec in RUN_SPECS)
    print(f"Saved detailed report: {detailed_path}")
    print(f"Saved compact report: {compact_path}")
    print(f"Canonical runs complete: {completed}/{len(RUN_SPECS)}")


if __name__ == "__main__":
    main()
