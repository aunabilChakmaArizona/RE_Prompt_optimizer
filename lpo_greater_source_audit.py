#!/usr/bin/env python3
"""Audit original LPO/GreaTer sources for selected stage-2 prompt changes.

The script reads canonical records from ``all_the_prompts.txt``, keeps only
non-empty LPO, GreaTer, and GreaTer-TG outputs, resolves each declared SOURCE
log, follows the log's ``run_dir`` to JSON artifacts when they still exist,
and extracts validation scores and exact optimizer span counts.

Span semantics:
* LPO: number reported by ``location tagging: locations=N`` in the source log.
* GreaTer/GreaTer-TG: one accepted token-edit location per changed prompt.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


METHOD_RE = re.compile(r"_(?P<method>lpo|greater|greater-tg)$")
SCORE_RE = (
    r"P=(?P<precision>[\d.]+) \+/- (?P<precision_std>[\d.]+), "
    r"R=(?P<recall>[\d.]+) \+/- (?P<recall_std>[\d.]+), "
    r"F1=(?P<f1>[\d.]+) \+/- (?P<f1_std>[\d.]+) "
    r"stable_f1=(?P<stable_f1>[-\d.]+)"
)


@dataclass
class Record:
    code: str
    prompt: str
    source: str


@dataclass
class AuditRow:
    code: str
    base_code: str
    method: str
    source_declared: str
    source_log: str
    source_resolution: str
    run_dir_declared: str
    run_dir_resolved: str
    run_dir_exists: bool
    summary_json: str
    summary_exists: bool
    steps_jsonl: str
    steps_exists: bool
    prompt_found_in_log: bool
    score_source: str
    candidate_rank: str
    accepted_by_optimizer: str
    selection_note: str
    precision: str
    precision_std: str
    recall: str
    recall_std: str
    f1: str
    f1_std: str
    stable_f1: str
    spans_changed: int
    span_evidence: str
    selected_token: str


def normalize_prompt(text: str) -> str:
    return " ".join(text.split())


def parse_records(path: Path) -> Dict[str, Record]:
    text = path.read_text(encoding="utf-8")
    records: Dict[str, Record] = {}
    for chunk in re.split(r"(?=^###### CODE:)", text, flags=re.MULTILINE):
        code_match = re.search(
            r"^###### CODE:[ \t]*(.*?)[ \t]*$", chunk, flags=re.MULTILINE
        )
        if not code_match:
            continue
        status_match = re.search(r"^###### STATUS:.*?$", chunk, flags=re.MULTILINE)
        source_match = re.search(
            r"^###### SOURCE:[ \t]*(.*?)[ \t]*$", chunk, flags=re.MULTILINE
        )
        code = code_match.group(1).strip()
        records[code] = Record(
            code=code,
            prompt=chunk[status_match.end() :].strip() if status_match else "",
            source=source_match.group(1).strip() if source_match else "",
        )
    return records


def clean_source(source: str) -> str:
    return re.sub(r"\s*\((?:unchanged|marked)\)\s*$", "", source).strip()


def resolve_declared_log(root: Path, source: str) -> Optional[Path]:
    cleaned = clean_source(source)
    if not cleaned or cleaned == "skipped":
        return None
    for candidate in (root / cleaned, root / "codes" / cleaned):
        if candidate.is_file():
            return candidate.resolve()
    return None


def fallback_log_search(root: Path, prompt: str) -> Optional[Path]:
    """Find a moved/misnamed log by its selected prompt's first line."""
    first_line = prompt.splitlines()[0].strip()
    if not first_line:
        return None
    result = subprocess.run(
        [
            "rg",
            "-l",
            "-F",
            first_line,
            str(root / "codes" / "nohup_outs"),
            "-g",
            "*.txt",
            "-g",
            "*.out",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    candidates = [Path(line) for line in result.stdout.splitlines() if line.strip()]
    normalized = normalize_prompt(prompt)
    for candidate in candidates:
        log_text = candidate.read_text(encoding="utf-8", errors="replace")
        if (
            "[relation_extraction_greater]" in log_text
            and normalized in normalize_prompt(log_text)
        ):
            return candidate.resolve()
    return candidates[0].resolve() if len(candidates) == 1 else None


def resolve_run_dir(root: Path, log_text: str) -> tuple[str, Optional[Path]]:
    match = re.search(r"run_dir:\s*(\S+)", log_text)
    if not match:
        return "", None
    declared = match.group(1)
    # Experiments were launched from root/codes, so ../ points back to root.
    resolved = (root / "codes" / declared).resolve()
    return declared, resolved


def score_values(match: Optional[re.Match[str]]) -> Dict[str, str]:
    keys = (
        "precision",
        "precision_std",
        "recall",
        "recall_std",
        "f1",
        "f1_std",
        "stable_f1",
    )
    return {key: match.group(key) if match else "" for key in keys}


def extract_lpo(log_text: str, prompt: str) -> tuple[Dict[str, str], int, bool, str]:
    locations = [
        int(value)
        for value in re.findall(r"location tagging: locations=(\d+)", log_text)
    ]
    span_count = locations[-1] if locations else 0

    candidate_pattern = re.compile(
        r"full eval candidate: rank=(?P<rank>\d+) candidate_index=\d+.*?"
        r"full eval candidate prompt BEGIN\n(?P<prompt>.*?)\n"
        r"\[relation_extraction_lpo\] full eval candidate prompt END.*?"
        r"dev candidate: rank=(?P=rank) " + SCORE_RE,
        flags=re.DOTALL,
    )
    target = normalize_prompt(prompt)
    candidates = list(candidate_pattern.finditer(log_text))
    matched = next(
        (item for item in candidates if normalize_prompt(item.group("prompt")) == target),
        None,
    )
    if matched:
        return score_values(matched), span_count, True, "matching dev candidate"

    current_match = re.search(
        r"current prompt after selection:\n(?P<prompt>.*?)"
        r"(?=\n\[relation_extraction_lpo\] step result:)",
        log_text,
        flags=re.DOTALL,
    )
    prompt_found = bool(
        current_match
        and normalize_prompt(current_match.group("prompt")).removeprefix("prompt ")
        == target
    )
    # LPO selects by stable dev F1. This fallback is useful for one log whose
    # printed current prompt has a stray leading literal "prompt".
    best = max(
        candidates,
        key=lambda item: float(item.group("stable_f1")),
        default=None,
    )
    return (
        score_values(best),
        span_count,
        prompt_found,
        "highest stable-F1 dev candidate fallback" if best else "score unavailable",
    )


def extract_greater(
    log_text: str, prompt: str
) -> tuple[Dict[str, str], bool, str, str]:
    selected = list(
        re.finditer(
            r"top-z selected prompt: step=(?P<step>\d+) rank=(?P<rank>\d+) "
            r"token=(?P<token>.+?) stable_f1=",
            log_text,
        )
    )
    if not selected:
        return score_values(None), False, "score unavailable", ""
    choice = selected[-1]
    score_match = re.search(
        rf"top-z dev score: step={choice.group('step')} "
        rf"rank={choice.group('rank')} " + SCORE_RE,
        log_text,
    )
    selected_prompt = re.search(
        r"\[Prompt selected by top-z dev\] prompt = (?P<prompt>.*?)"
        r"(?=\n(?:\[scores\]|\[Prompt after step|\[relation_extraction_greater\] "
        r"(?:balanced train after step|step result)))",
        log_text,
        flags=re.DOTALL,
    )
    prompt_found = bool(
        selected_prompt
        and normalize_prompt(selected_prompt.group("prompt")) == normalize_prompt(prompt)
    )
    return (
        score_values(score_match),
        prompt_found,
        "selected top-z dev candidate" if score_match else "score unavailable",
        choice.group("token").strip(),
    )


def extract_greater_artifact(
    steps_path: Optional[Path], prompt: str
) -> Optional[tuple[Dict[str, str], bool, str, str, str, str, str]]:
    """Match the stored prompt to an evaluated candidate in steps.jsonl."""
    if not steps_path or not steps_path.is_file():
        return None
    steps = [
        json.loads(line)
        for line in steps_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    target = normalize_prompt(prompt)
    for step in reversed(steps):
        for evaluation in step.get("top_z_evaluations", []):
            if normalize_prompt(evaluation.get("prompt", "")) != target:
                continue
            prf = evaluation.get("full_evaluation", {}).get("prf", {})
            scores = {
                "precision": str(prf.get("precision", "")),
                "precision_std": str(prf.get("precision_std", "")),
                "recall": str(prf.get("recall", "")),
                "recall_std": str(prf.get("recall_std", "")),
                "f1": str(prf.get("f1", "")),
                "f1_std": str(prf.get("f1_std", "")),
                "stable_f1": str(evaluation.get("stable_f1", "")),
            }
            accepted = bool(step.get("accepted"))
            selected = step.get("top_z_selected", {})
            is_selected = (
                evaluation.get("rank") == selected.get("rank")
                and normalize_prompt(evaluation.get("prompt", ""))
                == normalize_prompt(selected.get("prompt", ""))
            )
            if accepted and is_selected:
                note = "stored prompt is the accepted top-z selection"
            elif is_selected:
                note = "stored prompt is selected no-op/rejected candidate"
            else:
                note = "stored prompt is an evaluated but non-selected top-z candidate"
            return (
                scores,
                True,
                "steps.jsonl matching top-z evaluation",
                str(evaluation.get("token_text", "")).strip(),
                str(evaluation.get("rank", "")),
                str(accepted),
                note,
            )
    return None


def build_rows(root: Path, records: Dict[str, Record]) -> Iterable[AuditRow]:
    for code, record in records.items():
        method_match = METHOD_RE.search(code)
        if not method_match or not record.prompt:
            continue
        method = method_match.group("method")
        log_path = resolve_declared_log(root, record.source)
        resolution = "declared"
        if log_path is None:
            log_path = fallback_log_search(root, record.prompt)
            resolution = "recovered by selected-prompt search" if log_path else "missing"
        log_text = (
            log_path.read_text(encoding="utf-8", errors="replace") if log_path else ""
        )
        run_declared, run_path = resolve_run_dir(root, log_text)
        summary_path = run_path / "summary.json" if run_path else None
        steps_path = run_path / "steps.jsonl" if run_path else None

        selected_token = ""
        candidate_rank = ""
        accepted_by_optimizer = ""
        selection_note = ""
        if method == "lpo":
            scores, spans, prompt_found, score_source = extract_lpo(
                log_text, record.prompt
            )
            span_evidence = "source log: location tagging locations=N"
            accepted_matches = re.findall(r"step result: accepted=(True|False)", log_text)
            accepted_by_optimizer = accepted_matches[-1] if accepted_matches else ""
            selection_note = "stored prompt selected by LPO source workflow"
        else:
            artifact_result = extract_greater_artifact(steps_path, record.prompt)
            if artifact_result:
                (
                    scores,
                    prompt_found,
                    score_source,
                    selected_token,
                    candidate_rank,
                    accepted_by_optimizer,
                    selection_note,
                ) = artifact_result
            else:
                scores, prompt_found, score_source, selected_token = extract_greater(
                    log_text, record.prompt
                )
                selected_match = re.search(
                    r"top-z selected prompt: step=\d+ rank=(\d+).*?improved=(True|False)",
                    log_text,
                )
                if selected_match:
                    candidate_rank = selected_match.group(1)
                    accepted_by_optimizer = selected_match.group(2)
                    selection_note = "stored prompt matched selected candidate in source log"
            spans = 1
            span_evidence = (
                "algorithm/artifact: one evaluated token location; "
                "acceptance reported separately"
            )

        yield AuditRow(
            code=code,
            base_code=METHOD_RE.sub("", code),
            method=method,
            source_declared=record.source,
            source_log=str(log_path.relative_to(root)) if log_path else "",
            source_resolution=resolution,
            run_dir_declared=run_declared,
            run_dir_resolved=str(run_path.relative_to(root)) if run_path else "",
            run_dir_exists=bool(run_path and run_path.is_dir()),
            summary_json=(str(summary_path.relative_to(root)) if summary_path else ""),
            summary_exists=bool(summary_path and summary_path.is_file()),
            steps_jsonl=(str(steps_path.relative_to(root)) if steps_path else ""),
            steps_exists=bool(steps_path and steps_path.is_file()),
            prompt_found_in_log=prompt_found,
            score_source=score_source,
            candidate_rank=candidate_rank,
            accepted_by_optimizer=accepted_by_optimizer,
            selection_note=selection_note,
            spans_changed=spans,
            span_evidence=span_evidence,
            selected_token=selected_token,
            **scores,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-file", type=Path, default=Path("all_the_prompts.txt"))
    parser.add_argument(
        "--output", type=Path, default=Path("lpo_greater_changed_source_audit.csv")
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    rows = list(build_rows(root, parse_records(args.prompts_file)))
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AuditRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    print(f"Changed LPO/GreaTer records: {len(rows)}")
    for method in ("lpo", "greater", "greater-tg"):
        selected = [row for row in rows if row.method == method]
        print(
            f"{method}: prompts={len(selected)}, "
            f"spans={sum(row.spans_changed for row in selected)}, "
            f"avg_spans={sum(row.spans_changed for row in selected) / len(selected):.2f}"
        )
    print(f"Declared logs resolved: {sum(row.source_resolution == 'declared' for row in rows)}")
    print(
        "Moved/misnamed logs recovered: "
        f"{sum(row.source_resolution.startswith('recovered') for row in rows)}"
    )
    print(f"Prompts verified in logs: {sum(row.prompt_found_in_log for row in rows)}")
    print(f"Scores recovered: {sum(bool(row.f1) for row in rows)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
