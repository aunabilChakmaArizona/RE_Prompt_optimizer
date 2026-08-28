#!/usr/bin/env python3
"""Recover exact GradPO span counts for non-empty canonical prompt records."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional


METHOD_RE = re.compile(r"_(?P<method>gradpo-gen|gradpo-prob)$")


@dataclass
class PromptRecord:
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
    run_dir: str
    summary_json: str
    summary_exists: bool
    prompt_verified: bool
    changed_spans: int
    span_source: str
    generation_index: str
    beam_index: str
    selected_by_optimizer: bool
    selection_note: str


def norm(text: str) -> str:
    return " ".join(text.split())


def parse_records(path: Path) -> Dict[str, PromptRecord]:
    records: Dict[str, PromptRecord] = {}
    text = path.read_text(encoding="utf-8")
    for chunk in re.split(r"(?=^###### CODE:)", text, flags=re.MULTILINE):
        cm = re.search(r"^###### CODE:[ \t]*(.*?)[ \t]*$", chunk, re.MULTILINE)
        if not cm:
            continue
        sm = re.search(r"^###### STATUS:.*?$", chunk, re.MULTILINE)
        src = re.search(r"^###### SOURCE:[ \t]*(.*?)[ \t]*$", chunk, re.MULTILINE)
        code = cm.group(1).strip()
        records[code] = PromptRecord(
            code,
            chunk[sm.end() :].strip() if sm else "",
            src.group(1).strip() if src else "",
        )
    return records


def source_filename(source: str) -> str:
    value = re.sub(r"\s*\((?:unchanged|marked)\)\s*$", "", source).strip()
    return re.sub(r"\s+node\s+\d+\s*$", "", value).strip()


def source_node_hint(source: str) -> Optional[int]:
    match = re.search(r"\snode\s+(\d+)\s*$", re.sub(r"\s*\([^)]*\)\s*$", "", source))
    return int(match.group(1)) if match else None


def resolve_log(root: Path, source: str) -> Optional[Path]:
    filename = source_filename(source)
    for candidate in (
        root / filename,
        root / "codes" / filename,
        root / "codes" / "nohup_outs" / filename,
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_run(root: Path, log_text: str) -> tuple[str, Optional[Path]]:
    matches = list(re.finditer(r"saved run directory to\s+(\S+)", log_text))
    if not matches:
        return "", None
    declared = matches[-1].group(1)
    return declared, (root / "codes" / declared).resolve()


def summary_match(
    summary_path: Path, target: str
) -> Optional[tuple[int, str, str, bool, str]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    target_norm = norm(target)
    matches = []
    for iteration in data.get("iterations", []):
        selected = iteration.get("selected_prompt", {})
        selected_prompt_match = norm(selected.get("prompt", "")) == target_norm
        selected_beam = selected.get("beam_index")
        selected_generation = selected.get("generation_index")
        for step in iteration.get("beam_search_steps", []):
            for node in step.get("expanded_unique_nodes", []):
                if norm(node.get("prompt", "")) != target_norm:
                    continue
                is_selected = bool(
                    selected_prompt_match
                    and node.get("beam_index") == selected_beam
                )
                matches.append(
                    (
                        is_selected,
                        int(step.get("step_index", 0)),
                        int(node.get("num_changed_spans", 0)),
                        str(selected_generation if is_selected else ""),
                        str(node.get("beam_index", "")),
                        bool(selected.get("changed_vs_input")) if is_selected else False,
                    )
                )
    if not matches:
        return None
    # Prefer the selected beam; within a beam, use its latest region step.
    chosen = max(matches, key=lambda item: (item[0], item[1]))
    selected = chosen[0] and chosen[5]
    note = (
        "stored prompt is optimizer-selected"
        if selected
        else "stored prompt is a retained/evaluated but non-selected candidate"
    )
    return chosen[2], chosen[3], chosen[4], selected, note


def log_match(
    log_text: str, source: str
) -> tuple[int, str, str, bool, str]:
    selections = list(
        re.finditer(
            r"selected prompt for next iteration:.*?"
            r"generation_index=(?P<generation>\w+) "
            r"beam_index=(?P<beam>\w+).*?changed=(?P<changed>\w+)",
            log_text,
        )
    )
    hint = source_node_hint(source)
    if hint is not None:
        beam_index = hint
        generation = str(hint)
        selected = bool(
            selections
            and selections[-1].group("beam") == str(hint)
            and selections[-1].group("changed") == "True"
        )
    elif selections and selections[-1].group("beam") != "None":
        choice = selections[-1]
        beam_index = int(choice.group("beam"))
        generation = choice.group("generation")
        selected = choice.group("changed") == "True"
    else:
        raise ValueError("No selected beam or source node hint")

    nodes = [
        (int(m.group("step")), int(m.group("spans")))
        for m in re.finditer(
            rf"\[beam_node\] step_index=(?P<step>\d+) "
            rf"beam_index={beam_index}\b.*?num_changed_spans=(?P<spans>\d+)",
            log_text,
        )
    ]
    if not nodes:
        raise ValueError(f"No beam nodes for beam_index={beam_index}")
    spans = max(nodes, key=lambda item: item[0])[1]
    note = (
        "stored prompt is optimizer-selected"
        if selected
        else "stored prompt identified by SOURCE node hint, not final selection"
    )
    return spans, generation, str(beam_index), selected, note


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts-file", type=Path, default=Path("all_the_prompts.txt"))
    parser.add_argument(
        "--output", type=Path, default=Path("gradpo_changed_span_audit.csv")
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    rows = []
    for record in parse_records(args.prompts_file).values():
        mm = METHOD_RE.search(record.code)
        if not mm or not record.prompt:
            continue
        log_path = resolve_log(root, record.source)
        if not log_path:
            raise FileNotFoundError(f"Cannot resolve source for {record.code}: {record.source}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        run_declared, run_path = resolve_run(root, log_text)
        summary_path = run_path / "summary.json" if run_path else None
        result = summary_match(summary_path, record.prompt) if summary_path and summary_path.is_file() else None
        if result:
            spans, generation, beam, selected, note = result
            span_source = "summary.json matching beam node"
        else:
            spans, generation, beam, selected, note = log_match(log_text, record.source)
            span_source = "source log selected/hinted beam_node"
        prompt_verified = norm(record.prompt) in norm(log_text)
        rows.append(
            AuditRow(
                code=record.code,
                base_code=METHOD_RE.sub("", record.code),
                method=mm.group("method"),
                source_declared=record.source,
                source_log=str(log_path.relative_to(root)),
                run_dir=run_declared,
                summary_json=(str(summary_path.relative_to(root)) if summary_path else ""),
                summary_exists=bool(summary_path and summary_path.is_file()),
                prompt_verified=prompt_verified,
                changed_spans=spans,
                span_source=span_source,
                generation_index=generation,
                beam_index=beam,
                selected_by_optimizer=selected,
                selection_note=note,
            )
        )

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AuditRow.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    print(f"Changed GradPO prompts: {len(rows)}")
    for method in ("gradpo-gen", "gradpo-prob"):
        selected = [row for row in rows if row.method == method]
        total = sum(row.changed_spans for row in selected)
        print(f"{method}: prompts={len(selected)} spans={total} avg={total/len(selected):.2f}")
    print(f"Prompts present in source logs: {sum(row.prompt_verified for row in rows)}")
    print(f"Optimizer-selected: {sum(row.selected_by_optimizer for row in rows)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
