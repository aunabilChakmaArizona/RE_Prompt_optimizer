"""Consistent output files and prompt loading for QA optimizer runs."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from prompt_optimization.qa_task import REPO_ROOT, QAMode


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "qa_prompt_optimization"


def safe_name(value: str) -> str:
    """Convert a run label into a filesystem-safe name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    if not cleaned:
        raise ValueError("Run CODE must contain at least one letter or number.")
    return cleaned


def create_run_directory(
    output_root: str | Path,
    optimizer_name: str,
    qa_mode: str,
    code: str,
    overwrite: bool,
) -> Path:
    """Create one protected output directory for an optimizer attempt."""
    root = Path(output_root).expanduser()
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    run_dir = root / safe_name(qa_mode) / safe_name(optimizer_name) / safe_name(code)
    protected_files = [run_dir / "summary.json", run_dir / "config.json"]
    if not overwrite and any(path.exists() for path in protected_files):
        raise FileExistsError(
            f"Optimizer output already exists in {run_dir}. Use a new --code or --overwrite."
        )
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: Path, payload: Any) -> None:
    """Write readable UTF-8 JSON and create its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Any) -> None:
    """Append one JSON object to a UTF-8 JSONL audit log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def save_text(path: Path, text: str) -> None:
    """Write normalized prompt text with one trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def load_initial_prompt(
    mode: QAMode,
    prompt_value: str | None,
    prompt_file: str | Path | None,
) -> str:
    """Load an instruction from a value, a text file, or the mode default."""
    if prompt_value and prompt_file:
        raise ValueError("Use only one of --initial-prompt and --initial-prompt-file.")
    if prompt_file:
        prompt = Path(prompt_file).expanduser().read_text(encoding="utf-8").strip()
    elif prompt_value:
        prompt = prompt_value.strip()
    else:
        prompt = mode.initial_prompt
    if not prompt:
        raise ValueError("The initial instruction prompt is empty.")
    return prompt


class RunLogger:
    """Write shared run artifacts as an optimizer progresses."""

    def __init__(self, run_dir: Path):
        """Initialize paths inside one run directory."""
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        self.candidates_path = run_dir / "candidates.jsonl"

    def event(self, event_type: str, **payload: Any) -> None:
        """Append one timestamp-free deterministic experiment event."""
        append_jsonl(self.events_path, {"event": event_type, **payload})

    def candidate(self, **payload: Any) -> None:
        """Append one candidate prompt and its metrics."""
        append_jsonl(self.candidates_path, payload)

    def evaluation(
        self,
        name: str,
        evaluation: dict[str, Any],
        save_predictions: bool = True,
    ) -> None:
        """Save an evaluation summary and optionally its per-example predictions."""
        directory = self.run_dir / "evaluations"
        summary = {key: value for key, value in evaluation.items() if key != "predictions"}
        save_json(directory / f"{safe_name(name)}_summary.json", summary)
        if save_predictions:
            path = directory / f"{safe_name(name)}_predictions.jsonl"
            for prediction in evaluation.get("predictions", []):
                append_jsonl(path, prediction)
