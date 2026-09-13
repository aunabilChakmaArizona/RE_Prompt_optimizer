"""Shared on-disk cache for repeated source-prompt validation results."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

from agents.agent_decoding import model_default_sampling_parameters
from prompt_optimization.qa_task import REPO_ROOT


SOURCE_VALIDATION_CACHE_VERSION = 1
DEFAULT_SOURCE_VALIDATION_CACHE_ROOT = (
    REPO_ROOT / "outputs" / "shared_source_validation_cache"
)


def _json_hash(payload: Any) -> str:
    """Hash a JSON-compatible value with stable key ordering."""
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _safe_component(value: str) -> str:
    """Convert one cache-path component into a filesystem-safe name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "unknown"


def _package_version(package_name: str) -> str:
    """Return an installed package version or an unavailable marker."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def resolve_source_validation_cache_root(path_value: str | Path) -> Path:
    """Resolve a source-validation cache directory from the repository root."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def build_source_validation_identity(
    *,
    model_id: str,
    generation_backend: str,
    task_name: str,
    mode_name: str,
    source_prompt: str,
    validation_records: Sequence[dict[str, Any]],
    rendered_prompts: Sequence[str],
    seed: int,
    max_new_tokens: int,
    enable_thinking: bool,
    validation_std_penalty: float,
) -> dict[str, Any]:
    """Describe every setting that must match before a source score is reused."""
    if len(validation_records) != len(rendered_prompts):
        raise ValueError("Validation records and rendered prompts must have equal lengths.")
    package_name = "vllm" if generation_backend == "vllm" else "transformers"
    return {
        "cache_version": SOURCE_VALIDATION_CACHE_VERSION,
        "model_id": model_id,
        "generation_backend": generation_backend,
        "generation_library_version": _package_version(package_name),
        "task_name": task_name,
        "mode_name": mode_name,
        "source_prompt": source_prompt,
        "source_prompt_hash": _json_hash(source_prompt),
        "record_ids": [str(record["id"]) for record in validation_records],
        "record_content_hash": _json_hash(list(validation_records)),
        "rendered_prompts_hash": _json_hash(list(rendered_prompts)),
        "validation_record_count": len(validation_records),
        "seed": int(seed),
        "generation_settings": {
            "max_new_tokens": int(max_new_tokens),
            "enable_thinking": bool(enable_thinking),
            "do_sample": True,
            **model_default_sampling_parameters(model_id),
        },
        "validation_std_penalty": float(validation_std_penalty),
    }


def source_validation_cache_path(
    cache_root: str | Path,
    identity: dict[str, Any],
) -> tuple[Path, str]:
    """Return the shared cache path and complete identity hash."""
    cache_key = _json_hash(identity)
    root = resolve_source_validation_cache_root(cache_root)
    path = (
        root
        / _safe_component(str(identity["task_name"]))
        / _safe_component(str(identity["mode_name"]))
        / _safe_component(str(identity["model_id"]))
        / str(identity["source_prompt_hash"])[:16]
        / f"{cache_key}.json"
    )
    return path, cache_key


@contextmanager
def lock_source_validation_cache(cache_path: Path) -> Iterator[None]:
    """Lock one source cache entry while it is checked or generated."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def load_source_validation_cache(
    cache_path: Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any] | None:
    """Load one matching source evaluation or return None when absent."""
    if not cache_path.is_file():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Source validation cache must be an object: {cache_path}")
    if payload.get("metadata") != expected_identity:
        raise ValueError(f"Source validation cache metadata mismatch: {cache_path}")
    if not isinstance(payload.get("evaluation"), dict):
        raise ValueError(f"Source validation cache lacks an evaluation: {cache_path}")
    return payload


def save_source_validation_cache(
    cache_path: Path,
    payload: dict[str, Any],
) -> None:
    """Atomically save one complete source-prompt validation evaluation."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, cache_path)
