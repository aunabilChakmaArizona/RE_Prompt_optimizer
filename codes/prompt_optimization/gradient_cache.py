"""Shared on-disk training-response cache for LPO and gradient refiners."""

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
from typing import Any, Callable, Sequence

from prompt_optimization.qa_task import REPO_ROOT


GRADIENT_CACHE_VERSION = 1
DEFAULT_GRADIENT_CACHE_ROOT = REPO_ROOT / "outputs" / "shared_gradient_cache"


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
    """Convert one cache-path component into a short filesystem-safe name."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or "unknown"


def _package_version(package_name: str) -> str:
    """Return an installed package version or an explicit unavailable marker."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def resolve_gradient_cache_root(path_value: str | Path) -> Path:
    """Resolve a gradient-cache directory relative to the repository root."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def build_gradient_pool_identity(
    *,
    model_id: str,
    generation_backend: str,
    task_name: str,
    mode_name: str,
    source_prompt: str,
    pool_records: Sequence[dict[str, Any]],
    rendered_prompts: Sequence[str],
    pool_seed: int,
    requested_pool_size: int,
    max_new_tokens: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    """Describe every setting that must match before responses can be reused."""
    if len(pool_records) != len(rendered_prompts):
        raise ValueError("Pool records and rendered prompts must have equal lengths.")
    package_name = "vllm" if generation_backend == "vllm" else "transformers"
    return {
        "cache_version": GRADIENT_CACHE_VERSION,
        "model_id": model_id,
        "generation_backend": generation_backend,
        "generation_library_version": _package_version(package_name),
        "task_name": task_name,
        "mode_name": mode_name,
        "source_prompt": source_prompt,
        "source_prompt_hash": _json_hash(source_prompt),
        "pool_seed": int(pool_seed),
        "requested_pool_size": int(requested_pool_size),
        "actual_pool_size": len(pool_records),
        "record_ids": [str(record["id"]) for record in pool_records],
        "record_content_hash": _json_hash(list(pool_records)),
        "rendered_prompts_hash": _json_hash(list(rendered_prompts)),
        "generation_settings": {
            "max_new_tokens": int(max_new_tokens),
            "enable_thinking": bool(enable_thinking),
            "do_sample": False,
            "decoding_mode": "greedy",
        },
    }


def gradient_pool_cache_path(
    cache_root: str | Path,
    identity: dict[str, Any],
) -> tuple[Path, str]:
    """Return the shared cache JSON path and its complete identity hash."""
    cache_key = _json_hash(identity)
    root = resolve_gradient_cache_root(cache_root)
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
def lock_gradient_cache(cache_path: Path) -> Iterator[None]:
    """Lock one cache key so parallel optimizer runs generate it only once."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)


def load_gradient_cache(
    cache_path: Path,
    expected_identity: dict[str, Any],
) -> dict[str, Any] | None:
    """Load one matching cache JSON, returning None when it does not exist."""
    if not cache_path.is_file():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Gradient cache must contain a JSON object: {cache_path}")
    if payload.get("metadata") != expected_identity:
        raise ValueError(f"Gradient cache metadata does not match its key: {cache_path}")
    if not isinstance(payload.get("pool_results"), list):
        raise ValueError(f"Gradient cache is missing pool_results: {cache_path}")
    if not isinstance(payload.get("balanced_subsets", {}), dict):
        raise ValueError(f"Gradient cache has invalid balanced_subsets: {cache_path}")
    payload.setdefault("balanced_subsets", {})
    return payload


def find_compatible_pool_cache(
    cache_root: str | Path,
    expected_identity: dict[str, Any],
    training_records: Sequence[dict[str, Any]],
    render_prompt: Callable[[dict[str, Any]], str],
) -> tuple[Path, dict[str, Any]] | None:
    """Find an exact or larger matching pool, verifying current record contents."""
    exact_path, _ = gradient_pool_cache_path(cache_root, expected_identity)
    size_fields = {
        "requested_pool_size", "actual_pool_size", "record_ids",
        "record_content_hash", "rendered_prompts_hash",
    }
    fixed_identity = {
        key: value for key, value in expected_identity.items() if key not in size_fields
    }
    records_by_id = {str(record["id"]): record for record in training_records}
    if len(records_by_id) != len(training_records):
        raise ValueError("Training records must have unique IDs for shared caching.")
    compatible = []
    for path in sorted(exact_path.parent.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if path == exact_path:
                raise
            continue
        if not isinstance(payload, dict):
            continue
        identity = payload.get("metadata", {})
        if not isinstance(identity, dict):
            continue
        if {key: value for key, value in identity.items() if key not in size_fields} != fixed_identity:
            continue
        ids = identity.get("record_ids", [])
        if (
            len(ids) < expected_identity["actual_pool_size"]
            or len(ids) != identity.get("actual_pool_size")
            or len(set(ids)) != len(ids)
            or any(str(record_id) not in records_by_id for record_id in ids)
        ):
            continue
        records = [records_by_id[str(record_id)] for record_id in ids]
        reconstructed = build_gradient_pool_identity(
            model_id=identity["model_id"],
            generation_backend=identity["generation_backend"],
            task_name=identity["task_name"],
            mode_name=identity["mode_name"],
            source_prompt=identity["source_prompt"],
            pool_records=records,
            rendered_prompts=[render_prompt(record) for record in records],
            pool_seed=identity["pool_seed"],
            requested_pool_size=identity["requested_pool_size"],
            max_new_tokens=identity["generation_settings"]["max_new_tokens"],
            enable_thinking=identity["generation_settings"]["enable_thinking"],
        )
        if reconstructed != identity:
            continue
        rows = payload.get("pool_results", [])
        if not isinstance(rows, list) or len(rows) != len(ids):
            continue
        if any(not isinstance(row, dict) for row in rows):
            continue
        if [str(row.get("record_id")) for row in rows] != [str(record_id) for record_id in ids]:
            continue
        if any(not isinstance(row.get("raw_response"), str) for row in rows):
            continue
        compatible.append((path != exact_path, len(ids), str(path), path, identity))
    if not compatible:
        return None
    selected = min(compatible, key=lambda item: item[:3])
    return selected[3], selected[4]


def save_gradient_cache(cache_path: Path, payload: dict[str, Any]) -> None:
    """Atomically write one readable shared gradient-cache JSON file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.tmp"
    )
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, cache_path)


def prediction_outcome_hash(predictions: Sequence[dict[str, Any]]) -> str:
    """Hash graded outcomes so stale balanced subsets are never reused."""
    outcomes = [
        {
            "id": str(prediction["id"]),
            "correct": bool(prediction["correct"]),
            "gold_answer": prediction.get("gold_answer"),
            "predicted_answer": prediction.get("predicted_answer"),
        }
        for prediction in predictions
    ]
    return _json_hash(outcomes)


def balanced_subset_cache_key(
    *,
    gradient_sample_size: int,
    balance_seed: int,
    outcome_hash: str,
) -> str:
    """Identify one deterministic balanced selection from cached outcomes."""
    return _json_hash(
        {
            "gradient_sample_size": int(gradient_sample_size),
            "balance_seed": int(balance_seed),
            "outcome_hash": outcome_hash,
        }
    )
