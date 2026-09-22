"""Load models and tokenizers for the optional vLLM inference backend."""

from __future__ import annotations

import importlib.metadata
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


VLLM_REQUIREMENTS_PATH = "requirements_vllm.txt"
SUPPORTED_VLLM_VERSION = "0.8.5.post1"
SUPPORTED_VLLM_VERSIONS = (SUPPORTED_VLLM_VERSION, "0.11.0")


def _resolve_visible_gpu(device: str | None) -> str | None:
    """Resolve a logical CUDA index to its physical visible-device value."""
    if device is None or device == "cuda":
        return os.getenv("CUDA_VISIBLE_DEVICES")

    match = re.fullmatch(r"cuda:(\d+)", device)
    if not match:
        raise ValueError(
            "The vLLM backend requires --device cuda:N, such as --device cuda:0."
        )

    logical_index = int(match.group(1))
    existing_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if existing_devices:
        visible_devices = [value.strip() for value in existing_devices.split(",")]
        if logical_index >= len(visible_devices):
            raise ValueError(
                f"{device} is unavailable within CUDA_VISIBLE_DEVICES={existing_devices!r}."
            )
        selected_device = visible_devices[logical_index]
    else:
        selected_device = str(logical_index)

    return selected_device


@contextmanager
def _temporary_visible_gpu(device: str | None) -> Iterator[str | None]: #aunabil3rd: interesting!!
    """Expose one GPU while vLLM starts, then restore the parent's GPU list."""
    original_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected_device = _resolve_visible_gpu(device)
    if selected_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_device
    try:
        yield selected_device
    finally:
        if original_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_devices


def _validate_vllm_version() -> str: #note: reduntant
    """Accept the original environment or the separately pinned v2 environment."""
    try:
        installed_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "vLLM is not installed. Create the separate environment and run: "
            f"python -m pip install -r {VLLM_REQUIREMENTS_PATH}"
        ) from error

    if installed_version not in SUPPORTED_VLLM_VERSIONS:
        raise RuntimeError(
            f"Supported vLLM versions are {SUPPORTED_VLLM_VERSIONS}, "
            f"but found vllm=={installed_version}. Install requirements_vllm.txt "
            "for the original environment or requirements_vllm_v2.txt for v2."
        )
    return installed_version


def load_vllm_model_and_tokenizer(
    model_id: str,
    device: str | None = None,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int | None = None,
    disable_image_inputs: bool = False,
    enforce_eager: bool = False,
    max_num_batched_tokens: int | None = None,
    max_num_seqs: int | None = None,
    enable_prefix_caching: bool | None = None,
) -> tuple[Any, Any]:
    """Load one vLLM engine and return it with its tokenizer."""
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be greater than 0 and at most 1.")
    if max_model_len is not None and max_model_len <= 0:
        raise ValueError("max_model_len must be positive when provided.")
    if max_num_batched_tokens is not None and max_num_batched_tokens <= 0:
        raise ValueError("max_num_batched_tokens must be positive when provided.")
    if max_num_seqs is not None and max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive when provided.")

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    with _temporary_visible_gpu(device) as selected_device:
        installed_version = _validate_vllm_version()

        if (
            installed_version == "0.11.0"
            and "gemma-3" in model_id.lower()
            and disable_image_inputs
        ):
            raise ValueError(
                "Do not disable image inputs for Gemma 3 with vLLM 0.11.0. "
                "The text-only limit_mm_per_prompt configuration produced "
                "empty and corrupted generations in this project."
            )

        try:
            from vllm import LLM
        except ImportError as error:
            raise RuntimeError(
                f"Unable to import vLLM. Install {VLLM_REQUIREMENTS_PATH} "
                "in a clean environment."
            ) from error

        print(f"[agent_vllm_models] loading model: {model_id}")
        print(f"[agent_vllm_models] vLLM version: {installed_version}")
        if selected_device is not None:
            print(
                "[agent_vllm_models] vLLM startup CUDA_VISIBLE_DEVICES: "
                f"{selected_device}"
            )
        if disable_image_inputs:
            print("[agent_vllm_models] image inputs disabled for text-only inference")

        model_options = {
            "model": model_id,
            "dtype": "bfloat16",
            "trust_remote_code": True,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
        }
        if max_model_len is not None:
            model_options["max_model_len"] = max_model_len
        if disable_image_inputs:
            model_options["limit_mm_per_prompt"] = {"image": 0}
        if max_num_batched_tokens is not None:
            model_options["max_num_batched_tokens"] = max_num_batched_tokens
        if max_num_seqs is not None:
            model_options["max_num_seqs"] = max_num_seqs
        if enable_prefix_caching is not None:
            model_options["enable_prefix_caching"] = enable_prefix_caching
        model = LLM(**model_options)
        tokenizer = model.get_tokenizer()
    print(f"[agent_vllm_models] model loading done: {model_id}")
    print(
        "[agent_vllm_models] restored parent CUDA_VISIBLE_DEVICES: "
        f"{os.getenv('CUDA_VISIBLE_DEVICES')}"
    )
    return model, tokenizer


def shutdown_vllm_model(model: Any) -> None:
    """Stop the vLLM engine process so its GPU memory is released."""
    llm_engine = getattr(model, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    shutdown = getattr(engine_core, "shutdown", None)
    if callable(shutdown):
        print("[agent_vllm_models] shutting down vLLM engine")
        shutdown()
        print("[agent_vllm_models] vLLM engine shutdown done")


def vllm_backend_metadata() -> dict[str, str]:
    """Describe the pinned vLLM backend used by an inference run."""
    return {
        "name": "vllm",
        "version": _validate_vllm_version(),
        "batching": "continuous",
    }
