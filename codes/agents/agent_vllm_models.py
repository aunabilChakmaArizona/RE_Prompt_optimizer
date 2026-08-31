"""Load models and tokenizers for the optional vLLM inference backend."""

from __future__ import annotations

import importlib.metadata
import os
import re
from typing import Any


VLLM_REQUIREMENTS_PATH = "requirements_vllm.txt"
SUPPORTED_VLLM_VERSION = "0.8.5.post1"


def _select_visible_gpu(device: str | None) -> str | None:
    """Expose only the requested CUDA device before vLLM is imported."""
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

    os.environ["CUDA_VISIBLE_DEVICES"] = selected_device
    return selected_device


def _validate_vllm_version() -> str: #note: reduntant
    """Check that the CUDA-12.4-compatible vLLM version is installed."""
    try:
        installed_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "vLLM is not installed. Create the separate environment and run: "
            f"python -m pip install -r {VLLM_REQUIREMENTS_PATH}"
        ) from error

    if installed_version != SUPPORTED_VLLM_VERSION:
        raise RuntimeError(
            f"Expected vllm=={SUPPORTED_VLLM_VERSION}, but found vllm=={installed_version}. "
            f"Install the pinned dependencies from {VLLM_REQUIREMENTS_PATH}."
        )
    return installed_version


def load_vllm_model_and_tokenizer(
    model_id: str,
    device: str | None = None,
    gpu_memory_utilization: float = 0.90,
) -> tuple[Any, Any]:
    """Load one vLLM engine and return it with its tokenizer."""
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be greater than 0 and at most 1.")

    selected_device = _select_visible_gpu(device)
    installed_version = _validate_vllm_version()

    try:
        from vllm import LLM
    except ImportError as error:
        raise RuntimeError(
            f"Unable to import vLLM. Install {VLLM_REQUIREMENTS_PATH} in a clean environment."
        ) from error

    print(f"[agent_vllm_models] loading model: {model_id}")
    print(f"[agent_vllm_models] vLLM version: {installed_version}")
    if selected_device is not None:
        print(f"[agent_vllm_models] CUDA_VISIBLE_DEVICES: {selected_device}")

    model = LLM(
        model=model_id,
        dtype="bfloat16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    tokenizer = model.get_tokenizer()
    print(f"[agent_vllm_models] model loading done: {model_id}")
    return model, tokenizer


def vllm_backend_metadata() -> dict[str, str]:
    """Describe the pinned vLLM backend used by an inference run."""
    return {
        "name": "vllm",
        "version": _validate_vllm_version(),
        "batching": "continuous",
    }
