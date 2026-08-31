"""Provide reproducible model-default decoding settings for generated answers."""

from __future__ import annotations

from typing import TypeAlias


DecodingValue: TypeAlias = float | int


def model_default_sampling_parameters(model_id: str) -> dict[str, DecodingValue]:
    """Return the paper's default sampling settings for Qwen3 or Gemma3."""
    normalized_model_id = model_id.casefold()
    if "qwen3" in normalized_model_id:
        return {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    if "gemma-3" in normalized_model_id or "gemma3" in normalized_model_id:
        return {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    raise ValueError(
        "No default sampling configuration is registered for "
        f"{model_id!r}. Expected a Qwen3 or Gemma3 model."
    )
