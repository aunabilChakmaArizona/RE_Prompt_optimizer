from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available


def _default_device_map(device_map: Optional[str]) -> str:
    if device_map:
        return device_map
    env_device = os.getenv("DEVICE_MAP") or os.getenv("CUDA_DEVICE")
    if env_device:
        return env_device
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _default_dtype(device_map: str) -> torch.dtype:
    if device_map != "cpu" and torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def _default_attn_implementation(device_map: str) -> str | None:
    if device_map == "cpu":
        return None
    if torch.cuda.is_available() and is_flash_attn_2_available():
        return "flash_attention_2"
    return None


def load_model_and_tokenizer(
    model_id: str, device_map: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    device_map = _default_device_map(device_map)
    attn_implementation = _default_attn_implementation(device_map)
    model_kwargs = {
        "torch_dtype": _default_dtype(device_map),
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
