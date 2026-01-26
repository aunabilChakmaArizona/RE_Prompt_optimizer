from __future__ import annotations

import os
from typing import Tuple

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer


def _default_device_map() -> str:
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


def load_model_and_tokenizer(model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    device_map = _default_device_map()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=_default_dtype(device_map),
        trust_remote_code=True,
        device_map=device_map,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer
