"""Target and optimizer model lifecycle management for QA prompt optimization."""

from __future__ import annotations

import gc
import random
from typing import Any, Sequence

from agents.agent_decoding import model_default_sampling_parameters


TARGET_ROLE = "target"
OPTIMIZER_ROLE = "optimizer"


def seed_everything(seed: int) -> None:
    """Seed Python and Torch before model generation or sampling."""
    random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ModelPool:
    """Load target and optimizer models with Transformers or vLLM."""

    def __init__(
        self,
        *,
        target_model_id: str,
        optimizer_model_id: str | None,
        target_device: str | None,
        optimizer_device: str | None,
        keep_models_loaded: bool,
        seed: int,
        backend: str = "transformers",
        gpu_memory_utilization: float = 0.90,
    ):
        """Store role settings without loading a model yet."""
        if backend not in {"transformers", "vllm"}:
            raise ValueError(f"Unsupported model backend: {backend!r}")
        if not 0.0 < gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be greater than 0 and at most 1.")
        self.model_ids = {
            TARGET_ROLE: target_model_id,
            OPTIMIZER_ROLE: optimizer_model_id or target_model_id,
        }
        self.device_maps = {
            TARGET_ROLE: target_device,
            OPTIMIZER_ROLE: optimizer_device or target_device,
        }
        self.keep_models_loaded = keep_models_loaded
        self.seed = seed
        self.backend = backend
        self.gpu_memory_utilization = gpu_memory_utilization
        self.generation_call_index = 0
        self.loaded: dict[str, tuple[Any, Any]] = {}

    def ensure(self, role: str) -> tuple[Any, Any]:
        """Return the requested role model, loading or switching when required."""
        if role not in self.model_ids:
            raise ValueError(f"Unknown model role: {role!r}")
        if role in self.loaded:
            return self.loaded[role]

        requested_id = self.model_ids[role]
        for loaded_role, pair in list(self.loaded.items()):
            if self.model_ids[loaded_role] == requested_id:
                self.loaded[role] = pair
                return pair

        if not self.keep_models_loaded:
            self.unload_all()
        seed_everything(self.seed)
        if self.backend == "vllm":
            from agents.agent_vllm_models import load_vllm_model_and_tokenizer

            pair = load_vllm_model_and_tokenizer(
                requested_id,
                device=self.device_maps[role],
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
        else:
            from agents.agent_models import load_model_and_tokenizer

            pair = load_model_and_tokenizer(
                requested_id,
                device_map=self.device_maps[role],
            )
        self.loaded[role] = pair
        return pair

    def generate(
        self,
        role: str,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        batch_size: int,
        enable_thinking: bool,
        do_sample: bool = True,
        log_label: str | None = None,
        return_token_usage: bool = False,
        seed: int | None = None,
    ):
        """Generate text with the paper's fixed Qwen3 or Gemma3 decoding settings."""
        model, tokenizer = self.ensure(role)
        if self.backend == "vllm":
            from agents.agent_vllm_prompting import run_prompts_vllm

            effective_seed = seed
            if effective_seed is None:
                effective_seed = self.seed + self.generation_call_index * 1_000_003
            self.generation_call_index += 1
            return run_prompts_vllm(
                list(prompts),
                model_id=self.model_ids[role],
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                enable_thinking=enable_thinking,
                do_sample=do_sample,
                do_log=True,
                log_label=log_label,
                return_token_usage=return_token_usage,
                seed=effective_seed,
            )

        from agents.agent_llm_prompting import run_prompts

        decoding = model_default_sampling_parameters(self.model_ids[role])
        return run_prompts(
            list(prompts),
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            enable_thinking=enable_thinking,
            do_sample=do_sample,
            do_log=True,
            log_label=log_label,
            return_token_usage=return_token_usage,
            **decoding,
        )

    def unload_role(self, role: str) -> None:
        """Release one role while preserving a shared model used by another role."""
        pair = self.loaded.pop(role, None)
        if pair is None:
            return
        if any(other_pair is pair for other_pair in self.loaded.values()):
            return
        from agents.agent_memory import clear_model_memory

        del pair
        gc.collect()
        clear_model_memory()

    def unload_all(self) -> None:
        """Release every loaded model and clear CUDA allocator caches."""
        from agents.agent_memory import clear_model_memory

        self.loaded.clear()
        gc.collect()
        clear_model_memory()

    def close(self) -> None:
        """Release all model resources at the end of a run."""
        self.unload_all()
