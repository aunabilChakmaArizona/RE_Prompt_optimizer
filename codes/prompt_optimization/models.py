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


class ModelPool: #aunabil3rd: just a question to ask: what is difference between the agent_models and this model py file?
    """Load target and optimizer models with Transformers, vLLM, or both."""

    def __init__(
        self,
        *,
        target_model_id: str,
        optimizer_model_id: str | None,
        target_device: str | None,
        hf_target_device: str | None = None,
        optimizer_device: str | None,
        keep_models_loaded: bool,
        seed: int,
        backend: str = "transformers",
        gpu_memory_utilization: float = 0.90,
        vllm_max_model_len: int | None = None,
        vllm_disable_images: bool = False,
        vllm_conservative_settings: bool = False,
        dual_vllm_max_num_batched_tokens: int = 4096,
        dual_vllm_max_num_seqs: int = 128,
    ):
        """Store role settings without loading a model yet."""
        if backend not in {"transformers", "vllm", "dual"}:
            raise ValueError(f"Unsupported model backend: {backend!r}")
        if not 0.0 < gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be greater than 0 and at most 1.")
        if vllm_max_model_len is not None and vllm_max_model_len <= 0:
            raise ValueError("vllm_max_model_len must be positive when provided.")
        if dual_vllm_max_num_batched_tokens <= 0:
            raise ValueError("dual_vllm_max_num_batched_tokens must be positive.")
        if dual_vllm_max_num_seqs <= 0:
            raise ValueError("dual_vllm_max_num_seqs must be positive.")
        self.model_ids = {
            TARGET_ROLE: target_model_id,
            OPTIMIZER_ROLE: optimizer_model_id or target_model_id,
        }
        self.vllm_device_maps = {
            TARGET_ROLE: target_device,
            OPTIMIZER_ROLE: optimizer_device or target_device,
        }
        self.hf_device_maps = {
            TARGET_ROLE: hf_target_device or target_device,
            OPTIMIZER_ROLE: optimizer_device or target_device,
        }
        self.keep_models_loaded = keep_models_loaded
        self.seed = seed
        self.backend = backend
        self.gpu_memory_utilization = gpu_memory_utilization
        self.vllm_max_model_len = vllm_max_model_len
        self.vllm_disable_images = vllm_disable_images
        self.vllm_conservative_settings = vllm_conservative_settings
        self.dual_vllm_max_num_batched_tokens = dual_vllm_max_num_batched_tokens
        self.dual_vllm_max_num_seqs = dual_vllm_max_num_seqs
        self.generation_call_index = 0
        self.loaded: dict[str, tuple[Any, Any]] = {}
        self.vllm_loaded: dict[str, tuple[Any, Any]] = {}

    def ensure(self, role: str) -> tuple[Any, Any]: #aunabil3rd: this call is meaningless for the dual mode, since we will be always have both the models loaded
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

        if not self.keep_models_loaded and self.backend != "dual":
            self.unload_all()
        seed_everything(self.seed)
        if self.backend == "vllm":
            from agents.agent_vllm_models import load_vllm_model_and_tokenizer

            pair = load_vllm_model_and_tokenizer(
                requested_id,
                device=self.vllm_device_maps[role],
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.vllm_max_model_len,
                disable_image_inputs=self.vllm_disable_images,
                enforce_eager=self.vllm_conservative_settings,
                max_num_batched_tokens=(
                    self.dual_vllm_max_num_batched_tokens
                    if self.vllm_conservative_settings
                    else None
                ),
                max_num_seqs=(
                    self.dual_vllm_max_num_seqs
                    if self.vllm_conservative_settings
                    else None
                ),
                enable_prefix_caching=(
                    False if self.vllm_conservative_settings else None
                ),
            )
        else:
            from agents.agent_models import load_model_and_tokenizer

            pair = load_model_and_tokenizer(
                requested_id,
                device_map=self.hf_device_maps[role],
            )
        self.loaded[role] = pair
        return pair

    def ensure_vllm(self, role: str) -> tuple[Any, Any]:
        """Return the requested role's vLLM copy, loading it when necessary."""
        if self.backend not in {"vllm", "dual"}:
            raise RuntimeError("A vLLM model is available only in vLLM or dual mode.")
        if self.backend == "vllm":
            return self.ensure(role)
        if role not in self.model_ids:
            raise ValueError(f"Unknown model role: {role!r}")
        if role in self.vllm_loaded:
            return self.vllm_loaded[role]

        requested_id = self.model_ids[role]
        for loaded_role, pair in list(self.vllm_loaded.items()):
            if self.model_ids[loaded_role] == requested_id:
                self.vllm_loaded[role] = pair
                return pair

        seed_everything(self.seed)
        from agents.agent_vllm_models import load_vllm_model_and_tokenizer

        pair = load_vllm_model_and_tokenizer(
            requested_id,
            device=self.vllm_device_maps[role],
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.vllm_max_model_len,
            disable_image_inputs=self.vllm_disable_images,
            enforce_eager=self.vllm_conservative_settings,
            max_num_batched_tokens=(
                self.dual_vllm_max_num_batched_tokens
                if self.vllm_conservative_settings
                else None
            ),
            max_num_seqs=(
                self.dual_vllm_max_num_seqs
                if self.vllm_conservative_settings
                else None
            ),
            enable_prefix_caching=(
                False if self.vllm_conservative_settings else None
            ),
        )
        self.vllm_loaded[role] = pair
        return pair

    @property
    def uses_vllm_generation(self) -> bool:
        """Report whether ordinary generation is routed through vLLM."""
        return self.backend in {"vllm", "dual"}

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
        seeds: Sequence[int] | None = None,
    ):
        """Generate text with the paper's fixed Qwen3 or Gemma3 decoding settings."""
        if seeds is not None and len(seeds) != len(prompts):
            raise ValueError("One generation seed is required for every prompt.")
        if self.uses_vllm_generation:
            model, tokenizer = self.ensure_vllm(role)
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
                seeds=seeds,
            )

        if seeds is not None:
            raise ValueError("Per-prompt seeds are currently supported only by vLLM.")
        model, tokenizer = self.ensure(role)
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

    def record_cached_generation(self) -> None:
        """Preserve later vLLM seeds when one generation call comes from cache."""
        if self.uses_vllm_generation:
            self.generation_call_index += 1

    def unload_all(self) -> None: #aunabil3rd: is hf model unloaded here too? i don't see the codes
        """Release every loaded model and clear CUDA allocator caches."""
        from agents.agent_memory import clear_model_memory

        unique_pairs = {id(pair): pair for pair in self.loaded.values()}
        if self.backend == "vllm":
            from agents.agent_vllm_models import shutdown_vllm_model

            for model, _ in unique_pairs.values():
                shutdown_vllm_model(model)
        unique_vllm_pairs = {
            id(pair): pair for pair in self.vllm_loaded.values()
        }
        if unique_vllm_pairs:
            from agents.agent_vllm_models import shutdown_vllm_model

            for model, _ in unique_vllm_pairs.values():
                shutdown_vllm_model(model)
        self.loaded.clear()
        self.vllm_loaded.clear()
        unique_pairs.clear()
        unique_vllm_pairs.clear()
        gc.collect()
        clear_model_memory()

    def activate_vllm_only(self, gpu_memory_utilization: float) -> None:
        """Release HF models and switch the pool to one full-size vLLM engine."""
        if not 0.0 < gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be greater than 0 and at most 1.")
        self.unload_all()
        self.backend = "vllm"
        self.gpu_memory_utilization = gpu_memory_utilization

    def close(self) -> None:
        """Release all model resources at the end of a run."""
        self.unload_all()
