"""Regression tests for batched validation and dual-backend helpers."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "codes"))

from prompt_optimization.evaluation import QAEvaluator
from prompt_optimization.models import ModelPool, TARGET_ROLE
from prompt_optimization.qa_task import resolve_mode
from prompt_optimization.sequence_gradients import score_instruction_task_losses
from agents.agent_vllm_models import (
    _temporary_visible_gpu,
    load_vllm_model_and_tokenizer,
)


class _FakeVLLMPool:
    """Capture one flattened vLLM request without loading a real model."""

    uses_vllm_generation = True

    def __init__(self) -> None:
        """Prepare storage for the most recent generation request."""
        self.prompts = []
        self.seeds = []

    def generate(self, _role, prompts, **kwargs):
        """Return the correct label for every captured request."""
        self.prompts = list(prompts)
        self.seeds = list(kwargs["seeds"])
        outputs = ["<answer>A</answer>"] * len(prompts)
        usages = [
            {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}
            for _prompt in prompts
        ]
        return outputs, usages


class _CharacterTokenizer:
    """Provide deterministic character tokens and offsets for an HF scoring test."""

    padding_side = "left"

    def apply_chat_template(self, messages, **_kwargs):
        """Render the user and assistant messages as one character sequence."""
        return f"{messages[0]['content']}\n{messages[1]['content']}"

    def __call__(self, texts, **_kwargs):
        """Left-pad character IDs and retain one offset for each character."""
        text_list = [texts] if isinstance(texts, str) else list(texts)
        maximum = max(len(text) for text in text_list)
        input_rows = []
        attention_rows = []
        offset_rows = []
        for text in text_list:
            padding = maximum - len(text)
            input_rows.append([0] * padding + [ord(character) for character in text])
            attention_rows.append([0] * padding + [1] * len(text))
            offset_rows.append(
                [(0, 0)] * padding
                + [(index, index + 1) for index in range(len(text))]
            )
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "offset_mapping": torch.tensor(offset_rows, dtype=torch.long),
        }


class _UniformHFModel:
    """Return uniform logits while recording the number of HF forward passes."""

    def __init__(self) -> None:
        """Create a CPU embedding table used only to expose the model device."""
        self.embedding = torch.nn.Embedding(256, 2)
        self.forward_calls = 0

    def get_input_embeddings(self):
        """Return the small CPU embedding table."""
        return self.embedding

    def __call__(self, *, input_ids, attention_mask, use_cache):
        """Return uniform next-token logits for every sequence position."""
        del attention_mask, use_cache
        self.forward_calls += 1
        logits = torch.zeros((*input_ids.shape, 256), dtype=torch.float32)
        return SimpleNamespace(logits=logits)


class BatchedGradientRuntimeTests(unittest.TestCase):
    """Check bulk request layout, shared seeds, and optional vLLM safeguards."""

    def test_candidate_evaluation_flattens_every_prompt_record_pair(self) -> None:
        """Submit all candidate and validation-example combinations together."""
        records = [
            {
                "id": f"item-{fold}",
                "task_type": "multiple_choice_qa",
                "question": "Choose A.",
                "choices": [
                    {"label": "A", "text": "Correct"},
                    {"label": "B", "text": "Wrong"},
                ],
                "answer": "A",
                "validation_fold": fold,
            }
            for fold in (1, 2, 3)
        ]
        pool = _FakeVLLMPool()
        evaluator = QAEvaluator(
            model_pool=pool,
            mode=resolve_mode("non_reasoning", "openbookqa"),
            batch_size=4,
            max_new_tokens=10,
            seed=42,
        )

        evaluations = evaluator.evaluate_many(
            ["Instruction one.", "Instruction two."],
            records,
            split_name="validation",
            log_label="batched_test",
        )

        self.assertEqual(len(pool.prompts), 6)
        self.assertEqual(pool.seeds[:3], pool.seeds[3:])
        self.assertEqual(
            [item["metrics"]["accuracy"] for item in evaluations],
            [1.0, 1.0],
        )

    def test_conservative_vllm_settings_are_disabled_by_default(self) -> None:
        """Leave all four optional engine settings at their original defaults."""
        pool = ModelPool(
            target_model_id="target",
            optimizer_model_id=None,
            target_device="cuda:0",
            hf_target_device=None,
            optimizer_device=None,
            keep_models_loaded=False,
            seed=42,
            backend="dual",
        )
        with patch(
            "agents.agent_vllm_models.load_vllm_model_and_tokenizer",
            return_value=(object(), object()),
        ) as loader:
            pool.ensure_vllm(TARGET_ROLE)

        self.assertFalse(loader.call_args.kwargs["enforce_eager"])
        self.assertIsNone(loader.call_args.kwargs["max_num_batched_tokens"])
        self.assertIsNone(loader.call_args.kwargs["max_num_seqs"])
        self.assertIsNone(loader.call_args.kwargs["enable_prefix_caching"])

    @unittest.skipIf(torch is None, "PyTorch is required to import the HF loader.")
    def test_dual_backend_routes_hf_and_vllm_to_separate_devices(self) -> None:
        """Send vLLM to the generation GPU and HF to its optional scoring GPU."""
        from agents import agent_models

        pool = ModelPool(
            target_model_id="target",
            optimizer_model_id=None,
            target_device="cuda:0",
            hf_target_device="cuda:1",
            optimizer_device=None,
            keep_models_loaded=False,
            seed=42,
            backend="dual",
        )
        with patch(
            "agents.agent_vllm_models.load_vllm_model_and_tokenizer",
            return_value=(object(), object()),
        ) as vllm_loader:
            pool.ensure_vllm(TARGET_ROLE)
        with patch.object(
            agent_models,
            "load_model_and_tokenizer",
            return_value=(object(), object()),
        ) as hf_loader:
            pool.ensure(TARGET_ROLE)

        self.assertEqual(vllm_loader.call_args.kwargs["device"], "cuda:0")
        self.assertEqual(hf_loader.call_args.kwargs["device_map"], "cuda:1")

    def test_vllm_gpu_selection_restores_parent_visible_devices(self) -> None:
        """Restore both parent GPUs after temporarily isolating the vLLM GPU."""
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear=False):
            with _temporary_visible_gpu("cuda:0") as selected_device:
                self.assertEqual(selected_device, "2")
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,3")

    def test_vllm_loader_restores_gpus_after_worker_startup(self) -> None:
        """Keep vLLM isolated during construction and restore GPUs afterward."""
        startup_state = {}
        fake_vllm = ModuleType("vllm")

        class _FakeLLM:
            """Capture the environment seen when the fake vLLM engine starts."""

            def __init__(self, **_kwargs) -> None:
                """Record startup visibility without creating a real engine."""
                startup_state["visible_devices"] = os.getenv("CUDA_VISIBLE_DEVICES")
                startup_state["worker_method"] = os.getenv(
                    "VLLM_WORKER_MULTIPROC_METHOD"
                )

            def get_tokenizer(self):
                """Return a placeholder tokenizer for the loader contract."""
                return object()

        fake_vllm.LLM = _FakeLLM
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear=False):
            os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
            with patch.dict(sys.modules, {"vllm": fake_vllm}):
                with patch(
                    "agents.agent_vllm_models._validate_vllm_version",
                    return_value="test",
                ):
                    load_vllm_model_and_tokenizer("fake", device="cuda:0")

            self.assertEqual(startup_state["visible_devices"], "2")
            self.assertEqual(startup_state["worker_method"], "spawn")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2,3")

    def test_conservative_vllm_settings_are_opt_in(self) -> None:
        """Apply all four engine safeguards only when explicitly requested."""
        pool = ModelPool(
            target_model_id="target",
            optimizer_model_id=None,
            target_device="cuda:0",
            hf_target_device=None,
            optimizer_device=None,
            keep_models_loaded=False,
            seed=42,
            backend="dual",
            vllm_conservative_settings=True,
        )
        with patch(
            "agents.agent_vllm_models.load_vllm_model_and_tokenizer",
            return_value=(object(), object()),
        ) as loader:
            pool.ensure_vllm(TARGET_ROLE)

        self.assertTrue(loader.call_args.kwargs["enforce_eager"])
        self.assertEqual(loader.call_args.kwargs["max_num_batched_tokens"], 4096)
        self.assertEqual(loader.call_args.kwargs["max_num_seqs"], 128)
        self.assertFalse(loader.call_args.kwargs["enable_prefix_caching"])

    @unittest.skipIf(torch is None, "PyTorch is unavailable in the lightweight test env.")
    def test_hf_task_loss_batches_across_candidates(self) -> None:
        """Process mixed candidate/example sequences in shared HF forward batches."""
        records = [
            {
                "id": f"item-{index}",
                "task_type": "multiple_choice_qa",
                "question": "Choose A.",
                "choices": [
                    {"label": "A", "text": "Correct"},
                    {"label": "B", "text": "Wrong"},
                ],
                "answer": "A",
            }
            for index in range(3)
        ]
        model = _UniformHFModel()
        losses = score_instruction_task_losses(
            ["Instruction one.", "Instruction two."],
            records,
            mode=resolve_mode("non_reasoning", "openbookqa"),
            model=model,
            tokenizer=_CharacterTokenizer(),
            batch_size=2,
        )

        self.assertEqual(model.forward_calls, 3)
        self.assertEqual(len(losses), 2)
        self.assertAlmostEqual(losses[0], losses[1])


if __name__ == "__main__":
    unittest.main()
