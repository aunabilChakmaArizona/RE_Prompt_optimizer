"""Tests for shared GreaTer and GradPO gradient-pool caching."""

from __future__ import annotations

import json
import random
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "codes"))

from prompt_optimization.qa_task import resolve_mode
from prompt_optimization.second_stage import _build_balanced_gradient_subset


class _FakeGradientPoolModel:
    """Return fixed initial answers while counting expensive generation calls."""

    uses_vllm_generation = True

    def __init__(self, fail_if_called: bool = False) -> None:
        """Configure whether a cache hit must prevent generation entirely."""
        self.fail_if_called = fail_if_called
        self.generation_calls = 0
        self.cached_generation_calls = 0

    def generate(self, _role, prompts, **_kwargs):
        """Return answer A for every sampled multiple-choice record."""
        if self.fail_if_called:
            raise AssertionError("Gradient-pool generation should have used the cache.")
        self.generation_calls += 1
        return ["<answer>A</answer>" for _prompt in prompts]

    def record_cached_generation(self) -> None:
        """Record the virtual generation call consumed by a cache hit."""
        self.cached_generation_calls += 1


def _cache_test_records() -> list[dict]:
    """Create two correct and two incorrect outcomes for the fixed response."""
    return [
        {
            "id": f"record-{index}",
            "task_type": "multiple_choice_qa",
            "question": f"Question {index}",
            "choices": [
                {"label": "A", "text": "First"},
                {"label": "B", "text": "Second"},
            ],
            "answer": answer,
        }
        for index, answer in enumerate(("A", "B", "A", "B"), start=1)
    ]


def _cache_test_context(cache_root: Path, model_pool) -> SimpleNamespace:
    """Build the minimal optimizer context used by gradient-pool preparation."""
    return SimpleNamespace(
        args=SimpleNamespace(
            seed=42,
            model="fake/model",
            gradient_cache_root=str(cache_root),
            disable_gradient_cache=False,
            refresh_gradient_cache=False,
        ),
        model_pool=model_pool,
        evaluator=SimpleNamespace(max_new_tokens=10, batch_size=4),
        mode=resolve_mode("non_reasoning", "openbookqa"),
        train_records=_cache_test_records(),
        rng=random.Random(42),
        optimizer_name="gradpo_gen",
        started_at=time.monotonic(),
    )


class GradientPoolCacheTests(unittest.TestCase):
    """Verify raw-response and exact balanced-subset reuse across refiners."""

    def test_second_run_reuses_raw_responses_and_balanced_subset(self) -> None:
        """Skip generation and restore identical selected IDs on a cache hit."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_root = Path(temporary_directory)
            first_model = _FakeGradientPoolModel()
            first_context = _cache_test_context(cache_root, first_model)
            first_records, first_traces, first_metadata = (
                _build_balanced_gradient_subset(
                    first_context,
                    "Choose the best answer.",
                    4,
                    4,
                    log_label="cache_test_first",
                )
            )

            second_model = _FakeGradientPoolModel(fail_if_called=True)
            second_context = _cache_test_context(cache_root, second_model)
            second_context.optimizer_name = "greater"
            second_records, second_traces, second_metadata = (
                _build_balanced_gradient_subset(
                    second_context,
                    "Choose the best answer.",
                    4,
                    4,
                    log_label="cache_test_second",
                )
            )

            first_ids = [record["id"] for record in first_records]
            second_ids = [record["id"] for record in second_records]
            self.assertEqual(first_model.generation_calls, 1)
            self.assertEqual(second_model.generation_calls, 0)
            self.assertEqual(second_model.cached_generation_calls, 1)
            self.assertEqual(first_ids, second_ids)
            self.assertIsNone(first_traces)
            self.assertIsNone(second_traces)
            self.assertFalse(first_metadata["cache"]["pool_cache_hit"])
            self.assertFalse(
                first_metadata["cache"]["balanced_subset_cache_hit"]
            )
            self.assertTrue(second_metadata["cache"]["pool_cache_hit"])
            self.assertTrue(
                second_metadata["cache"]["balanced_subset_cache_hit"]
            )

            cache_files = list(cache_root.rglob("*.json"))
            self.assertEqual(len(cache_files), 1)
            payload = json.loads(cache_files[0].read_text(encoding="utf-8"))
            self.assertEqual(len(payload["pool_results"]), 4)
            self.assertTrue(
                all(
                    row["raw_response"] == "<answer>A</answer>"
                    for row in payload["pool_results"]
                )
            )
            self.assertEqual(len(payload["balanced_subsets"]), 1)
            subset = next(iter(payload["balanced_subsets"].values()))
            self.assertEqual(len(subset["correct_record_ids"]), 2)
            self.assertEqual(len(subset["incorrect_record_ids"]), 2)
            self.assertEqual(subset["selected_record_ids"], first_ids)


if __name__ == "__main__":
    unittest.main()
