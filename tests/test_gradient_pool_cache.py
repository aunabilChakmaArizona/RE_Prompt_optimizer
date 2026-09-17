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
from prompt_optimization.second_stage import (
    _build_balanced_gradient_subset,
    _get_shared_training_pool,
)


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

    def test_lpo_uses_requested_prefix_of_larger_pool_without_truncating_cache(self) -> None:
        """Reuse a larger gradient cache for LPO while retaining all saved rows."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = _cache_test_context(root, _FakeGradientPoolModel())
            _build_balanced_gradient_subset(first, "Choose.", 4, 4, log_label="seed")
            path = next(root.rglob("*.json"))
            before = json.loads(path.read_text())
            second = _cache_test_context(root, _FakeGradientPoolModel(fail_if_called=True))
            second.optimizer_name = "lpo"
            records, predictions, cache, _ = _get_shared_training_pool(
                second, "Choose.", 2, log_label="lpo",
            )
            self.assertEqual([r["id"] for r in records], before["metadata"]["record_ids"][:2])
            self.assertEqual(len(predictions), 2)
            self.assertEqual(cache["cache_pool_size"], 4)
            self.assertEqual(cache["used_pool_size"], 2)
            self.assertTrue(cache["pool_cache_hit"])
            after = json.loads(path.read_text())
            self.assertEqual(after, before)
            self.assertEqual(len(list(root.rglob("*.json"))), 1)

    def test_lpo_created_pool_is_reused_by_gradient_method(self) -> None:
        """Allow either LPO or GradPO/GreaTer to populate the shared cache first."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = _cache_test_context(root, _FakeGradientPoolModel())
            first.optimizer_name = "lpo"
            _get_shared_training_pool(first, "Choose.", 4, log_label="lpo")
            second = _cache_test_context(root, _FakeGradientPoolModel(fail_if_called=True))
            records, _, metadata = _build_balanced_gradient_subset(
                second, "Choose.", 4, 4, log_label="greater",
            )
            self.assertEqual(len(records), 4)
            self.assertTrue(metadata["cache"]["pool_cache_hit"])
            self.assertEqual(first.model_pool.generation_calls, 1)

    def test_smaller_gradient_pool_gets_its_own_balanced_subset_in_large_cache(self) -> None:
        """Keep subset identities distinct for different requested prefix sizes."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = _cache_test_context(root, _FakeGradientPoolModel())
            _build_balanced_gradient_subset(first, "Choose.", 4, 4, log_label="large")
            second = _cache_test_context(root, _FakeGradientPoolModel(fail_if_called=True))
            _, _, small = _build_balanced_gradient_subset(
                second, "Choose.", 2, 2, log_label="small",
            )
            third = _cache_test_context(root, _FakeGradientPoolModel(fail_if_called=True))
            _, _, repeat = _build_balanced_gradient_subset(
                third, "Choose.", 2, 2, log_label="repeat",
            )
            self.assertFalse(small["cache"]["balanced_subset_cache_hit"])
            self.assertTrue(repeat["cache"]["balanced_subset_cache_hit"])
            self.assertEqual(small["initial_pool_record_ids"], repeat["initial_pool_record_ids"])
            payload = json.loads(next(root.rglob("*.json")).read_text())
            self.assertEqual(len(payload["pool_results"]), 4)
            self.assertEqual(len(payload["balanced_subsets"]), 2)

    def test_changed_record_content_does_not_reuse_larger_cache(self) -> None:
        """Reject cached examples when their questions or answers have changed."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = _cache_test_context(root, _FakeGradientPoolModel())
            _get_shared_training_pool(first, "Choose.", 4, log_label="first")
            second = _cache_test_context(root, _FakeGradientPoolModel())
            second.train_records[-1]["question"] = "Updated question"
            _, _, cache, _ = _get_shared_training_pool(second, "Choose.", 2, log_label="changed")
            self.assertFalse(cache["pool_cache_hit"])
            self.assertEqual(second.model_pool.generation_calls, 1)

    def test_changed_generation_identity_does_not_reuse_larger_cache(self) -> None:
        """Never mix source prompts, models, seeds, backends, or token limits."""
        for field in ("source", "model", "seed", "backend", "limit"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                first = _cache_test_context(root, _FakeGradientPoolModel())
                _get_shared_training_pool(first, "Choose.", 4, log_label="first")
                second = _cache_test_context(root, _FakeGradientPoolModel())
                prompt = "Choose."
                if field == "source":
                    prompt = "A different instruction."
                elif field == "model":
                    second.args.model = "other/model"
                elif field == "seed":
                    second.args.seed = 43
                elif field == "backend":
                    second.model_pool.uses_vllm_generation = False
                else:
                    second.evaluator.max_new_tokens = 20
                _, _, cache, _ = _get_shared_training_pool(second, prompt, 2, log_label="changed")
                self.assertFalse(cache["pool_cache_hit"])
                self.assertEqual(second.model_pool.generation_calls, 1)

    def test_refresh_and_disable_bypass_existing_larger_cache(self) -> None:
        """Honor explicit regeneration controls without destroying a larger cache."""
        for option in ("refresh_gradient_cache", "disable_gradient_cache"):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                first = _cache_test_context(root, _FakeGradientPoolModel())
                _get_shared_training_pool(first, "Choose.", 4, log_label="first")
                large_path = next(root.rglob("*.json"))
                before = large_path.read_text()
                second = _cache_test_context(root, _FakeGradientPoolModel())
                setattr(second.args, option, True)
                _, _, cache, _ = _get_shared_training_pool(second, "Choose.", 2, log_label="bypass")
                self.assertFalse(cache["pool_cache_hit"])
                self.assertEqual(second.model_pool.generation_calls, 1)
                self.assertEqual(large_path.read_text(), before)


if __name__ == "__main__":
    unittest.main()
