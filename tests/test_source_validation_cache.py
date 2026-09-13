"""Regression tests for shared source-prompt validation caching."""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "codes"))

from prompt_optimization.qa_task import resolve_mode
from prompt_optimization.second_stage import _evaluate_with_cached_source
from prompt_optimization.source_validation_cache import (
    build_source_validation_identity,
    source_validation_cache_path,
)


def _evaluation_for_prompt(prompt: str, accuracy: float) -> dict:
    """Build one small serializable validation result for a prompt."""
    return {
        "split": "validation",
        "instruction_prompt": prompt,
        "metrics": {
            "accuracy": accuracy,
            "stable_accuracy": accuracy,
        },
        "predictions": [],
    }


class SourceValidationCacheTests(unittest.TestCase):
    """Check exact cache matching and repeated-source reuse."""

    def test_validation_record_changes_produce_different_cache_keys(self) -> None:
        """Separate cache entries for different validation subsets or sizes."""
        common = {
            "model_id": "Qwen/Qwen3-4B",
            "generation_backend": "vllm",
            "task_name": "openbookqa",
            "mode_name": "non_reasoning",
            "source_prompt": "Answer carefully.",
            "seed": 42,
            "max_new_tokens": 10,
            "enable_thinking": False,
            "validation_std_penalty": 1.0,
        }
        first = build_source_validation_identity(
            **common,
            validation_records=[{"id": "one", "validation_fold": 1}],
            rendered_prompts=["rendered one"],
        )
        second = build_source_validation_identity(
            **common,
            validation_records=[
                {"id": "one", "validation_fold": 1},
                {"id": "two", "validation_fold": 2},
            ],
            rendered_prompts=["rendered one", "rendered two"],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            first_path, first_key = source_validation_cache_path(
                temporary_directory,
                first,
            )
            second_path, second_key = source_validation_cache_path(
                temporary_directory,
                second,
            )

        self.assertNotEqual(first_key, second_key)
        self.assertNotEqual(first_path, second_path)

    def test_second_method_skips_only_the_cached_source_prompt(self) -> None:
        """Reuse the source result while still evaluating each new candidate."""
        mode = resolve_mode("non_reasoning", "openbookqa")
        records = [
            {
                "id": "question-1",
                "task_type": "multiple_choice",
                "question": "Which option is correct?",
                "choices": [
                    {"label": "A", "text": "First"},
                    {"label": "B", "text": "Second"},
                ],
                "answer": "A",
                "validation_fold": 1,
            }
        ]
        evaluated_batches: list[list[str]] = []

        def fake_evaluate_candidates(
            _context,
            candidates,
            _records,
            **_kwargs,
        ):
            """Record each submitted batch and return deterministic scores."""
            evaluated_batches.append(list(candidates))
            output = []
            for candidate_index, prompt in enumerate(candidates):
                accuracy = 0.5 + 0.01 * candidate_index
                evaluation = _evaluation_for_prompt(prompt, accuracy)
                output.append(
                    {
                        "candidate_index": candidate_index,
                        "prompt": prompt,
                        "accuracy": accuracy,
                        "selection_score": accuracy,
                        "metrics": evaluation["metrics"],
                        "evaluation": evaluation,
                    }
                )
            return output

        with tempfile.TemporaryDirectory() as temporary_directory:
            args = SimpleNamespace(
                model="Qwen/Qwen3-4B",
                source_validation_cache_root=temporary_directory,
                refresh_source_validation_cache=False,
                disable_source_validation_cache=False,
            )
            context = SimpleNamespace(
                args=args,
                initial_prompt="Answer carefully.",
                mode=mode,
                validation_records=records,
                model_pool=SimpleNamespace(uses_vllm_generation=True),
                evaluator=SimpleNamespace(
                    seed=42,
                    max_new_tokens=10,
                    validation_std_penalty=1.0,
                ),
                optimizer_name="greater",
                started_at=time.monotonic(),
                logger=SimpleNamespace(event=lambda *_args, **_kwargs: None),
            )
            with patch(
                "prompt_optimization.second_stage.evaluate_candidates",
                side_effect=fake_evaluate_candidates,
            ):
                first_initial, _first_source, _first_candidates = (
                    _evaluate_with_cached_source(
                        context,
                        ["Candidate A."],
                        phase="first_validation",
                    )
                )
                second_initial, _second_source, _second_candidates = (
                    _evaluate_with_cached_source(
                        context,
                        ["Candidate B."],
                        phase="second_validation",
                    )
                )

            cache_files = list(Path(temporary_directory).rglob("*.json"))

        self.assertEqual(
            evaluated_batches,
            [
                ["Answer carefully.", "Candidate A."],
                ["Candidate B."],
            ],
        )
        self.assertEqual(first_initial, second_initial)
        self.assertEqual(len(cache_files), 1)


if __name__ == "__main__":
    unittest.main()
