"""Regression tests for ANLI test inference and RPO task support."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "codes"))

from anli_task_common import ANLI_ANSWER_INSTRUCTION, ANLI_INITIAL_PROMPT
from prompt_optimization.meta_prompts import (
    qa_task_description,
    rpo_feedback_prompt,
)
from prompt_optimization.qa_task import (
    render_qa_prompt,
    resolve_mode,
    rpo_feedback_example,
    score_qa_response,
)


class AnliPromptOptimizationTests(unittest.TestCase):
    """Check ANLI rendering, exact-label scoring, and RPO feedback wording."""

    def setUp(self) -> None:
        """Create one three-way inference record."""
        self.mode = resolve_mode("non_reasoning", "anli")
        self.record = {
            "id": "anli-test-1",
            "dataset": "anli",
            "task_type": "multiple_choice_qa",
            "premise": "A dog is sleeping on the rug.",
            "hypothesis": "An animal is asleep.",
            "question": (
                "Premise:\nA dog is sleeping on the rug.\n\n"
                "Hypothesis:\nAn animal is asleep."
            ),
            "choices": [
                {"label": "A", "text": "Entailment"},
                {"label": "B", "text": "Neutral"},
                {"label": "C", "text": "Contradiction"},
            ],
            "answer": "A",
        }

    def test_mode_reuses_test_inference_prompts(self) -> None:
        """Keep the ANLI editable and fixed instructions in one shared source."""
        self.assertEqual(self.mode.initial_prompt, ANLI_INITIAL_PROMPT)
        self.assertEqual(self.mode.answer_instruction, ANLI_ANSWER_INSTRUCTION)
        self.assertFalse(self.mode.enable_thinking)
        self.assertEqual(self.mode.default_max_new_tokens, 16)

    def test_render_and_score_use_exact_abc_labels(self) -> None:
        """Render premise and hypothesis while accepting only a valid option label."""
        rendered = render_qa_prompt(ANLI_INITIAL_PROMPT, self.record, self.mode)
        self.assertIn(ANLI_ANSWER_INSTRUCTION, rendered)
        self.assertIn("Premise:\nA dog is sleeping", rendered)
        self.assertIn("Hypothesis:\nAn animal is asleep", rendered)
        self.assertTrue(
            score_qa_response(self.record, "<answer>A</answer>", self.mode)["correct"]
        )
        self.assertTrue(
            score_qa_response(
                self.record,
                "<answer>entailment</answer>",
                self.mode,
            )["invalid_choice_label"]
        )

    def test_rpo_feedback_is_anli_specific(self) -> None:
        """Show premise and hypothesis without exposing the fixed answer format."""
        example = rpo_feedback_example(
            self.record,
            {
                "predicted_answer": "B",
                "correct": False,
                "raw_response": "<answer>B</answer>",
            },
            1,
            self.mode,
        )
        meta_prompt = rpo_feedback_prompt(self.mode, example)
        self.assertIn("Premise: A dog is sleeping", example)
        self.assertIn("Hypothesis: An animal is asleep", example)
        self.assertNotIn("<answer>", example)
        self.assertIn("three-way natural language inference", meta_prompt)
        self.assertIn("entailment, neutral, or contradiction", qa_task_description(self.mode))


if __name__ == "__main__":
    unittest.main()
