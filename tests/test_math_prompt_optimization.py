"""Regression tests for the MATH first-stage prompt-optimization task layer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "codes"))

from math_inference_common import ANSWER_INSTRUCTION_PROMPT
from prompt_optimization.meta_prompts import (
    etgpo_first_taxonomy_prompt,
    evoprompt_de_prompt,
    qa_task_description,
    rpo_feedback_prompt,
)
from prompt_optimization.qa_evoprompt_seeds import QA_EVOPROMPT_SEEDS
from prompt_optimization.qa_task import (
    load_qa_records,
    render_qa_prompt,
    resolve_mode,
    rpo_feedback_example,
    score_qa_response,
    summarize_qa_predictions,
)


class MathPromptOptimizationTests(unittest.TestCase):
    """Check math-specific rendering, grading, feedback, and meta-prompts."""

    def setUp(self) -> None:
        """Create one small symbolic-answer record used by the tests."""
        self.mode = resolve_mode("reasoning", "math500")
        self.record = {
            "id": "math-test-1",
            "task_type": "math_symbolic_answer",
            "question": "Compute $1/2+1/2$.",
            "answer": "1",
            "subject": "Algebra",
            "level": 1,
        }

    def test_mode_and_rendering_match_test_inference(self) -> None:
        """Keep the editable instruction separate from the fixed answer format."""
        rendered = render_qa_prompt("Solve this exactly.", self.record, self.mode)

        self.assertEqual(self.mode.default_max_new_tokens, 8192)
        self.assertTrue(rendered.startswith("Solve this exactly."))
        self.assertIn(ANSWER_INSTRUCTION_PROMPT, rendered)
        self.assertTrue(rendered.endswith("Question:\nCompute $1/2+1/2$."))

    @patch("prompt_optimization.qa_task.grade_math_answer")
    def test_openai_grade_is_primary_with_boxed_fallback(self, mock_grade) -> None:
        """Select prompts by OpenAI grading while retaining both diagnostics."""
        mock_grade.return_value = {
            "normalized_prediction": "1",
            "normalized_gold_answer": "1",
            "simple_correct": False,
            "openai_correct": True,
            "math_verify_correct": True,
            "openai_error": None,
            "math_verify_error": None,
        }

        prediction = score_qa_response(
            self.record,
            "Reasoning here. Therefore, $\\boxed{1}$.",
            self.mode,
        )
        metrics = summarize_qa_predictions([prediction])

        mock_grade.assert_called_once_with("1", "1", "math_symbolic_answer")
        self.assertTrue(prediction["correct"])
        self.assertEqual(prediction["answer_extraction_source"], "boxed_fallback")
        self.assertTrue(prediction["missing_answer_tag"])
        self.assertEqual(metrics["primary_grader"], "openai_prm800k")
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["simple_accuracy"], 0.0)

    def test_feedback_exposes_reasoning_but_not_fixed_answer_instruction(self) -> None:
        """Give RPO the solution trace and selected answer without answer instructions."""
        prediction = {
            "predicted_answer": "2",
            "correct": False,
            "raw_response": "Added incorrectly. <answer>2</answer>",
        }

        example = rpo_feedback_example(self.record, prediction, 1, self.mode)

        self.assertIn("LLM Reasoning:\nAdded incorrectly.", example)
        self.assertIn("LLM Selected Answer: 2", example)
        self.assertNotIn("<answer>", example)
        self.assertNotIn("After you finish reasoning", example)

    def test_math_meta_prompts_preserve_task_generic_wording(self) -> None:
        """Use math-specific instructions without naming the benchmark."""
        feedback_prompt = rpo_feedback_prompt(self.mode, "Task 1")
        taxonomy_prompt = etgpo_first_taxonomy_prompt(["Failure 1"], self.mode)
        de_prompt = evoprompt_de_prompt(
            "Base instruction",
            "Donor one",
            "Donor two",
            "Donor three",
            self.mode,
        )

        self.assertIn("mathematical problem-solving", feedback_prompt)
        self.assertIn("mathematically equivalent", taxonomy_prompt)
        self.assertIn("Basic Prompt: Base instruction", de_prompt)
        self.assertNotIn("MATH-500", feedback_prompt + taxonomy_prompt + de_prompt)
        self.assertNotIn("math500", feedback_prompt + taxonomy_prompt + de_prompt)
        self.assertIn("exact final answer", qa_task_description(self.mode))

    def test_math_evoprompt_has_four_fixed_seeds(self) -> None:
        """Provide four distinct math seeds beside the source prompt."""
        seeds = QA_EVOPROMPT_SEEDS["math500_reasoning"]

        self.assertEqual(len(seeds), 4)
        self.assertEqual(len({item["prompt"] for item in seeds}), 4)

    def test_prepared_validation_split_loads(self) -> None:
        """Accept the prepared 1,500-example symbolic math validation set."""
        records = load_qa_records(
            REPO_ROOT / "data" / "processed" / "math500" / "validation.jsonl",
            "math500",
        )

        self.assertEqual(len(records), 1500)
        self.assertEqual(
            {int(record["validation_fold"]) for record in records},
            {1, 2, 3},
        )
        self.assertEqual(
            {
                sum(
                    int(record["validation_fold"]) == fold for record in records
                )
                for fold in range(1, 4)
            },
            {500},
        )


if __name__ == "__main__":
    unittest.main()
