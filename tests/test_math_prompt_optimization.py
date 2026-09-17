"""Regression tests for the MATH first-stage prompt-optimization task layer."""

from __future__ import annotations

import random
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import torch
    import torch.nn.functional as functional
except ImportError:
    torch = None
    functional = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "codes"))

from math_inference_common import ANSWER_INSTRUCTION_PROMPT
from prompt_optimization.first_stage import run_rpo
from prompt_optimization.meta_prompts import (
    etgpo_first_taxonomy_prompt,
    evoprompt_de_prompt,
    qa_task_description,
    rpo_feedback_prompt,
    rpo_rewrite_prompt,
)
from prompt_optimization.qa_evoprompt_seeds import QA_EVOPROMPT_SEEDS
from prompt_optimization.qa_task import (
    feedback_example,
    load_qa_records,
    render_qa_prompt,
    resolve_mode,
    rpo_feedback_example,
    score_qa_response,
    select_validation_fold_subset,
    summarize_qa_predictions,
)
from prompt_optimization.second_stage import (
    _balance_gradient_pairs,
    _beam_search_replacements,
    _filter_gradpo_region_candidates,
    _gradpo_synthesis_prompt,
    _normalize_synthesized_prompt,
    run_lpo,
)
from prompt_optimization.sequence_gradients import (
    _label_loss_from_logits,
    qa_proposal_header,
    render_teacher_forced_qa,
    same_length_single_token_replacement_prompt,
)


class _PlainChatTokenizer:
    """Render chat messages without changing their text for alignment tests."""

    def apply_chat_template(self, messages, **_kwargs):
        """Join one user and assistant message into a predictable transcript."""
        return "\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in messages
        )


class _RoundTripTokenizer:
    """Expose equal-length and length-changing retokenization for testing."""

    def decode(self, token_ids, **_kwargs):
        """Decode integer IDs as a whitespace-separated prompt."""
        return " ".join(str(token_id) for token_id in token_ids)

    def encode(self, text, **_kwargs):
        """Change ID 9 but expand ID 8 to distinguish IDs from token count."""
        token_ids = []
        for value in text.split():
            token_id = int(value)
            if token_id == 8:
                token_ids.extend([8, 88])
            else:
                token_ids.append(99 if token_id == 9 else token_id)
        return token_ids


class _NoGenerationPool:
    """Reject unexpected synthesis calls made by a no-op beam branch."""

    def generate(self, *_args, **_kwargs):
        """Fail if exact no-op preservation incorrectly invokes synthesis."""
        raise AssertionError("The exact GradPO no-op branch must not be synthesized.")


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

    def test_rpo_failure_types_use_actual_generated_token_counts(self) -> None:
        """Distinguish exhausted, missing, and wrong answers without blaming tags."""
        for answer, tokens, expected in (
            (None, 8192, "reasoning token limit exceeded"),
            (None, 4000, "missing final answer"),
            (None, None, "missing final answer"),
            ("", 8192, "reasoning token limit exceeded"),
            ("2", 8192, "incorrect final answer"),
        ):
            with self.subTest(answer=answer, tokens=tokens):
                prediction = {
                    "predicted_answer": answer,
                    "correct": False,
                    "raw_response": "Working through the calculation...",
                    "token_usage": {"output_tokens": tokens},
                }
                example = rpo_feedback_example(
                    self.record, prediction, 1, self.mode,
                    output_token_limit=8192,
                )
                self.assertIn(f"Failure type: {expected}", example)
                self.assertIn("Output-token limit: 8192", example)
                self.assertNotIn(ANSWER_INSTRUCTION_PROMPT, example)

    def test_rpo_correct_answer_at_limit_is_not_a_completion_failure(self) -> None:
        """Do not label a correct boxed or tagged answer as token exhaustion."""
        example = rpo_feedback_example(
            self.record,
            {
                "predicted_answer": "1",
                "correct": True,
                "raw_response": "The result is one.",
                "token_usage": {"output_tokens": 8192},
            },
            1, self.mode, output_token_limit=8192,
        )
        self.assertNotIn("Failure type:", example)

    def test_rpo_shorter_reasoning_guidance_is_math_only(self) -> None:
        """Keep original QA wording and the fixed-answer restriction unchanged."""
        math_feedback = rpo_feedback_prompt(self.mode, "Task 1")
        math_rewrite = rpo_rewrite_prompt("Solve carefully.", ["Feedback"], self.mode)
        self.assertIn("reasoning token limit exceeded", math_feedback)
        self.assertIn("concise, focused reasoning", math_rewrite)
        self.assertIn("Do not add answer-format instructions", math_rewrite)
        for task in ("openbookqa", "hotpotqa"):
            mode = resolve_mode("reasoning", task)
            self.assertNotIn("reasoning token limit exceeded", rpo_feedback_prompt(mode, "Task 1"))
            self.assertNotIn("For token-limit failures", rpo_rewrite_prompt("Solve.", ["Feedback"], mode))

    def test_math_rpo_selects_only_errors_and_preserves_shortage_snapshots(self) -> None:
        """Use available errors without correct fillers or empty-feedback rewrites."""
        for failure_count in (0, 2, 3):
            with self.subTest(failure_count=failure_count), tempfile.TemporaryDirectory() as folder:
                records = [{**self.record, "id": f"math-{i}"} for i in range(4)]
                predictions = [
                    {
                        "correct": i >= failure_count,
                        "predicted_answer": None if i < failure_count else "1",
                        "raw_response": "Reasoning still in progress...",
                        "token_usage": {"output_tokens": 8192},
                    }
                    for i in range(4)
                ]
                evaluation = {
                    "metrics": {"accuracy": 0.8, "stable_accuracy": 0.7},
                    "predictions": predictions,
                }
                context = SimpleNamespace(
                    mode=self.mode, initial_prompt="Solve carefully.",
                    train_records=records, validation_records=records,
                    rng=random.Random(42), run_dir=Path(folder),
                    evaluator=SimpleNamespace(
                        max_new_tokens=8192,
                        evaluate=lambda *_args, **_kwargs: evaluation,
                    ),
                    model_pool=SimpleNamespace(ensure=lambda _role: (None, None)),
                    logger=MagicMock(),
                )
                args = SimpleNamespace(
                    iterations=1, population_sampling_temperature=1.0,
                    feedback_sample_size=100, feedback_examples=3,
                    optimizer_feedback_max_tokens=2000,
                    snapshot_iterations=[1], population_size=10,
                )
                with (
                    patch("prompt_optimization.first_stage.log_progress") as log,
                    patch("prompt_optimization.first_stage.generate_optimizer_texts",
                          side_effect=lambda _context, prompts, **_kwargs: ["<feedback>Be concise.</feedback>"] * len(prompts)) as generate,
                    patch("prompt_optimization.first_stage.generate_tagged_candidates", return_value=([], [])) as rewrite,
                    patch("prompt_optimization.first_stage.finalize_run", return_value={}) as finalize,
                ):
                    run_rpo(context, args)
                trace = next(
                    call.kwargs for call in context.logger.event.call_args_list
                    if call.args[0] == "rpo_optimizer_trace"
                )
                self.assertEqual(len(trace["feedback_examples"]), failure_count)
                self.assertTrue(all("Outcome: incorrect" in x for x in trace["feedback_examples"]))
                self.assertTrue(all("Failure type: reasoning token limit exceeded" in x for x in trace["feedback_examples"]))
                self.assertEqual(generate.call_count, int(failure_count > 0))
                self.assertEqual(rewrite.call_count, int(failure_count > 0))
                self.assertIn("1", finalize.call_args.kwargs["extra_summary"]["snapshots"])
                if failure_count < 3:
                    self.assertTrue(any("feedback shortage" in call.args[1] for call in log.call_args_list))

    def test_lpo_feedback_keeps_the_full_reasoning_trace(self) -> None:
        """Show LPO every part of a reasoning trace and the answer separately."""
        prediction = {
            "predicted_answer": "2",
            "correct": False,
            "raw_response": (
                "Beginning of reasoning.\nMiddle of reasoning.\n"
                "End of reasoning. <answer>2</answer>"
            ),
        }

        example = feedback_example(self.record, prediction, 1, self.mode)

        self.assertIn("Beginning of reasoning.", example)
        self.assertIn("Middle of reasoning.", example)
        self.assertIn("End of reasoning.", example)
        self.assertIn("LLM Selected Answer: 2", example)
        self.assertNotIn("<answer>", example)

    def test_lpo_validates_every_distinct_rewrite_without_train_preselection(self) -> None:
        """Send all five LPO rewrites and the source directly to validation."""
        rewrite_outputs = [
            f"<p>Candidate instruction {index}.</p>"
            for index in range(1, 6)
        ]
        captured: dict[str, object] = {}

        def fake_evaluate_candidates(
            _context,
            candidates,
            records,
            **_kwargs,
        ):
            """Capture the candidate batch and return simple scored items."""
            captured["candidates"] = list(candidates)
            captured["records"] = records
            output = []
            for candidate_index, prompt in enumerate(candidates):
                accuracy = 0.5 + 0.01 * candidate_index
                evaluation = {
                    "metrics": {
                        "accuracy": accuracy,
                        "stable_accuracy": accuracy,
                    }
                }
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
            context = SimpleNamespace(
                initial_prompt="Solve carefully.",
                mode=self.mode,
                train_records=[self.record],
                validation_records=[{**self.record, "validation_fold": 1}],
                rng=random.Random(42),
                evaluator=SimpleNamespace(
                    evaluate=lambda *_args, **_kwargs: {"metrics": {"accuracy": 0.0}}
                ),
                run_dir=Path(temporary_directory),
                optimizer_name="lpo",
                started_at=time.monotonic(),
            )
            args = SimpleNamespace(
                train_sample_size=1,
                feedback_examples=1,
                max_locations=5,
                max_words_per_location=3,
                num_candidates=5,
                top_z=1,
                disable_source_validation_cache=True,
            )
            context.args = args
            prediction = {
                "predicted_answer": "2",
                "correct": False,
                "raw_response": "Incorrect reasoning. <answer>2</answer>",
            }
            with (
                patch(
                    "prompt_optimization.second_stage._get_shared_training_pool",
                    return_value=([self.record], [prediction], {"pool_cache_hit": True}, {}),
                ),
                patch(
                    "prompt_optimization.second_stage.select_incorrect_feedback",
                    return_value=[(self.record, prediction)],
                ),
                patch(
                    "prompt_optimization.second_stage.generate_optimizer_texts",
                    side_effect=[
                        ["<p>Solve <edit>carefully</edit>.</p>"],
                        rewrite_outputs,
                    ],
                ),
                patch(
                    "prompt_optimization.second_stage.evaluate_candidates",
                    side_effect=fake_evaluate_candidates,
                ),
                patch(
                    "prompt_optimization.second_stage.finalize_run",
                    return_value={},
                ),
            ):
                run_lpo(context, args)

        self.assertEqual(len(captured["candidates"]), 6)
        self.assertEqual(captured["candidates"][0], context.initial_prompt)
        self.assertEqual(captured["candidates"][1:], [
            f"Candidate instruction {index}." for index in range(1, 6)
        ])
        self.assertIs(captured["records"], context.validation_records)

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

    def test_validation_can_reuse_three_fixed_300_example_prefixes(self) -> None:
        """Select 300 records per existing fold without creating another file."""
        records = load_qa_records(
            REPO_ROOT / "data" / "processed" / "math500" / "validation.jsonl",
            "math500",
        )

        selected = select_validation_fold_subset(records, 300)

        self.assertEqual(len(selected), 900)
        self.assertEqual(
            {
                sum(
                    int(record["validation_fold"]) == fold for record in selected
                )
                for fold in range(1, 4)
            },
            {300},
        )

    def test_reasoning_conditioned_target_preserves_exact_math_answer(self) -> None:
        """Condition on reasoning but calculate loss only over exact answer text."""
        record = {**self.record, "answer": r"\frac{1}{2}"}

        rendered = render_teacher_forced_qa(
            "Solve this exactly.",
            record,
            self.mode,
            _PlainChatTokenizer(),
            reasoning_trace="First derive an equivalent fraction.",
        )

        target = rendered["text"][rendered["label_start"] : rendered["label_end"]]
        self.assertEqual(target, r"\frac{1}{2}")
        self.assertNotEqual(target, r"\FRAC{1}{2}")
        self.assertTrue(rendered["reasoning_conditioned"])
        self.assertIn("First derive an equivalent fraction.", rendered["text"])

    def test_reasoning_target_replaces_existing_answer_with_gold(self) -> None:
        """Remove a predicted answer before teacher-forcing the gold answer."""
        rendered = render_teacher_forced_qa(
            "Solve this exactly.",
            self.record,
            self.mode,
            _PlainChatTokenizer(),
            reasoning_trace="Derivation. <answer>2</answer>",
        )

        self.assertNotIn("<answer>2</answer>", rendered["text"])
        self.assertEqual(rendered["text"].count("<answer>1</answer>"), 1)
        self.assertEqual(rendered["answer_target_mode"], "replaced_existing_answer")

    def test_reasoning_target_appends_fallback_when_answer_is_missing(self) -> None:
        """Append GreaTer's extractor only when the trace has no answer block."""
        rendered = render_teacher_forced_qa(
            "Solve this exactly.",
            self.record,
            self.mode,
            _PlainChatTokenizer(),
            reasoning_trace="An unfinished derivation.",
        )

        self.assertIn(
            "Therefore, the final answer is <answer>1</answer>",
            rendered["text"],
        )
        self.assertEqual(
            rendered["answer_target_mode"],
            "appended_missing_answer_fallback",
        )

    def test_gradient_subset_balances_correct_and_incorrect_results(self) -> None:
        """Use two outcome buckets without requiring answer-label buckets."""
        import random

        records = [{"id": f"record-{index}"} for index in range(6)]
        predictions = [
            {"correct": value}
            for value in (True, True, True, True, False, False)
        ]

        selected, counts = _balance_gradient_pairs(
            records,
            predictions,
            random.Random(42),
            4,
        )

        self.assertEqual(counts["available_correct"], 4)
        self.assertEqual(counts["available_incorrect"], 2)
        self.assertEqual(counts["selected_correct"], 2)
        self.assertEqual(counts["selected_incorrect"], 2)
        self.assertEqual(sum(pair[1]["correct"] for pair in selected), 2)
        self.assertEqual(len(selected), 4)

    def test_gradient_subset_uses_available_balanced_outcomes(self) -> None:
        """Use the largest balanced subset when one outcome bucket is undersized."""
        import random

        records = [{"id": f"record-{index}"} for index in range(4)]
        predictions = [
            {"correct": value}
            for value in (True, True, True, False)
        ]

        selected, counts = _balance_gradient_pairs(
            records,
            predictions,
            random.Random(42),
            4,
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(counts["selected_correct"], 1)
        self.assertEqual(counts["selected_incorrect"], 1)
        self.assertEqual(counts["shortfall"], 2)

    def test_gradpo_synthesis_uses_math_task_description(self) -> None:
        """Describe Math rather than multiple-choice QA during beam synthesis."""
        prompt = _gradpo_synthesis_prompt(
            "Solve carefully.",
            [
                {
                    "region_rank": 1,
                    "region_text": "carefully",
                    "start_char": 6,
                    "end_char": 15,
                }
            ],
            {1: "step by step"},
            self.mode,
        )

        self.assertIn("mathematical problem-solving", prompt)
        self.assertIn("exact final answer", prompt)
        self.assertNotIn("multiple-choice", prompt)
        self.assertIn("<prompt>", prompt)
        self.assertIn("</prompt>", prompt)

    def test_gradpo_synthesis_extracts_last_complete_tagged_prompt(self) -> None:
        """Use the final complete prompt block and remove leaked span markers."""
        raw_output = (
            "Draft: <prompt>Ignore this draft.</prompt>\n"
            "<prompt>Solve <span_1>systematically</span_1>.</prompt>"
        )

        prompt = _normalize_synthesized_prompt(raw_output)

        self.assertEqual(prompt, "Solve systematically.")

    def test_gradpo_synthesis_rejects_truncated_tagged_prompt(self) -> None:
        """Reject an output that ends before its closing prompt tag."""
        prompt = _normalize_synthesized_prompt(
            "<prompt>Output only the letter corresponding"
        )

        self.assertEqual(prompt, "")

    def test_gradpo_rejects_openbookqa_true_false_candidate(self) -> None:
        """Remove the task-changing true-false replacement only for OpenBookQA."""
        openbookqa_mode = resolve_mode("non_reasoning", "openbookqa")

        candidates, rejected = _filter_gradpo_region_candidates(
            "choice",
            ["true-false", "selection", "option"],
            2,
            openbookqa_mode,
        )

        self.assertEqual(candidates, ["choice", "selection", "option"])
        self.assertEqual(rejected[0]["candidate"], "true-false")
        self.assertIn("restricted", rejected[0]["reason"])

    def test_greater_proposal_uses_math_task_description(self) -> None:
        """Keep GreaTer's token proposal context task-correct for Math."""
        prompt = qa_proposal_header(self.record, self.mode)

        self.assertIn("mathematical problem-solving", prompt)
        self.assertIn("exact final answer", prompt)
        self.assertIn(self.record["question"], prompt)
        self.assertNotIn("multiple-choice", prompt)
        self.assertNotIn("Choices:", prompt)

    def test_greater_requires_same_token_count_after_retokenization(self) -> None:
        """Follow GreaTer by checking token count rather than exact token IDs."""
        tokenizer = _RoundTripTokenizer()

        stable = same_length_single_token_replacement_prompt(tokenizer, [1, 2, 3], 1, 4)
        changed_ids = same_length_single_token_replacement_prompt(
            tokenizer, [1, 2, 3], 1, 9
        )
        changed_length = same_length_single_token_replacement_prompt(
            tokenizer, [1, 2, 3], 1, 8
        )

        self.assertEqual(stable, "1 4 3")
        self.assertEqual(changed_ids, "1 9 3")
        self.assertIsNone(changed_length)

    @patch("prompt_optimization.second_stage._score_objective_candidates")
    def test_gradpo_no_op_branch_preserves_the_exact_source_prompt(
        self,
        mock_score,
    ) -> None:
        """Keep GradPO's unchanged beam branch without an LLM synthesis call."""
        mock_score.return_value = [
            {
                "task_loss": 1.0,
                "fluency_loss": 1.0,
                "combined_score": 1.0,
            }
        ]
        context = SimpleNamespace(
            initial_prompt="Solve carefully.",
            mode=self.mode,
            model_pool=_NoGenerationPool(),
            optimizer_name="gradpo_gen",
            started_at=time.monotonic(),
            logger=SimpleNamespace(event=lambda *_args, **_kwargs: None),
        )

        beam, _trace = _beam_search_replacements(
            context,
            [
                {
                    "region_rank": 1,
                    "region_text": "carefully",
                    "start_char": 6,
                    "end_char": 15,
                }
            ],
            [
                {
                    "region_rank": 1,
                    "candidates": ["carefully"],
                }
            ],
            [self.record],
            model=object(),
            tokenizer=object(),
            beam_width=1,
            selection_batch_size=1,
            fluency_lambda=0.5,
            replacement_mode="llm_synthesis",
            synthesis_max_new_tokens=100,
            synthesis_batch_size=1,
            reasoning_traces=["Reasoning. <answer>2</answer>"],
        )

        self.assertEqual(beam[0]["prompt"], context.initial_prompt)
        self.assertEqual(beam[0]["replacements"], {})
        self.assertTrue(beam[0]["replacement_metadata"]["no_op"])

    @patch("prompt_optimization.second_stage._score_objective_candidates")
    def test_gradpo_beam_synthesis_uses_greedy_decoding(
        self,
        mock_score,
    ) -> None:
        """Keep constrained beam synthesis deterministic rather than sampled."""
        mock_score.return_value = [
            {"task_loss": 1.0, "fluency_loss": 1.0, "combined_score": 1.0},
            {"task_loss": 2.0, "fluency_loss": 1.0, "combined_score": 2.0},
        ]
        generate = MagicMock(
            return_value=["<prompt>Solve systematically.</prompt>"]
        )
        context = SimpleNamespace(
            initial_prompt="Solve carefully.",
            mode=self.mode,
            model_pool=SimpleNamespace(generate=generate),
            optimizer_name="gradpo_gen",
            started_at=time.monotonic(),
            logger=SimpleNamespace(event=lambda *_args, **_kwargs: None),
        )

        _beam_search_replacements(
            context,
            [
                {
                    "region_rank": 1,
                    "region_text": "carefully",
                    "start_char": 6,
                    "end_char": 15,
                }
            ],
            [
                {
                    "region_rank": 1,
                    "candidates": ["carefully", "systematically"],
                }
            ],
            [self.record],
            model=object(),
            tokenizer=object(),
            beam_width=2,
            selection_batch_size=1,
            fluency_lambda=0.5,
            replacement_mode="llm_synthesis",
            synthesis_max_new_tokens=100,
            synthesis_batch_size=1,
            reasoning_traces=["Reasoning. <answer>2</answer>"],
        )

        generate.assert_called_once()
        self.assertFalse(generate.call_args.kwargs["do_sample"])

    @patch("prompt_optimization.second_stage._score_objective_candidates")
    def test_gradpo_beam_synthesis_discards_truncated_output(
        self,
        mock_score,
    ) -> None:
        """Discard a synthesis candidate that lacks a closing prompt tag."""
        mock_score.return_value = [
            {"task_loss": 1.0, "fluency_loss": 1.0, "combined_score": 1.0},
        ]
        context = SimpleNamespace(
            initial_prompt="Solve carefully.",
            mode=self.mode,
            model_pool=SimpleNamespace(
                generate=MagicMock(
                    return_value=["<prompt>Solve systematically."]
                )
            ),
            optimizer_name="gradpo_gen",
            started_at=time.monotonic(),
            logger=SimpleNamespace(event=lambda *_args, **_kwargs: None),
        )

        beam, _trace = _beam_search_replacements(
            context,
            [
                {
                    "region_rank": 1,
                    "region_text": "carefully",
                    "start_char": 6,
                    "end_char": 15,
                }
            ],
            [
                {
                    "region_rank": 1,
                    "candidates": ["carefully", "systematically"],
                }
            ],
            [self.record],
            model=object(),
            tokenizer=object(),
            beam_width=2,
            selection_batch_size=1,
            fluency_lambda=0.5,
            replacement_mode="llm_synthesis",
            synthesis_max_new_tokens=100,
            synthesis_batch_size=1,
            reasoning_traces=["Reasoning. <answer>2</answer>"],
        )

        self.assertEqual(len(beam), 1)
        self.assertEqual(beam[0]["prompt"], context.initial_prompt)
        self.assertEqual(beam[0]["replacements"], {})

    @patch("prompt_optimization.second_stage._score_objective_candidates")
    def test_gradpo_beam_accepts_complete_tagged_synthesis(
        self,
        mock_score,
    ) -> None:
        """Use the complete tagged synthesis without comparing it to direct edits."""
        mock_score.return_value = [
            {"task_loss": 1.0, "fluency_loss": 1.0, "combined_score": 1.0},
            {"task_loss": 2.0, "fluency_loss": 1.0, "combined_score": 2.0},
        ]
        context = SimpleNamespace(
            initial_prompt="Solve carefully.",
            mode=self.mode,
            model_pool=SimpleNamespace(
                generate=MagicMock(
                    return_value=[
                        "<prompt>Please solve systematically and verify.</prompt>"
                    ]
                )
            ),
            optimizer_name="gradpo_gen",
            started_at=time.monotonic(),
            logger=SimpleNamespace(event=lambda *_args, **_kwargs: None),
        )

        beam, _trace = _beam_search_replacements(
            context,
            [
                {
                    "region_rank": 1,
                    "region_text": "carefully",
                    "start_char": 6,
                    "end_char": 15,
                }
            ],
            [
                {
                    "region_rank": 1,
                    "candidates": ["carefully", "systematically"],
                }
            ],
            [self.record],
            model=object(),
            tokenizer=object(),
            beam_width=2,
            selection_batch_size=1,
            fluency_lambda=0.5,
            replacement_mode="llm_synthesis",
            synthesis_max_new_tokens=100,
            synthesis_batch_size=1,
            reasoning_traces=["Reasoning. <answer>2</answer>"],
        )

        synthesized = next(item for item in beam if item["replacements"])
        metadata = synthesized["replacement_metadata"]
        self.assertEqual(
            synthesized["prompt"],
            "Please solve systematically and verify.",
        )
        self.assertFalse(metadata["rejected"])
        self.assertIsNone(metadata["rejection_reason"])

    @unittest.skipIf(torch is None, "PyTorch is not installed in this test environment.")
    def test_answer_loss_weights_problems_not_answer_token_counts(self) -> None:
        """Give short and long symbolic answers equal problem-level weight."""
        logits = torch.zeros((2, 5, 3), dtype=torch.float32)
        input_ids = torch.zeros((2, 5), dtype=torch.long)
        input_ids[0, 1] = 0
        input_ids[1, 1:4] = 1
        logits[0, 0] = torch.tensor([4.0, 0.0, 0.0])
        logits[1, 0:3] = torch.tensor([4.0, 0.0, 0.0])

        loss = _label_loss_from_logits(
            logits,
            input_ids,
            [[1], [1, 2, 3]],
            reduction="mean",
        )
        short_loss = functional.cross_entropy(logits[0, 0:1], input_ids[0, 1:2])
        long_loss = functional.cross_entropy(logits[1, 0:3], input_ids[1, 1:4])

        self.assertTrue(torch.allclose(loss, (short_loss + long_loss) / 2))


if __name__ == "__main__":
    unittest.main()
