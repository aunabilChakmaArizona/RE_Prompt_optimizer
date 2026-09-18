"""CPU regression tests for the frozen, single-run OpenBookQA test matrix."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "codes"))

import run_openbookqa_final_test_evaluation as runner
from report_openbookqa_final_test_results import combine_reports


def make_selections(family: str = "qwen") -> list[dict[str, str]]:
    """Build a complete labeled matrix with only three unique instructions."""
    rows = [{"family": family, "row_id": "initial", "stage": "initial", "source": "initial",
             "actual_iteration": "0", "method": "initial", "parent_row_id": "",
             "instruction_prompt": "Initial instruction."}]
    for source in runner.SOURCE_SLOTS:
        rows.append({"family": family, "row_id": f"first_stage_{source}", "stage": "first_stage",
                     "source": source, "actual_iteration": "3", "method": "rpo",
                     "parent_row_id": "initial", "instruction_prompt": "Source instruction."})
        for method in runner.REFINERS:
            rows.append({"family": family, "row_id": f"second_stage_{source}_{method}",
                         "stage": "second_stage", "source": source, "actual_iteration": "3",
                         "method": method, "parent_row_id": f"first_stage_{source}",
                         "instruction_prompt": ("Edited instruction." if method == "gradpo_gen"
                                                else "Source instruction.")})
    for row in rows:
        row.update(optimization_code="saved_optimization_code", prompt_file="saved_prompt.txt",
                   prompt_sha256="a" * 64)
    return rows


def make_evaluation(prompt: str, accuracy: float) -> dict:
    """Provide a small complete evaluation for score-to-method mapping checks."""
    return {"instruction_prompt": prompt, "evaluation_seed": 123,
            "metrics": {"correct": int(accuracy * 500), "total": 500, "accuracy": accuracy,
                        "missing_answer_tags": 0, "invalid_choice_labels": 0,
                        "token_usage": {"averages": {"output_tokens": 7.0}}}}


class FakeVLLMPool:
    """Generate cheap tagged outputs while recording one bulk inference request."""

    uses_vllm_generation = True

    def __init__(self) -> None:
        """Initialize generation and shutdown tracking without a GPU."""
        self.calls = []
        self.closed = False

    def generate(self, role, prompts, **kwargs):
        """Return option A with token counts for every submitted question."""
        self.calls.append((role, list(prompts), kwargs))
        return (["<answer>A</answer>"] * len(prompts),
                [{"input_tokens": 50, "output_tokens": 7, "total_tokens": 57}
                 for _ in prompts])

    def unload_all(self) -> None:
        """Mark the sole target-model pool as closed."""
        self.closed = True


class OpenBookQAFinalTestTests(unittest.TestCase):
    """Check source mapping, deduplication, artifacts, and protocol safety."""

    def test_current_manifest_covers_both_models(self) -> None:
        """Check every selected file/hash when local optimizer outputs are available."""
        if not (REPO_ROOT / "outputs/qa_prompt_optimization/non_reasoning").exists():
            self.skipTest("Selected optimizer artifacts are local and Git-ignored.")
        for family in runner.MODEL_IDS:
            rows = runner.load_selected_prompts(runner.DEFAULT_MANIFEST, family)
            self.assertEqual(len(rows), 36)
            self.assertEqual(sum(row["stage"] == "second_stage" for row in rows), 30)
            self.assertEqual(sum(row["stage"] == "first_stage" for row in rows), 5)

    def test_unchanged_methods_share_scores_and_prediction_files(self) -> None:
        """Retain all 36 rows but score the three unique instructions only once."""
        selected = make_selections()
        prompts = runner.unique_instructions(selected)
        self.assertEqual(len(prompts), 3)
        evaluations = [make_evaluation(prompt, score) for prompt, score in zip(prompts, (0.7, 0.8, 0.82))]
        rows = runner.build_result_rows(selected, evaluations, Path("/test-output"))
        by_id = {row["row_id"]: row for row in rows}
        unchanged = by_id["second_stage_rpo5_lpo"]
        source = by_id["first_stage_rpo5"]
        self.assertEqual(len(rows), 36)
        self.assertEqual(unchanged["predictions_file"], source["predictions_file"])
        self.assertEqual(unchanged["gain_percentage_points"], 0.0)
        self.assertFalse(unchanged["changed_from_source"])
        self.assertAlmostEqual(by_id["second_stage_rpo5_gradpo_gen"]["gain_percentage_points"], 2.0)
        self.assertAlmostEqual(source["gain_percentage_points"], 10.0)

    def test_main_batches_all_pairs_once_and_saves_outputs(self) -> None:
        """Exercise the real evaluator and artifact writer with one mocked model pool."""
        pool = FakeVLLMPool()
        with tempfile.TemporaryDirectory() as directory:
            argv = ["runner", "--family", "qwen", "--code", "cpu_test", "--output-root", directory]
            with patch.object(sys, "argv", argv), patch.object(runner, "load_selected_prompts", return_value=make_selections()), \
                 patch.object(runner, "ModelPool", return_value=pool) as constructor, contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            constructor.assert_called_once()
            self.assertEqual(constructor.call_args.kwargs["backend"], "vllm")
            self.assertIsNone(constructor.call_args.kwargs["optimizer_model_id"])
            self.assertEqual(len(pool.calls), 1)
            role, prompts, kwargs = pool.calls[0]
            self.assertEqual(role, "target")
            self.assertEqual(len(prompts), 1500)
            self.assertEqual(kwargs["seeds"][:500], kwargs["seeds"][500:1000])
            self.assertTrue(kwargs["do_sample"])
            self.assertFalse(kwargs["enable_thinking"])
            self.assertEqual(kwargs["max_new_tokens"], 10)
            self.assertTrue(pool.closed)
            run_dir = Path(directory) / "non_reasoning/selected_prompts/cpu_test"
            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(len(summary["rows"]), 36)
            self.assertEqual(summary["protocol"]["run_count"], 1)
            self.assertEqual(summary["unique_prompt_count"], 3)
            self.assertEqual(len(list((run_dir / "evaluations").glob("*_predictions.jsonl"))), 3)
            predictions = Path(summary["rows"][0]["predictions_file"]).read_text().splitlines()
            self.assertEqual(len(predictions), 500)
            self.assertEqual(json.loads(predictions[0])["token_usage"]["output_tokens"], 7)
            self.assertIn("second_stage/lpo", (run_dir / "results.txt").read_text())

    def test_five_seed_mode_keeps_one_model_and_saves_each_run_separately(self) -> None:
        """Use one model load but five generation calls and five independent result trees."""
        pool = FakeVLLMPool()
        seeds = [42, 1, 100, 1000, 10000]
        with tempfile.TemporaryDirectory() as directory:
            argv = ["runner", "--family", "qwen", "--code", "five_seed_cpu",
                    "--output-root", directory, "--seeds", *map(str, seeds)]
            with patch.object(sys, "argv", argv), \
                 patch.object(runner, "load_selected_prompts", return_value=make_selections()), \
                 patch.object(runner, "ModelPool", return_value=pool) as constructor, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main()
            constructor.assert_called_once()
            self.assertEqual(len(pool.calls), 5)
            self.assertTrue(pool.closed)
            first_seed_lists = [call[2]["seeds"] for call in pool.calls]
            self.assertEqual(len({values[0] for values in first_seed_lists}), 5)
            self.assertTrue(all(values[:500] == values[500:1000] for values in first_seed_lists))
            run_dir = Path(directory) / "non_reasoning/selected_prompts/five_seed_cpu"
            progress = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(progress["status"], "complete")
            self.assertEqual(progress["completed_runs"], 5)
            self.assertEqual(progress["seeds"], seeds)
            self.assertNotIn("rows", progress)
            for seed in seeds:
                summary = json.loads((run_dir / f"seed_{seed}/summary.json").read_text())
                self.assertEqual(summary["protocol"]["seed"], seed)
                self.assertEqual(summary["protocol"]["run_count"], 1)
                self.assertEqual(len(summary["rows"]), 36)
                self.assertEqual(len(list((run_dir / f"seed_{seed}/evaluations").glob(
                    "*_predictions.jsonl"
                ))), 3)
            self.assertIn("No mean/std", (run_dir / "results.txt").read_text())

    def test_combined_report_rejects_different_protocols(self) -> None:
        """Prevent combining differently seeded or differently capped test results."""
        selected = make_selections()
        evaluations = [make_evaluation(prompt, 0.8) for prompt in runner.unique_instructions(selected)]
        rows = runner.build_result_rows(selected, evaluations, Path("/test"))
        qwen = {"family": "qwen", "protocol": {"seed": 42}, "rows": rows}
        gemma = {"family": "gemma", "protocol": {"seed": 43}, "rows": rows}
        with self.assertRaisesRegex(ValueError, "different"):
            combine_reports([qwen, gemma])

    def test_combined_report_includes_all_rows_and_model_specific_sampling(self) -> None:
        """Combine both complete models without requiring identical model-default temperatures."""
        summaries = []
        for family in runner.MODEL_IDS:
            selected = make_selections(family)
            evaluations = [make_evaluation(prompt, 0.8) for prompt in runner.unique_instructions(selected)]
            summaries.append({
                "family": family, "code": runner.FINAL_CODES[family], "model": runner.MODEL_IDS[family],
                "protocol": {"seed": 42, "max_new_tokens": 10, "test_size": 500},
                "test_path": "test.jsonl", "unique_prompt_count": 3,
                "decoding": runner.model_default_sampling_parameters(runner.MODEL_IDS[family]),
                "rows": runner.build_result_rows(selected, evaluations, Path("/test")),
            })
        report = combine_reports(summaries)
        self.assertEqual(report.count("| second_stage/"), 60)
        self.assertIn("'temperature': 0.6", report)
        self.assertIn("'temperature': 1.0", report)
        self.assertIn("72 rows", report)

    def test_shell_scripts_dry_run_without_inference(self) -> None:
        """Check both real launch scripts and selected files using CPU-only dry runs."""
        if not (REPO_ROOT / "outputs/qa_prompt_optimization/non_reasoning").exists():
            self.skipTest("Selected optimizer artifacts are local and Git-ignored.")
        env = {**os.environ, "PYTHON_BIN": sys.executable, "FINAL_TEST_DRY_RUN": "1"}
        for family in runner.MODEL_IDS:
            path = REPO_ROOT / f"codes/run_openbookqa_final_test_{family}.sh"
            result = subprocess.run(["bash", str(path)], cwd="/tmp", env=env,
                                    capture_output=True, text=True, check=True)
            self.assertIn("decoding runs=1", result.stdout)
            self.assertIn("Dry run passed", result.stdout)

    def test_batch_parser_defaults_to_eighty_percent_gpu_memory(self) -> None:
        """Reserve 80% rather than 90% when calling the batch evaluator directly."""
        with patch.object(sys, "argv", ["runner", "--family", "qwen", "--code", "cpu_check"]):
            self.assertEqual(runner.parse_args().gpu_memory_utilization, 0.8)

    def test_all_launch_scripts_pass_eighty_percent_gpu_memory(self) -> None:
        """Inspect launch arguments with echo instead of loading any model."""
        if not (REPO_ROOT / "outputs/qa_prompt_optimization/non_reasoning").exists():
            self.skipTest("Selected optimizer artifacts are local and Git-ignored.")
        env = {**os.environ, "PYTHON_BIN": "/bin/echo", "FINAL_TEST_DRY_RUN": "0"}
        env.pop("FINAL_TEST_GPU_RATIO", None)
        for name, expected_calls in (("qwen", 36), ("gemma", 36), ("batched", 2)):
            path = REPO_ROOT / f"codes/run_openbookqa_final_test_{name}.sh"
            result = subprocess.run(["bash", str(path)], cwd="/tmp", env=env,
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.count("--gpu-memory-utilization 0.8"), expected_calls)


if __name__ == "__main__":
    unittest.main()
