"""CPU checks for fresh Qwen commands and exact-prompt old/new comparisons."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))

from report_openbookqa_test_rerun import build_comparison
from run_openbookqa_final_test_evaluation import REFINERS, SOURCE_SLOTS


def make_summaries() -> tuple[dict, dict, dict]:
    """Build complete old/new matrices and a compatible solo result without model outputs."""
    rows = [{"row_id": "initial", "stage": "initial", "source": "initial", "method": "initial",
             "parent_row_id": "", "correct": 400}]
    for source in SOURCE_SLOTS:
        rows.append({"row_id": f"first_stage_{source}", "stage": "first_stage", "source": source,
                     "method": "first", "parent_row_id": "initial", "correct": 400})
        for method in REFINERS:
            rows.append({"row_id": f"second_stage_{source}_{method}", "stage": "second_stage",
                         "source": source, "method": method, "parent_row_id": f"first_stage_{source}",
                         "correct": 401})
    for row in rows:
        row.update(total=500, accuracy=row["correct"] / 500,
                   instruction_prompt=row["row_id"] + " instruction", prompt_sha256=row["row_id"] + "hash",
                   optimization_code=row["row_id"] + "code", evaluation_seed=123,
                   missing_answer_tags=0, invalid_choice_labels=0,
                   gain_percentage_points=(100*(row["correct"] / 500 - 0.8)
                                           if row["parent_row_id"] else None))
    protocol = {"task": "openbookqa", "qa_mode": "non_reasoning", "test_sha256": "testhash",
                "record_set_id": "ids", "test_size": 500, "run_count": 1, "seed": 42,
                "answer_instruction": "fixed tagged answer", "enable_thinking": False,
                "do_sample": True, "max_new_tokens": 10}
    old = {"family": "qwen", "model": "Qwen/Qwen3-4B", "code": "old", "protocol": protocol,
           "decoding": {"temperature": 0.6}, "rows": rows, "run_dir": "/old"}
    new = copy.deepcopy(old)
    new.update(code="fresh", run_dir="/fresh")
    new["protocol"]["max_new_tokens"] = 16
    solo = {"code": "solo", "model": old["model"], "prompt": rows[0]["instruction_prompt"],
            "run_count": 1, "test_size": 500,
            "runs": [{"accuracy": 0.8, "evaluation_seed": 123,
                      "metrics": {"missing_answer_tags": 0, "invalid_choice_labels": 0}}]}
    return old, new, solo


class RerunComparisonTests(unittest.TestCase):
    """Keep all rows and detect accidental source, prompt, or test changes."""

    def test_changed_limits_are_allowed_and_all_methods_remain(self) -> None:
        """Allow the intended 10-to-16 token change without removing unchanged prompts."""
        report = build_comparison(*make_summaries())
        self.assertIn("previous=10, fresh=16", report)
        self.assertIn("unchanged prompts included", report)
        for method in ("GradPO-Gen", "GradPO-Prob", "GradPO-Gen-Random", "LPO", "GreaTer", "GreaTer-TG"):
            self.assertEqual(sum(line.startswith("| " + method.ljust(17)) for line in report.splitlines()), 1)

    def test_changed_score_is_compared_against_its_own_source(self) -> None:
        """Average a one-question change over all five source slots, not just improved rows."""
        old, new, solo = make_summaries()
        row = next(r for r in new["rows"] if r["row_id"] == "second_stage_rpo5_gradpo_gen")
        row.update(correct=402, accuracy=402/500, gain_percentage_points=100*(402/500-0.8))
        report = build_comparison(old, new, solo)
        average = next(line for line in report.splitlines() if line.startswith("| GradPO-Gen "))
        self.assertIn("80.24", average)
        self.assertIn("+0.04", average)
        self.assertIn("1/4/0", average)

    def test_changed_prompt_or_test_is_rejected(self) -> None:
        """Prevent comparing a different instruction or test split as a rerun."""
        old, new, solo = make_summaries()
        new["rows"][1]["instruction_prompt"] = "another prompt"
        with self.assertRaisesRegex(ValueError, "Selected prompt/source changed"):
            build_comparison(old, new, solo)
        old, new, solo = make_summaries()
        new["protocol"]["test_sha256"] = "different test"
        with self.assertRaisesRegex(ValueError, "test_sha256"):
            build_comparison(old, new, solo)

    def test_incomplete_matrix_or_wrong_gain_is_rejected(self) -> None:
        """Catch missing attempts and incorrectly saved source-relative gains."""
        old, new, solo = make_summaries()
        new["rows"].pop()
        with self.assertRaisesRegex(ValueError, "complete 36-row"):
            build_comparison(old, new, solo)
        old, new, solo = make_summaries()
        new["rows"][-1]["gain_percentage_points"] = 100
        with self.assertRaisesRegex(ValueError, "Invalid source-relative gain"):
            build_comparison(old, new, solo)

    def test_qwen_script_arguments_without_gpu_calls(self) -> None:
        """Inspect both fresh commands and the comparison call using echo instead of Python."""
        env = {**os.environ, "PYTHON_BIN": "/bin/echo", "RUN_TAG": "cpu_test",
               "RUN_GPU": "3", "FINAL_TEST_DRY_RUN": "0", "FINAL_TEST_GPU_RATIO": "0.8"}
        result = subprocess.run(["bash", str(ROOT / "codes/run_openbookqa_qwen_matched_test.sh")],
                                cwd="/tmp", env=env, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.count("--gpu-memory-utilization 0.8"), 2)
        self.assertEqual(result.stdout.count("--max-new-tokens 16"), 3)
        self.assertIn("--vllm-max-model-len 40960", result.stdout)
        self.assertIn("report_openbookqa_test_rerun.py", result.stdout)
        self.assertNotIn("--overwrite", result.stdout)
        self.assertNotIn("--family gemma", result.stdout)


if __name__ == "__main__":
    unittest.main()
