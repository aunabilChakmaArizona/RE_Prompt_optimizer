"""Check portable Gemma Math tuning commands without loading a GPU model."""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
from run_qa_promptopt_gradpo import parse_args


class GemmaMathTuningScriptTests(unittest.TestCase):
    """Verify the eight-run plan, fixed budgets, sources, and GPU ownership."""

    def dry_run(self, **settings):
        """Preview expanded commands with controlled local or Slurm variables."""
        env = {
            key: value for key, value in os.environ.items()
            if not key.startswith("MATH_TUNING_")
            and key not in ("SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES")
        }
        env.update(MATH_TUNING_DRY_RUN="1", **settings)
        result = subprocess.run(
            ["bash", "codes/run_math_gradpo_tuning_gemma.sh"],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        commands = [shlex.split(line) for line in result.stdout.splitlines()]
        return result, commands

    def parsed_commands(self, commands):
        """Parse the actual emitted arguments using the existing runner parser."""
        parsed = []
        for command in commands:
            self.assertEqual(command[:3], ["python", "-u", "codes/run_qa_promptopt_gradpo.py"])
            with patch.object(sys, "argv", [command[2], *command[3:]]):
                parsed.append(parse_args())
        return parsed

    def test_default_matrix_and_fixed_budgets(self):
        """Run four configurations on both sources without changing other settings."""
        result, commands = self.dry_run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 8)
        args = self.parsed_commands(commands)
        expected = [(5, 3, 0.60)] * 2 + [(5, 5, 0.60)] * 2
        expected += [(5, 5, 0.45)] * 2 + [(7, 5, 0.45)] * 2
        self.assertEqual(
            [(a.num_edit_regions, a.max_region_tokens, a.region_expansion_threshold) for a in args],
            expected,
        )
        self.assertEqual(len({a.code for a in args}), 8)
        for index, a in enumerate(args):
            with self.subTest(code=a.code):
                self.assertEqual(a.model, "google/gemma-3-4b-it")
                self.assertEqual(a.qa_task, "math500")
                self.assertEqual(a.qa_mode, "reasoning")
                self.assertEqual(a.variant, "gen")
                self.assertEqual(a.train_sample_size, 600)
                self.assertEqual(a.gradient_sample_size, 200)
                self.assertEqual(a.gradient_batch_size, 1)
                self.assertEqual(a.selection_batch_size, 4)
                self.assertEqual(a.num_region_candidates, 7)
                self.assertEqual(a.beam_width, 5)
                self.assertEqual(a.fluency_lambda, 0.5)
                self.assertEqual(a.target_max_new_tokens, 4096)
                self.assertEqual(a.candidate_max_new_tokens, 10000)
                self.assertEqual(a.synthesis_max_new_tokens, 10000)
                self.assertEqual(a.validation_fold_size, 300)
                self.assertEqual(a.validation_std_penalty, 1.0)
                self.assertEqual(a.backend, "dual")
                self.assertEqual(a.objective_scoring_backend, "vllm")
                self.assertEqual(a.objective_scoring_batch_size, 128)
                self.assertEqual(a.final_evaluation_backend, "vllm")
                self.assertEqual(a.gpu_memory_utilization, 0.5)
                self.assertEqual(a.dual_vllm_gpu_memory_utilization, 0.5)
                self.assertEqual(a.device, "cuda:0")
                self.assertEqual(a.hf_device, "cuda:0")
                self.assertEqual(a.seed, 42)
                self.assertTrue(a.vllm_disable_images)
                self.assertTrue(a.overwrite)
                self.assertFalse(a.disable_gradient_cache)
                self.assertFalse(a.vllm_conservative_settings)
                source = "5" if index % 2 == 0 else "10"
                self.assertTrue(a.initial_prompt_file.endswith(f"/rpo{source}.txt"))
                self.assertTrue((ROOT / a.initial_prompt_file).is_file())

    def test_portable_sources_are_verbatim_original_snapshots(self):
        """Check exported prompt hashes without requiring ignored output folders."""
        hashes = {
            "rpo5.txt": "078c8d9274076e9d5ba27bfcffca256f55fe713e0fdc272efea3189d3c6fdcc9",
            "rpo10.txt": "b7454bf90b34fbeeed151e59c9fb55924748ee29a09a63851b2148d11cf0cb28",
        }
        prompts = []
        for filename, expected in hashes.items():
            content = (ROOT / "experiment_tracking/second_stage/math_gemma_sources" / filename).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected)
            self.assertGreater(len(content.decode().split()), 50)
            prompts.append(content)
        self.assertNotEqual(*prompts)

    def test_resume_and_single_attempt_range(self):
        """Run only the requested remaining commands or initial smoke attempt."""
        result, commands = self.dry_run(MATH_TUNING_START="3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 6)
        self.assertEqual(self.parsed_commands(commands)[0].max_region_tokens, 5)
        result, commands = self.dry_run(MATH_TUNING_END="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 1)

    def test_slurm_mask_is_preserved_and_local_gpu_override_rejected(self):
        """Accept Slurm's visible GPU but prevent replacing it with a local index."""
        result, commands = self.dry_run(SLURM_JOB_ID="123", CUDA_VISIBLE_DEVICES="2")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 8)
        result, commands = self.dry_run(
            SLURM_JOB_ID="123", CUDA_VISIBLE_DEVICES="2", MATH_TUNING_GPU="3",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(commands, [])
        self.assertIn("scheduler-assigned GPU", result.stderr)

    def test_invalid_ranges_launch_nothing(self):
        """Reject incorrect ranges before any optimizer command is executed."""
        for settings in (
            {"MATH_TUNING_START": "0"},
            {"MATH_TUNING_END": "9"},
            {"MATH_TUNING_START": "7", "MATH_TUNING_END": "2"},
        ):
            with self.subTest(settings=settings):
                result, commands = self.dry_run(**settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(commands, [])


if __name__ == "__main__":
    unittest.main()
