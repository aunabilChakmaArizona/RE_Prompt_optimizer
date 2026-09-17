"""Dry-run the adaptive Math tuning shell script without loading a model."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codes"))
from run_qa_promptopt_gradpo import parse_args


class MathGradPOTuningScriptTests(unittest.TestCase):
    """Verify fixed and adaptive arguments through the real argument parser."""

    def dry_run(self, **settings):
        """Capture Python arguments using a harmless shell function replacement."""
        env = {key: value for key, value in os.environ.items() if not key.startswith("MATH_TUNING_")}
        env.update(settings)
        result = subprocess.run(
            ["bash", "-c", 'python() { printf "%s\\0" "$@"; printf "\\n"; }; export -f python; bash codes/run_math_gradpo_tuning_qwen.sh'],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        commands = [line.rstrip("\0").split("\0") for line in result.stdout.splitlines()]
        return result, commands

    def check_commands(self, commands):
        """Parse every captured command and verify shared source/cache settings."""
        parsed = []
        for command in commands:
            self.assertEqual(command[:2], ["-u", "codes/run_qa_promptopt_gradpo.py"])
            with patch.object(sys, "argv", [command[1], *command[2:]]):
                args = parse_args()
            self.assertEqual(args.train_sample_size, 1600)
            self.assertEqual(args.gradient_sample_size, 200)
            self.assertEqual(args.num_region_candidates, 10)
            self.assertEqual(args.beam_width, 10)
            self.assertEqual(args.fluency_lambda, 0.5)
            self.assertEqual(args.target_max_new_tokens, 8192)
            self.assertEqual(args.validation_fold_size, 300)
            self.assertEqual(args.backend, "dual")
            self.assertEqual(args.hf_device, "cuda:0")
            self.assertFalse(args.disable_gradient_cache)
            self.assertTrue(args.overwrite)
            self.assertTrue((ROOT / args.initial_prompt_file).is_file())
            self.assertIn("error3_tokenlimit", args.initial_prompt_file)
            parsed.append(args)
        return parsed

    def test_default_runs_only_four_width_attempts(self):
        """Use the transferred configuration and wider cap on both sources."""
        result, commands = self.dry_run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(commands), 4)
        parsed = self.check_commands(commands)
        self.assertEqual([a.max_region_tokens for a in parsed], [3, 3, 5, 5])
        self.assertTrue(all(a.num_edit_regions == 3 for a in parsed))
        self.assertTrue(all(a.region_expansion_threshold == 0.45 for a in parsed))
        self.assertEqual(len({a.code for a in parsed}), 4)

    def test_adaptive_phases_require_and_use_selected_values(self):
        """Run two attempts per later phase using the supplied shared T/H."""
        for stage, extra in (
            ("expansion", {}),
            ("count", {"MATH_TUNING_H": "0.3"}),
        ):
            with self.subTest(stage=stage):
                result, commands = self.dry_run(
                    MATH_TUNING_STAGE=stage, MATH_TUNING_T="5", **extra,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(commands), 2)
                parsed = self.check_commands(commands)
                self.assertTrue(all(a.max_region_tokens == 5 for a in parsed))
                self.assertTrue(all(a.region_expansion_threshold == 0.3 for a in parsed))
                self.assertTrue(all(a.num_edit_regions == (5 if stage == "count" else 3) for a in parsed))

    def test_unselected_or_invalid_values_launch_nothing(self):
        """Reject adaptive phases without valid decisions before Python is called."""
        for settings in (
            {"MATH_TUNING_STAGE": "expansion"},
            {"MATH_TUNING_STAGE": "count", "MATH_TUNING_T": "3"},
            {"MATH_TUNING_STAGE": "expansion", "MATH_TUNING_T": "7"},
            {"MATH_TUNING_STAGE": "unknown"},
        ):
            with self.subTest(settings=settings):
                result, commands = self.dry_run(**settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(commands, [])


if __name__ == "__main__":
    unittest.main()
