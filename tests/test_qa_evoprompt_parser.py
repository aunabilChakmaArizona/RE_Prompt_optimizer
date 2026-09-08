"""Regression tests for QA EvoPrompt output parsing."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "codes"))

from prompt_optimization.first_stage import _parse_final_evoprompt_child


class SimpleTokenizer:
    """Provide deterministic token counts without loading a model."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        """Split words and punctuation into simple test tokens."""
        del add_special_tokens
        return re.findall(r"\w+|[^\w\s]", text)


class EvoPromptParserTests(unittest.TestCase):
    """Check final-tag selection and trivial-fragment rejection."""

    def test_selects_last_tagged_prompt(self) -> None:
        """Ignore an explanatory tag pair before the actual final prompt."""
        raw_output = (
            "Bracket the result with <prompt> and </prompt>.\n"
            "Final Prompt: <prompt>Answer the supplied question carefully.</prompt>"
        )

        candidate, tagged, token_count, rejection_reason = (
            _parse_final_evoprompt_child(raw_output, SimpleTokenizer())
        )

        self.assertEqual(candidate, "Answer the supplied question carefully.")
        self.assertEqual(tagged[0], "and")
        self.assertGreaterEqual(token_count, 5)
        self.assertIsNone(rejection_reason)

    def test_rejects_trivial_last_tag(self) -> None:
        """Reject a one-token candidate when no meaningful final prompt follows."""
        candidate, tagged, token_count, rejection_reason = (
            _parse_final_evoprompt_child(
                "Final Prompt: <prompt>and</prompt>",
                SimpleTokenizer(),
            )
        )

        self.assertIsNone(candidate)
        self.assertEqual(tagged, ["and"])
        self.assertEqual(token_count, 1)
        self.assertEqual(rejection_reason, "too_few_tokens")


if __name__ == "__main__":
    unittest.main()
