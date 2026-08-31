"""Create and summarize per-prompt token-usage records."""

from __future__ import annotations

from typing import Sequence, TypeAlias


TokenUsage: TypeAlias = dict[str, int]


def create_token_usage(input_tokens: int, output_tokens: int) -> TokenUsage:
    """Create input, output, and combined token counts for one prompt."""
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def summarize_token_usage(token_usages: Sequence[TokenUsage]) -> dict[str, object]:
    """Calculate total and average token counts across an inference run."""
    if not token_usages:
        raise ValueError("Cannot summarize an empty token-usage sequence.")

    count_names = ("input_tokens", "output_tokens", "total_tokens")
    totals = {
        count_name: sum(token_usage[count_name] for token_usage in token_usages)
        for count_name in count_names
    }
    example_count = len(token_usages)
    averages = {
        count_name: totals[count_name] / example_count for count_name in count_names
    }
    return {
        "examples": example_count,
        "totals": totals,
        "averages": averages,
    }
