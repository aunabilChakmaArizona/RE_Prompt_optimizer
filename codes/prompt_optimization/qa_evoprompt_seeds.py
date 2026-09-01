"""Fixed QA seed prompts for the five-member EvoPrompt-DE population."""

from __future__ import annotations


# The labels make the provisional source of each seed explicit. Replace these
# prompts with the final curated human and automatically generated prompts while
# preserving the same five-member experimental population.
QA_EVOPROMPT_SEEDS = {
    "reasoning": [
        {
            "label": "human_1_placeholder",
            "prompt": (
                "Read the question and all answer choices carefully, reason "
                "step by step, and select the single best answer."
            ),
        },
        {
            "label": "human_2_placeholder",
            "prompt": (
                "Use relevant knowledge and logical reasoning to compare the "
                "choices, check your conclusion, and choose the best answer."
            ),
        },
        {
            "label": "automatic_1_placeholder",
            "prompt": (
                "Analyze what the question asks, identify the relevant facts, "
                "eliminate inconsistent choices, and select the best-supported "
                "answer after reasoning carefully."
            ),
        },
        {
            "label": "automatic_2_placeholder",
            "prompt": (
                "Solve the question systematically by interpreting it, evaluating "
                "each choice, verifying the conclusion, and selecting the correct "
                "answer."
            ),
        },
    ],
    "non_reasoning": [
        {
            "label": "human_1_placeholder",
            "prompt": (
                "Read the question and all answer choices carefully, then select "
                "the single best answer directly."
            ),
        },
        {
            "label": "human_2_placeholder",
            "prompt": (
                "Use relevant knowledge to compare the choices and choose the "
                "best answer without providing an explanation."
            ),
        },
        {
            "label": "automatic_1_placeholder",
            "prompt": (
                "Identify what the question asks and select the option that is "
                "best supported by the available choices."
            ),
        },
        {
            "label": "automatic_2_placeholder",
            "prompt": (
                "Evaluate the choices and return the most appropriate answer "
                "directly and concisely."
            ),
        },
    ],
}
