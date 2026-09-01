"""Fixed QA seed prompts for the five-member EvoPrompt-DE population."""

from __future__ import annotations


# The labels make the provisional source of each seed explicit. Replace these
# prompts with the final curated human and automatically generated prompts while
# preserving the same five-member experimental population.
QA_EVOPROMPT_SEEDS = {
    "reasoning": [
        {
            "label": "human_1",
            "prompt": (
                "Answer the following multiple-choice question. Think step by step carefully and select the best answer."
            ),
        },
        {
            "label": "human_2",
            "prompt": (
                "Read the multiple-choice question and go through the options carefully before deciding. Pick the answer that seems most appropriate based on the question, and return only its label. Take a moment to think, but don’t overcomplicate it."
            ),
        },
        {
            "label": "automatic_1",
            "prompt": (
                "Read the multiple-choice question carefully, think through the options, and select the best answer. Return only the final answer."
            ),
        },
        {
            "label": "automatic_2",
            "prompt": (
                "Carefully read the multiple-choice question and all provided answer choices. Think through the problem before answering, compare the available options, and select the single best answer. Pay attention to important details, avoid unsupported assumptions, and choose the option that most directly answers the question. Return only the final selected answer, without showing your reasoning or adding extra commentary."
            ),
        },
    ],
    "non_reasoning": [
        {
            "label": "human_1",
            "prompt": (
                "Answer the following multiple-choice question. Select the best answer only."
            ),
        },
        {
            "label": "human_2",
            "prompt": (
                "You are given a multiple-choice question, and your task is to choose the best answer and output its label. Read the question and options carefully, but keep the decision straightforward and avoid overthinking. Return only the label of the corret answer."
            ),
        },
        {
            "label": "automatic_1",
            "prompt": (
                "Answer the following multiple-choice question by selecting the best option. Return only the final answer."
            ),
        },
        {
            "label": "automatic_2",
            "prompt": (
                "Read the multiple-choice question and all provided answer choices, then select the single best answer. Focus on the information given, choose the option that most accurately answers the question, and avoid adding assumptions beyond what is necessary. Do not provide explanations, reasoning, or additional commentary. Return only the final selected answer."
            ),
        },
    ],
}
