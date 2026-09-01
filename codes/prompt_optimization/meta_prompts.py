"""QA-specific meta-prompts and robust extraction helpers for all optimizers."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from prompt_optimization.qa_task import QAMode


PROMPT_PATTERN = re.compile(r"<prompt\s*>(.*?)</prompt\s*>", re.IGNORECASE | re.DOTALL)


def unique_nonempty(values: Sequence[str]) -> list[str]:
    """Keep distinct non-empty strings in their original order."""
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def extract_tagged_prompts(outputs: Sequence[str]) -> list[str]:
    """Extract every complete prompt tag from optimizer model outputs."""
    return unique_nonempty(
        [match for output in outputs for match in PROMPT_PATTERN.findall(output)]
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first plausible JSON object from a model response."""
    candidates = [text.strip()]
    candidates.extend(
        match.strip()
        for match in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    )
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def rpo_feedback_prompt(
    instruction_prompt: str,
    mode: QAMode,
    examples: Sequence[str],
) -> str:
    """Ask the optimizer to diagnose a small mixed set of QA predictions."""
    return f"""You are analyzing an instruction for OpenBookQA multiple-choice questions.

Only the first instruction is editable. The answer-format instruction, question, and choices are fixed.

Current editable instruction:
<prompt>{instruction_prompt}</prompt>

Fixed answer instruction:
{mode.answer_instruction}

Prediction examples:
{chr(10).join(examples)}

Briefly identify instruction-level weaknesses that explain the mistakes. Do not copy a question, answer choice, or dataset fact into the instruction. Return the diagnosis inside <feedback> and </feedback>."""


def rpo_rewrite_prompt(
    instruction_prompt: str,
    feedback: str,
    mode: QAMode,
) -> str:
    """Ask RPO to produce one revised editable QA instruction."""
    return f"""Improve the editable instruction using the feedback below.

Current instruction:
<prompt>{instruction_prompt}</prompt>

Feedback:
{feedback}

The fixed answer instruction is not part of the editable text:
{mode.answer_instruction}

Write one concise, general OpenBookQA instruction. Do not include specific questions, facts, or answer choices. Return only the revised instruction inside <prompt> and </prompt>."""


def evoprompt_seed_prompt(initial_prompt: str, mode: QAMode, count: int) -> str:
    """Request diverse AI seeds for EvoPrompt's initial population."""
    return f"""Create {count} diverse general instructions for OpenBookQA multiple-choice questions.

Basic instruction:
<prompt>{initial_prompt}</prompt>

The answer format is fixed separately as:
{mode.answer_instruction}

Each instruction must be self-contained, concise, and must not contain example-specific facts. Return each instruction in its own <prompt>...</prompt> block."""


def evoprompt_de_prompt(
    basic_prompt: str,
    target_prompt: str,
    donor_a: str,
    donor_b: str,
    donor_c: str,
    mode: QAMode,
) -> str:
    """Construct one differential-evolution crossover meta-prompt."""
    return f"""Generate a better OpenBookQA instruction through differential evolution.

1. Identify useful differences between Prompt 1 and Prompt 2.
2. Selectively combine those differences with Prompt 3.
3. Crossover the result with the target and basic prompts.

Basic Prompt: {basic_prompt}
Target Prompt: {target_prompt}
Prompt 1: {donor_a}
Prompt 2: {donor_b}
Prompt 3: {donor_c}

The following answer-format text is fixed and must not be copied into the instruction:
{mode.answer_instruction}

Return one concise general instruction, with no example-specific facts, inside <prompt> and </prompt>."""


def etgpo_taxonomy_prompt(
    current_taxonomy: Sequence[dict[str, Any]],
    error_examples: Sequence[str],
    min_categories: int,
    max_categories: int,
) -> str:
    """Ask ETGPO to update a compact taxonomy of instruction-level errors."""
    return f"""Analyze OpenBookQA mistakes and update a compact taxonomy of instruction-level failure types.

Current taxonomy:
{json.dumps(list(current_taxonomy), ensure_ascii=False)}

New mistakes:
{chr(10).join(error_examples)}

Return JSON with key "categories" containing between {min_categories} and {max_categories} categories. Each category must have "name", "description", and "count". Return the complete updated taxonomy and merge equivalent categories. Focus on general reasoning or answer-selection behavior, not individual science topics, questions, or facts."""


def etgpo_guidance_prompt(
    instruction_prompt: str,
    mode: QAMode,
    taxonomy: Sequence[dict[str, Any]],
    candidate_count: int,
) -> str:
    """Ask ETGPO to turn its error taxonomy into candidate instructions."""
    return f"""Use this error taxonomy to improve an OpenBookQA instruction.

Current instruction:
<prompt>{instruction_prompt}</prompt>

Error taxonomy:
{json.dumps(list(taxonomy), ensure_ascii=False, indent=2)}

The answer format is fixed separately:
{mode.answer_instruction}

Generate {candidate_count} concise, general candidate instructions. Do not include example-specific facts. Return each in a separate <prompt>...</prompt> block."""


def lpo_location_prompt(
    instruction_prompt: str,
    mode: QAMode,
    mistakes: Sequence[str],
    max_locations: int,
    max_words_per_location: int,
) -> str:
    """Ask LPO to tag a few local instruction spans worth rewriting."""
    return f"""Find local spans in an editable OpenBookQA instruction that should be improved.

Instruction:
<prompt>{instruction_prompt}</prompt>

Fixed answer instruction:
{mode.answer_instruction}

Representative mistakes:
{chr(10).join(mistakes)}

Select at most {max_locations} exact, non-overlapping substrings from the editable instruction. Each substring may contain at most {max_words_per_location} words. Return JSON only:
{{"locations": [{{"text": "exact substring", "reason": "short reason"}}]}}
Do not select text from the examples or the fixed answer instruction."""


def lpo_rewrite_prompt(
    marked_prompt: str,
    locations: Sequence[dict[str, Any]],
    mode: QAMode,
) -> str:
    """Ask LPO to rewrite only marked local instruction spans."""
    return f"""Locally improve the marked spans in this OpenBookQA instruction.

Marked instruction:
{marked_prompt}

Selected locations:
{json.dumps(list(locations), ensure_ascii=False)}

Preserve all unmarked wording as closely as possible. Do not add example-specific facts. The fixed answer instruction is separate:
{mode.answer_instruction}

Return the complete revised editable instruction inside <prompt> and </prompt>, without edit tags."""


def gradpo_candidate_prompt(
    marked_prompt: str,
    selected_regions: Sequence[dict[str, Any]],
    candidate_count: int,
) -> str:
    """Ask GradPO-Gen for short replacements at selected gradient spans."""
    region_lines = []
    for region in selected_regions:
        region_lines.append(
            f"span_{region['region_rank']}: {region['region_text']!r} "
            f"({region['token_count']} target-model tokens)"
        )
    return f"""Suggest local replacements for gradient-selected spans in an OpenBookQA instruction.

Marked instruction:
{marked_prompt}

Spans:
{chr(10).join(region_lines)}

For each span, give {candidate_count} concise replacement phrases that fit its context and preserve a general instruction. Do not copy any dataset example. Return JSON only in this form:
{{"span_1": {{"candidates": ["..."]}}, "span_2": {{"candidates": ["..."]}}}}"""
