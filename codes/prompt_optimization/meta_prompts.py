"""QA-specific meta-prompts and robust extraction helpers for all optimizers."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from prompt_optimization.qa_task import QAMode


PROMPT_PATTERN = re.compile(r"<prompt\s*>(.*?)</prompt\s*>", re.IGNORECASE | re.DOTALL)

QA_TASK_DESCRIPTIONS = {
    "reasoning": """A multiple-choice question contains several labeled options, with one best answer.
The task requires reasoning carefully and then outputting the correct option label.""",
    "non_reasoning": """A multiple-choice question contains several labeled options, with one best answer.
The task requires directly outputting the correct option label without reasoning or explanation.""",
}


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
    mode: QAMode,
    example: str,
) -> str:
    """Ask the optimizer for feedback about one correct or incorrect QA response."""
    task_description = QA_TASK_DESCRIPTIONS[mode.name]
    if mode.name == "reasoning":
        analysis_instruction = """Analyze the reasoning in the LLM response and explain how it led to the selected answer.
- If the answer is correct, explain which reasoning steps, evidence, or cues were useful.
- If the answer is incorrect, explain which reasoning step, misunderstanding, missing evidence, or heuristic likely caused the error."""
    else:
        analysis_instruction = """The LLM was asked to answer directly, so its response may not contain explicit reasoning. Infer the most likely evidence, cues, or heuristic that led to the selected answer.
- If the answer is correct, explain what likely supported the decision.
- If the answer is incorrect, explain what misunderstanding, missing evidence, or heuristic likely caused the error."""

    return f"""You are an expert feedback model for a multiple-choice question-answering task. You specialize in explaining why a question-answering system arrived at a particular answer, for both correct and incorrect predictions.

{task_description}

You are given one task instance containing the question, choices, ground-truth answer, the LLM's response, and whether its selected answer was correct or incorrect.

The ground-truth answer and outcome are provided as contextual information. Your task is to explain the most likely reasoning or decision process that led to the LLM's answer, not to solve the question again.

{analysis_instruction}

Instance:
```
{example}
```

Please reason through the problem, but provide your final feedback only inside <feedback> and </feedback>."""


def rpo_rewrite_prompt(
    instruction_prompt: str,
    feedback_examples: Sequence[str],
    mode: QAMode,
) -> str:
    """Ask RPO to revise one QA instruction from separate example feedback."""
    task_description = QA_TASK_DESCRIPTIONS[mode.name]

    return f"""You are an expert prompt generator for a multiple-choice question-answering task. You specialize in revising and improving prompts based on feedback from previous model predictions.

{task_description}

You are given below a prompt that is used by another LLM to answer the questions:
```
{instruction_prompt}
```

Using this prompt, another LLM was tested on {len(feedback_examples)} task instances. Below, you are given the inputs, responses, and feedback for each task.
```
{chr(10).join(feedback_examples)}
```

Carefully read the inputs, outputs, and feedback to identify problems with the current prompt.
Your task is to generate a revised version of the prompt that helps the other LLM generalize better when using it.
You may modify, add to, or remove any instructions or content in the current prompt to improve prediction and generalization.

Please reason through the problem, but output only the revised prompt inside <prompt> and </prompt>."""


def evoprompt_seed_prompt(initial_prompt: str, mode: QAMode, count: int) -> str:
    """Request diverse AI seeds for EvoPrompt's initial population."""
    return f"""Create {count} diverse general instructions for solving multiple-choice questions.

Basic instruction:
<prompt>{initial_prompt}</prompt>

The answer format is fixed separately as:
{mode.answer_instruction}

Each instruction must be self-contained, concise, task-generic, and must not contain question-specific text or mention any dataset or benchmark. Return each instruction in its own <prompt>...</prompt> block."""


def evoprompt_de_prompt(
    basic_prompt: str,
    target_prompt: str,
    donor_a: str,
    donor_b: str,
    donor_c: str,
    mode: QAMode,
) -> str:
    """Construct one differential-evolution crossover meta-prompt."""
    return f"""Generate a better instruction for solving multiple-choice questions through differential evolution.

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

Return one concise, task-generic instruction with no question-specific text or dataset or benchmark names inside <prompt> and </prompt>."""


def etgpo_taxonomy_prompt(
    current_taxonomy: Sequence[dict[str, Any]],
    error_examples: Sequence[str],
    min_categories: int,
    max_categories: int,
) -> str:
    """Ask ETGPO to update a compact taxonomy of instruction-level errors."""
    return f"""Analyze mistakes from multiple-choice question answering and update a compact taxonomy of instruction-level failure types.

Current taxonomy:
{json.dumps(list(current_taxonomy), ensure_ascii=False)}

New mistakes:
{chr(10).join(error_examples)}

Return JSON with key "categories" containing between {min_categories} and {max_categories} categories. Each category must have "name", "description", and "count". Return the complete updated taxonomy and merge equivalent categories. Focus on general reasoning or answer-selection behavior, not individual topics or questions. Do not mention any dataset or benchmark."""


def etgpo_guidance_prompt(
    instruction_prompt: str,
    mode: QAMode,
    taxonomy: Sequence[dict[str, Any]],
    candidate_count: int,
) -> str:
    """Ask ETGPO to turn its error taxonomy into candidate instructions."""
    return f"""Use this error taxonomy to improve an instruction for solving multiple-choice questions.

Current instruction:
<prompt>{instruction_prompt}</prompt>

Error taxonomy:
{json.dumps(list(taxonomy), ensure_ascii=False, indent=2)}

The answer format is fixed separately:
{mode.answer_instruction}

Generate {candidate_count} concise, task-generic candidate instructions. Do not include question-specific text or mention any dataset or benchmark. Return each in a separate <prompt>...</prompt> block."""


def lpo_location_prompt(
    instruction_prompt: str,
    mode: QAMode,
    mistakes: Sequence[str],
    max_locations: int,
    max_words_per_location: int,
) -> str:
    """Ask LPO to tag a few local instruction spans worth rewriting."""
    return f"""Find local spans in an editable instruction for solving multiple-choice questions that should be improved.

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
    return f"""Locally improve the marked spans in this instruction for solving multiple-choice questions.

Marked instruction:
{marked_prompt}

Selected locations:
{json.dumps(list(locations), ensure_ascii=False)}

Preserve all unmarked wording as closely as possible. Keep the result task-generic, with no question-specific text or dataset or benchmark names. The fixed answer instruction is separate:
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
    return f"""Suggest local replacements for gradient-selected spans in an instruction for solving multiple-choice questions.

Marked instruction:
{marked_prompt}

Spans:
{chr(10).join(region_lines)}

For each span, give {candidate_count} concise replacement phrases that fit its context and preserve a task-generic instruction. Do not copy text from any provided question or mention any dataset or benchmark. Return JSON only in this form:
{{"span_1": {{"candidates": ["..."]}}, "span_2": {{"candidates": ["..."]}}}}"""
