"""QA-specific meta-prompts and robust extraction helpers for all optimizers."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from agents.agent_prompts import (
    GRADIENT_REGION_CANDIDATE_SUGGESTION_PROMPT_V1,
    LPO_LOCAL_REWRITE_BODY_V1,
    LPO_LOCATION_TAGGING_BODY_V1,
)
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


EVOPROMPT_DE_EXAMPLES = {
    "reasoning": """Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between Prompt 1 and Prompt 2:
Prompt 1: Read the question and answer choices carefully, reason step by step, and choose the best-supported answer.
Prompt 2: Analyze what the question asks, recall the relevant facts, compare every option, and verify the final choice.
2. Randomly mutate the different parts.
3. Combine the different parts with Prompt 3, selectively replace it with the different parts from step 2, and generate a new prompt.
Prompt 3: Solve the question systematically, eliminate choices that conflict with the evidence, and check that the remaining answer fully addresses the question.
4. Crossover the prompt in step 3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: Answer the following multiple-choice question. Think step by step carefully and select the best answer.

1. Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: Read the question and answer choices carefully, reason step by step, and choose the best-supported answer.
Prompt 2: Analyze what the question asks, recall the relevant facts, compare every option, and verify the final choice.
Different parts:
"read the question and answer choices carefully" vs "analyze what the question asks"
"reason step by step" vs "recall the relevant facts"
"choose the best-supported answer" vs "compare every option and verify the final choice"

2. Randomly mutate the different parts:
"read the question and answer choices carefully" -> "identify the question's exact requirement"
"reason step by step" -> "build a logically connected solution"
"recall the relevant facts" -> "use only facts relevant to the decision"
"compare every option" -> "test each choice against the reasoning"

3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt:
Prompt 3: Solve the question systematically, eliminate choices that conflict with the evidence, and check that the remaining answer fully addresses the question.
New Prompt: Identify the question's exact requirement, build a logically connected solution using relevant facts, test each choice against the reasoning, and eliminate choices that conflict with the evidence.

4. Crossover the prompt in step 3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: Answer the following multiple-choice question. Think step by step carefully and select the best answer.
Final Prompt: <prompt>Identify exactly what the question asks, reason through the relevant facts step by step, test the answer choices against that reasoning, eliminate inconsistent choices, and select the best-supported answer.</prompt>""",
    "non_reasoning": """Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between Prompt 1 and Prompt 2:
Prompt 1: Read the question and choices carefully and select the single best answer directly.
Prompt 2: Identify what the question asks, compare the available choices, and return the most appropriate answer without explanation.
2. Randomly mutate the different parts.
3. Combine the different parts with Prompt 3, selectively replace it with the different parts from step 2, and generate a new prompt.
Prompt 3: Use relevant knowledge to evaluate the choices and choose the option best supported by the question.
4. Crossover the prompt in step 3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: Answer the following multiple-choice question. Select the best answer directly without reasoning or explanation.

1. Identifying the different parts between Prompt 1 and Prompt 2:
Prompt 1: Read the question and choices carefully and select the single best answer directly.
Prompt 2: Identify what the question asks, compare the available choices, and return the most appropriate answer without explanation.
Different parts:
"read the question and choices carefully" vs "identify what the question asks"
"select the single best answer" vs "compare the available choices"
"directly" vs "without explanation"

2. Randomly mutate the different parts:
"read the question and choices carefully" -> "focus on the question's exact requirement"
"select the single best answer" -> "choose the best-supported option"
"compare the available choices" -> "evaluate each choice for relevance and correctness"

3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2 and generate a new prompt:
Prompt 3: Use relevant knowledge to evaluate the choices and choose the option best supported by the question.
New Prompt: Focus on the question's exact requirement, use relevant knowledge to evaluate each choice for relevance and correctness, and choose the best-supported option.

4. Crossover the prompt in step 3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: Answer the following multiple-choice question. Select the best answer directly without reasoning or explanation.
Final Prompt: <prompt>Focus on exactly what the question asks, use relevant knowledge to evaluate the available choices, and select the single best-supported answer directly without explanation.</prompt>""",
}


def evoprompt_de_prompt(
    target_prompt: str,
    donor_a: str,
    donor_b: str,
    donor_c: str,
    mode: QAMode,
) -> str:
    """Construct the worked-example DE prompt used to mutate one population member."""
    return f"""{EVOPROMPT_DE_EXAMPLES[mode.name]}

Please follow the instruction step-by-step to generate a better prompt.
1. Identify the different parts between Prompt 1 and Prompt 2:
Prompt 1: {donor_a}
Prompt 2: {donor_b}
2. Randomly mutate the different parts.
3. Combine the different parts with Prompt 3, selectively replace it with the different parts in step 2, and generate a new prompt.
Prompt 3: {donor_c}
4. Crossover the prompt in step 3 with the following basic prompt and generate a final prompt bracketed with <prompt> and </prompt>:
Basic Prompt: {target_prompt}

1. """


def etgpo_first_taxonomy_prompt(
    error_examples: Sequence[str],
    mode: QAMode,
) -> str:
    """Ask ETGPO to create issue categories from its first batch of QA failures."""
    if mode.name == "reasoning":
        analysis_steps = """1. Find the EARLIEST point in the response's reasoning where it went wrong.
2. Explain what specifically went wrong.
3. Explain why that error led to the wrong selected answer."""
        reasoning_source_note = ""
    else:
        analysis_steps = """1. Use the post-hoc feedback to identify the earliest likely decision error, missing evidence, or misleading cue.
2. Explain what specifically went wrong in selecting the answer.
3. Explain why that error led to the wrong selected answer."""
        reasoning_source_note = """Important: the reasoning field is post-hoc feedback describing the most likely cause of the incorrect direct-answer prediction, not a verbatim chain of thought from the target model. Use it as probabilistic evidence together with the question, choices, correct answer, and wrong answer."""
    schema = {
        "categories": [
            {
                "category_name": "Short descriptive name",
                "summary": "One-sentence error pattern",
                "description": "Detailed self-contained description",
                "example": (
                    "A brief topic-independent illustration based on the observed "
                    "error pattern"
                ),
                "error_type": "Type of error",
                "why_leads_to_wrong_answer": "How this error causes wrong answers",
            }
        ],
        "failure_assignments": [
            {
                "failure_id": 1,
                "problem_id": "ID shown in the failure",
                "category_name": "Assigned category name",
                "trace_details": {
                    "trace_specific_location": "Where the error occurred",
                    "trace_specific_details": "What specifically went wrong",
                },
            }
        ],
    }
    return f"""You are an expert at analyzing why language models fail on multiple-choice question answering.

{chr(10).join(error_examples)}

## Your Task

Analyze every failure and identify its root cause. Be as descriptive as possible.

{reasoning_source_note}

For each failure:
{analysis_steps}

Create issue categories that capture each type of error. Categories should be general enough to potentially apply to other traces, but specific enough to be meaningful.

Identify reusable reasoning or decision errors rather than question-specific topics. Do not create categories based only on particular entities, answer choices, scientific terms, or isolated facts. Group failures that share the same underlying error even when their question topics differ.

IMPORTANT: Each category must be SELF-CONTAINED and understandable by someone who has NOT seen the original problems.

## Output Format

Return only one JSON object matching this structure:
```json
{json.dumps(schema, ensure_ascii=False, indent=2)}
```"""


def etgpo_non_reasoning_feedback_prompt(error_example: str) -> str:
    """Ask for the most likely cause of one incorrect direct-answer prediction."""
    return f"""You are an expert feedback model for a multiple-choice question-answering task. Specifically, you are skilled at providing feedback explaining why a question-answering system arrived at an incorrect prediction.

{QA_TASK_DESCRIPTIONS["non_reasoning"]}

You are given one task instance containing the question, choices, ground-truth answer, the LLM's response, and its selected answer.

The ground-truth answer is provided only as contextual information. The feedback model's task is to explain the most likely decision process that led to the incorrect answer, not merely to state that the prediction was wrong. Because the LLM was asked to answer directly, its response may not contain explicit reasoning. Infer the misunderstanding, missing evidence, misleading cue, or heuristic that most likely caused the error.

Instance:
```
{error_example}
```

Please reason through the problem, but provide your final feedback only inside <feedback> and </feedback>."""


def etgpo_update_taxonomy_prompt(
    existing_categories: Sequence[dict[str, Any]],
    error_examples: Sequence[str],
    mode: QAMode,
) -> str:
    """Ask ETGPO to assign another failure batch and add only genuinely new issues."""
    category_lines = []
    for category in existing_categories:
        category_lines.append(
            "\n".join(
                [
                    f"### Category: {category['category_name']}",
                    f"- Summary: {category.get('summary', '')}",
                    f"- Description: {category.get('description', '')}",
                    f"- Example: {category.get('example', '')}",
                    f"- Error Type: {category.get('error_type', '')}",
                    "- Why it leads to wrong answer: "
                    f"{category.get('why_leads_to_wrong_answer', '')}",
                    f"- Failures assigned so far: {category.get('trace_count', 0)}",
                ]
            )
        )
    trace_location = (
        "Earliest reasoning location" if mode.name == "reasoning" else "Decision error or misleading cue"
    )
    reasoning_source_note = (
        ""
        if mode.name == "reasoning"
        else """Important: the reasoning field is post-hoc feedback describing the most likely cause of the incorrect direct-answer prediction, not a verbatim chain of thought from the target model. Use it as probabilistic evidence together with the question, choices, correct answer, and wrong answer."""
    )
    schema = {
        "new_categories": [
            {
                "category_name": "Short descriptive name for a new error",
                "summary": "One-sentence error pattern",
                "description": "Detailed self-contained description",
                "example": (
                    "A brief topic-independent illustration based on the observed "
                    "error pattern"
                ),
                "error_type": "Type of error",
                "why_leads_to_wrong_answer": "How this error causes wrong answers",
            }
        ],
        "failure_assignments": [
            {
                "failure_id": 1,
                "problem_id": "ID shown in the failure",
                "is_new_category": False,
                "category_name": "Existing or new category name",
                "trace_details": {
                    "trace_specific_location": trace_location,
                    "trace_specific_details": "What specifically went wrong",
                },
            }
        ],
    }
    return f"""You are an expert at analyzing why language models fail on multiple-choice question answering.

## Existing Issue Categories

{chr(10).join(category_lines)}

## New Failures

{chr(10).join(error_examples)}

## Your Task

For every new failure, decide whether its root cause fits an existing category. Reuse that category whenever it fits. Create a new category only when the error is fundamentally different. New categories must be self-contained, generalizable, specific, and actionable.

{reasoning_source_note}

Identify reusable reasoning or decision errors rather than question-specific topics. Do not create categories based only on particular entities, answer choices, scientific terms, or isolated facts. Group failures that share the same underlying error even when their question topics differ.

## Output Format

Return only one JSON object matching this structure:
```json
{json.dumps(schema, ensure_ascii=False, indent=2)}
```

The "new_categories" list must contain only categories that do not already exist."""


def etgpo_guidance_prompt(
    instruction_prompt: str,
    mode: QAMode,
    taxonomy: Sequence[dict[str, Any]],
    total_failures: int,
) -> str:
    """Ask ETGPO to turn selected frequent error categories into one improved prompt."""
    category_sections = []
    for index, category in enumerate(taxonomy, start=1):
        failure_count = int(category.get("trace_count", 0))
        problem_count = len(category.get("problem_ids", []))
        coverage = 100.0 * failure_count / total_failures if total_failures else 0.0
        category_sections.append(
            f"""## Category {index}: {category['category_name']}

**Statistics:** {failure_count} failures ({coverage:.1f}%), {problem_count} problems

**Summary:** {category.get('summary', '')}

**Description:** {category.get('description', '')}

**Example:** {category.get('example', '')}

**Error Type:** {category.get('error_type', '')}

**Why it leads to wrong answer:** {category.get('why_leads_to_wrong_answer', '')}

---"""
        )
    schema = {
        "guidance_items": [
            {
                "category_name": "Name of the category",
                "guidance_text": "The full guidance text for this category",
            }
        ],
        "preamble": "1-2 sentence introduction",
        "full_prompt": "Complete enhanced prompt starting with base instruction",
    }
    return f"""You are an expert at improving language model performance on multiple-choice question answering.

I have identified the following error categories from model failures. Generate guidance to help avoid these errors.

{chr(10).join(category_sections)}

## Your Task

Generate guidance text that:
1. Addresses each failure category with specific, actionable advice
2. Is written as instructions TO the model
3. Expresses advice as a reusable reasoning or decision strategy
4. Is prioritized by frequency

Generate SHORT, CONCISE guidance. Each item should be 1-2 sentences.

## Critical Constraints

- The goal is ACCURACY, not caution. Never generate guidance that encourages the model to refuse, abstain, or say "not specified" when an answer can be reasonably provided.
- Keep the guidance task-general. Do not copy question-specific entities, answer choices, scientific terms, or isolated facts from the categories.
- Do not invent new WRONG/CORRECT question-answer examples.
- Preserve this task behavior: {QA_TASK_DESCRIPTIONS[mode.name]}

## Output Format

Return a JSON object with:

```json
{json.dumps(schema, ensure_ascii=False, indent=2)}
```

The "full_prompt" should start with:
"{instruction_prompt}"

Then add the preamble and guidance items."""


def lpo_location_prompt(
    instruction_prompt: str,
    mode: QAMode,
    feedback_examples: Sequence[str],
    max_locations: int,
    max_words_per_location: int,
) -> str:
    """Adapt the shared LPO location-tagging body to one QA mode."""
    qa_prompt = "\n\n".join(
        [QA_TASK_DESCRIPTIONS[mode.name], LPO_LOCATION_TAGGING_BODY_V1]
    )
    return (
        qa_prompt
        .replace("#INFERENCE_PROMPT#", instruction_prompt)
        .replace("#FEEDBACK_EXAMPLES#", "\n\n".join(feedback_examples))
        .replace("#MAX_EDIT_TAGS#", str(max_locations))
        .replace("#MAX_WORDS_PER_EDIT_TAG#", str(max_words_per_location))
    )


def lpo_rewrite_prompt(
    marked_prompt: str,
    feedback_examples: Sequence[str],
    mode: QAMode,
) -> str:
    """Adapt the shared LPO local-rewrite body to one QA mode."""
    qa_prompt = "\n\n".join(
        [QA_TASK_DESCRIPTIONS[mode.name], LPO_LOCAL_REWRITE_BODY_V1]
    )
    return (
        qa_prompt
        .replace("#TAGGED_PROMPT#", marked_prompt)
        .replace("#FEEDBACK_EXAMPLES#", "\n\n".join(feedback_examples))
    )


def gradpo_candidate_prompt(
    marked_prompt: str,
    selected_regions: Sequence[dict[str, Any]],
    candidate_count: int,
) -> str:
    """Ask GradPO-Gen for short replacements at selected gradient spans."""
    region_blocks = []
    for region in selected_regions:
        region_blocks.append(
            "\n".join(
                [
                    f"Span {region['region_rank']}",
                    f"Text: ```{region['region_text']}```",
                ]
            )
        )
    return (
        GRADIENT_REGION_CANDIDATE_SUGGESTION_PROMPT_V1
        .replace("#MARKED_PROMPT#", marked_prompt)
        .replace("#REGION_CANDIDATE_REQUEST_BLOCKS#", "\n\n".join(region_blocks))
        .replace("#NUM_CANDIDATES#", str(candidate_count))
        .replace("#NUM_REGIONS#", str(len(selected_regions)))
    )
