"""Native span and final textual edit accounting for prompt refinements."""

from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Any, Sequence


WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def sequence_edit_distance(source: Sequence[Any], target: Sequence[Any]) -> int:
    """Calculate Levenshtein distance with substitution cost one."""
    if len(source) < len(target):
        source, target = target, source
    previous = list(range(len(target) + 1))
    for source_index, source_item in enumerate(source, start=1):
        current = [source_index]
        for target_index, target_item in enumerate(target, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[target_index] + 1,
                    previous[target_index - 1] + int(source_item != target_item),
                )
            )
        previous = current
    return previous[-1]


def word_units(text: str) -> list[str]:
    """Tokenize words and punctuation for word-level alignment."""
    return [match.group(0) for match in WORD_PATTERN.finditer(text)]


def _word_matches(text: str) -> list[re.Match[str]]:
    """Return word and punctuation matches with character offsets."""
    return list(WORD_PATTERN.finditer(text))


def textual_diff_spans(source: str, target: str) -> list[dict[str, Any]]:
    """Recover contiguous final edits through word-level sequence alignment."""
    source_matches = _word_matches(source)
    target_matches = _word_matches(target)
    source_units = [match.group(0) for match in source_matches]
    target_units = [match.group(0) for match in target_matches]
    matcher = difflib.SequenceMatcher(a=source_units, b=target_units, autojunk=False)
    spans: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        source_start = source_matches[i1].start() if i1 < len(source_matches) else len(source)
        source_end = source_matches[i2 - 1].end() if i2 > i1 else source_start
        target_start = target_matches[j1].start() if j1 < len(target_matches) else len(target)
        target_end = target_matches[j2 - 1].end() if j2 > j1 else target_start
        spans.append(
            {
                "operation": tag,
                "source_word_start": i1,
                "source_word_end": i2,
                "target_word_start": j1,
                "target_word_end": j2,
                "source_char_start": source_start,
                "source_char_end": source_end,
                "target_char_start": target_start,
                "target_char_end": target_end,
                "source_text": source[source_start:source_end],
                "target_text": target[target_start:target_end],
            }
        )
    return spans


def changed_sequence_counts(
    source: Sequence[Any],
    target: Sequence[Any],
) -> tuple[int, int]:
    """Count source and target units participating in non-equal alignment blocks."""
    matcher = difflib.SequenceMatcher(a=list(source), b=list(target), autojunk=False)
    source_changed = 0
    target_changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        source_changed += i2 - i1
        target_changed += j2 - j1
    return source_changed, target_changed


def normalize_for_overlap(text: str) -> str:
    """Normalize text for conservative example-copy detection."""
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def is_example_specific(text: str, corpus_texts: Sequence[str]) -> bool:
    """Flag substantial spans copied from a pre-normalized question, choice, or fact."""
    normalized = normalize_for_overlap(text)
    if len(normalized) < 12 or len(normalized.split()) < 3:
        return False
    return any(normalized in item for item in corpus_texts)


def categorize_span(text: str, corpus_texts: Sequence[str]) -> str:
    """Assign one transparent content category to an edited text span."""
    normalized = normalize_for_overlap(text)
    words = set(normalized.split())
    if is_example_specific(text, corpus_texts):
        return "example_specific_or_copied"
    if words & {"answer", "tag", "tags", "label", "bracket", "brackets", "format"}:
        return "output_format"
    if words & {"question", "choice", "choices", "option", "options"}:
        return "question_or_choice_reference"
    if words & {"reason", "reasoning", "think", "step", "steps", "derive", "infer"}:
        return "reasoning_or_problem_solving"
    if words & {"best", "correct", "select", "selection", "eliminate", "compare"}:
        return "answer_selection_criterion"
    if words & {"science", "scientific", "knowledge", "fact", "facts", "physical"}:
        return "science_or_domain_guidance"
    if words & {"answer", "respond", "determine", "identify", "solve", "choose"}:
        return "task_or_action_instruction"
    if not any(character.isalnum() for character in text):
        return "punctuation_or_formatting"
    if len(words) <= 3:
        return "grammar_fluency_or_connector"
    return "other_instruction_fragment"


def tokenizer_ids(tokenizer, text: str) -> list[int]:
    """Tokenize prompt text without model-added special tokens."""
    return list(tokenizer.encode(text, add_special_tokens=False))


def build_edit_audit(
    source_prompt: str,
    final_prompt: str,
    tokenizer,
    *,
    native_selected_spans: Sequence[dict[str, Any]] | None = None,
    corpus_texts: Sequence[str] = (),
) -> dict[str, Any]:
    """Measure native selections and actual retained prompt edits in one schema."""
    source_token_ids = tokenizer_ids(tokenizer, source_prompt)
    target_token_ids = tokenizer_ids(tokenizer, final_prompt)
    source_words = word_units(source_prompt)
    target_words = word_units(final_prompt)
    final_spans = textual_diff_spans(source_prompt, final_prompt)
    for span in final_spans:
        content = " ".join(
            part for part in [span["source_text"], span["target_text"]] if part
        )
        span["content_category"] = categorize_span(content, corpus_texts)

    native_spans = [dict(span) for span in native_selected_spans or []]
    for span in native_spans:
        span_text = str(
            span.get("text")
            or span.get("region_text")
            or span.get("source_text")
            or ""
        )
        span.setdefault("content_category", categorize_span(span_text, corpus_texts))

    source_characters_changed, target_characters_changed = changed_sequence_counts(
        source_prompt,
        final_prompt,
    )
    source_words_changed, target_words_changed = changed_sequence_counts(
        source_words,
        target_words,
    )
    source_tokens_changed, target_tokens_changed = changed_sequence_counts(
        source_token_ids,
        target_token_ids,
    )

    return {
        "changed": source_prompt != final_prompt,
        "native_selected_span_count": len(native_spans),
        "native_selected_source_token_count": sum(
            int(
                span.get("token_count")
                or len(span.get("token_indices", []))
                or len(span.get("region_token_indices", []))
            )
            for span in native_spans
        ),
        "native_selected_spans": native_spans,
        "actual_edit_span_count": len(final_spans),
        "actual_edit_spans": final_spans,
        "source_character_count": len(source_prompt),
        "target_character_count": len(final_prompt),
        "character_edit_distance": sequence_edit_distance(source_prompt, final_prompt),
        "source_characters_deleted_or_replaced": source_characters_changed,
        "target_characters_inserted_or_replaced": target_characters_changed,
        "source_word_count": len(source_words),
        "target_word_count": len(target_words),
        "word_edit_distance": sequence_edit_distance(source_words, target_words),
        "source_words_deleted_or_replaced": source_words_changed,
        "target_words_inserted_or_replaced": target_words_changed,
        "source_token_count": len(source_token_ids),
        "target_token_count": len(target_token_ids),
        "token_edit_distance": sequence_edit_distance(source_token_ids, target_token_ids),
        "source_tokens_deleted_or_replaced": source_tokens_changed,
        "target_tokens_inserted_or_replaced": target_tokens_changed,
        "content_category_counts": dict(
            Counter(span["content_category"] for span in final_spans)
        ),
    }
