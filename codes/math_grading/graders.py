"""Run simple, OpenAI PRM800K, and Math-Verify answer grading."""

from __future__ import annotations

import importlib.metadata
import re
import signal
from decimal import Decimal, InvalidOperation
from typing import Any


OPENAI_PRM800K_REVISION = "7ecc794703b2877f63226f2477a49b34f9b25163"
MATH_VERIFY_PACKAGE = "math-verify"
MATH_VERIFY_VERSION = "0.9.0"
GRADING_TIMEOUT_SECONDS = 5


class OpenAIGraderTimeoutError(TimeoutError):
    """Indicate that the vendored OpenAI grader exceeded its time limit."""


def _raise_openai_timeout(_signum: int, _frame: Any) -> None:
    """Interrupt an OpenAI grader call that exceeds its time limit."""
    raise OpenAIGraderTimeoutError("OpenAI PRM800K grading timed out.")


def _outer_command_content(value: str, command: str) -> str | None:
    """Return the content of one balanced outer LaTeX command."""
    prefix = f"\\{command}{{"
    if not value.startswith(prefix):
        return None

    depth = 1
    for index in range(len(prefix), len(value)):
        character = value[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                if index == len(value) - 1:
                    return value[len(prefix) : index]
                return None
    return None


def strip_answer_wrappers(value: str | None) -> str | None:
    """Remove only outer math delimiters and an outer boxed command."""
    if value is None:
        return None

    answer = value.strip()
    changed = True
    while answer and changed:
        changed = False
        wrapper_pairs = (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$"))
        for opening, closing in wrapper_pairs:
            if answer.startswith(opening) and answer.endswith(closing):
                answer = answer[len(opening) : -len(closing)].strip()
                changed = True
                break
        if changed:
            continue

        boxed_content = _outer_command_content(answer, "boxed")
        if boxed_content is not None:
            answer = boxed_content.strip()
            changed = True

    return answer


def normalize_numeric_answer(value: str | None) -> str | None:
    """Canonicalize the nonnegative numeric answers used by AIME."""
    answer = strip_answer_wrappers(value)
    if answer is None:
        return None

    numeric_text = re.sub(r"[^0-9.]", "", answer)
    if not numeric_text or numeric_text == "." or numeric_text.count(".") > 1:
        return None

    try:
        number = Decimal(numeric_text)
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None

    normalized = format(number, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def normalize_symbolic_answer(value: str | None) -> str | None:
    """Apply conservative formatting normalization without changing mathematics."""
    answer = strip_answer_wrappers(value)
    if answer is None:
        return None

    answer = answer.replace("−", "-").replace("–", "-")
    answer = answer.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    answer = answer.replace("\\left", "").replace("\\right", "")
    answer = re.sub(r"\\(?:!|,|;|:|quad|qquad)", "", answer)
    answer = re.sub(r"\s+", "", answer)
    answer = re.sub(r"_(?:\{(\d+)\}|(\d+))", lambda match: "_" + (match.group(1) or match.group(2)), answer)

    text_match = re.fullmatch(r"\\text\{(.+)\}", answer, flags=re.IGNORECASE)
    if text_match:
        return text_match.group(1).strip().casefold()
    return answer or None


def simple_answer_match(
    prediction: str | None, gold_answer: str, task_type: str
) -> tuple[bool, str | None, str | None]:
    """Compare answers with dataset-appropriate conservative normalization."""
    if task_type == "math_short_answer":
        normalized_prediction = normalize_numeric_answer(prediction)
        normalized_gold = normalize_numeric_answer(gold_answer)
    else:
        normalized_prediction = normalize_symbolic_answer(prediction)
        normalized_gold = normalize_symbolic_answer(gold_answer)

    is_correct = (
        normalized_prediction is not None
        and normalized_gold is not None
        and normalized_prediction == normalized_gold
    )
    return is_correct, normalized_prediction, normalized_gold


def openai_answer_match(
    prediction: str | None, gold_answer: str
) -> tuple[bool, str | None]:
    """Grade an answer with the vendored OpenAI PRM800K grader."""
    if prediction is None:
        return False, None

    from math_grading.openai_prm800k.grader import grade_answer

    cleaned_prediction = strip_answer_wrappers(prediction)
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_openai_timeout)
    signal.setitimer(signal.ITIMER_REAL, GRADING_TIMEOUT_SECONDS)
    try:
        return bool(grade_answer(cleaned_prediction, gold_answer)), None
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _math_verify_input(value: str) -> str:
    """Place one isolated answer in a LaTeX environment for Math-Verify."""
    cleaned = strip_answer_wrappers(value) or ""
    return f"${cleaned}$"


def math_verify_answer_match(
    prediction: str | None, gold_answer: str
) -> tuple[bool, str | None]:
    """Grade an answer with the pinned Math-Verify package."""
    if prediction is None:
        return False, None

    from math_verify import LatexExtractionConfig, parse, verify

    try:
        extraction_config = [LatexExtractionConfig()]
        parsed_gold = parse(
            _math_verify_input(gold_answer),
            extraction_config=extraction_config,
            fallback_mode="first_match",
            extraction_mode="first_match",
            parsing_timeout=GRADING_TIMEOUT_SECONDS,
        )
        parsed_prediction = parse(
            _math_verify_input(prediction),
            extraction_config=extraction_config,
            fallback_mode="first_match",
            extraction_mode="first_match",
            parsing_timeout=GRADING_TIMEOUT_SECONDS,
        )
        if not parsed_gold:
            return False, "gold_parse_failed"
        if not parsed_prediction:
            return False, "prediction_parse_failed"
        return bool(
            verify(
                parsed_gold,
                parsed_prediction,
                strict=True,
                timeout_seconds=GRADING_TIMEOUT_SECONDS,
            )
        ), None
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def grade_math_answer(
    prediction: str | None, gold_answer: str, task_type: str
) -> dict[str, Any]:
    """Return independent results from all three configured graders."""
    simple_correct, normalized_prediction, normalized_gold = simple_answer_match(
        prediction, gold_answer, task_type
    )
    openai_correct, openai_error = openai_answer_match(prediction, gold_answer)
    math_verify_correct, math_verify_error = math_verify_answer_match(prediction, gold_answer)
    return {
        "normalized_prediction": normalized_prediction,
        "normalized_gold_answer": normalized_gold,
        "simple_correct": simple_correct,
        "openai_correct": openai_correct,
        "math_verify_correct": math_verify_correct,
        "openai_error": openai_error,
        "math_verify_error": math_verify_error,
    }


def validate_grading_dependencies() -> None:
    """Raise a clear error when either external grader dependency is missing."""
    missing = []
    for module_name in ("sympy", "pylatexenc", "math_verify"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        requirements_path = "codes/math_grading/requirements.txt"
        raise RuntimeError(
            f"Missing grading dependencies: {', '.join(missing)}. "
            f"Install them with: python -m pip install -r {requirements_path}"
        )

    installed_math_verify = importlib.metadata.version(MATH_VERIFY_PACKAGE)
    if installed_math_verify != MATH_VERIFY_VERSION:
        raise RuntimeError(
            f"Expected {MATH_VERIFY_PACKAGE}=={MATH_VERIFY_VERSION}, "
            f"but found {installed_math_verify}. Install the pinned grading requirements."
        )


def grader_metadata() -> dict[str, str]:
    """Describe the pinned graders used for the saved evaluation summary."""
    return {
        "simple": "dataset-specific conservative normalization",
        "openai_prm800k_revision": OPENAI_PRM800K_REVISION,
        "math_verify_version": importlib.metadata.version(MATH_VERIFY_PACKAGE),
    }
