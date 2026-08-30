"""Run non-reasoning multiple-choice QA test inference."""

from qa_test_inference_common import (
    NON_REASONING_ANSWER_INSTRUCTION,
    run_qa_test_inference,
)


DEFAULT_INSTRUCTION_PROMPT = (
    "Answer the following multiple-choice question. Select the best answer and respond directly without reasoning or explanation."
)


def main() -> None:
    """Run QA inference with model thinking disabled."""
    run_qa_test_inference(
        mode_name="non_reasoning",
        default_instruction=DEFAULT_INSTRUCTION_PROMPT,
        answer_instruction=NON_REASONING_ANSWER_INSTRUCTION,
        enable_thinking=False,
    )


if __name__ == "__main__":
    main()
