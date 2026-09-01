"""Run non-reasoning multiple-choice QA test inference."""

from qa_test_inference_common import (
    NON_REASONING_ANSWER_INSTRUCTION,
    NON_REASONING_INITIAL_PROMPT,
    run_qa_test_inference,
)


def main() -> None:
    """Run QA inference with model thinking disabled."""
    run_qa_test_inference(
        mode_name="non_reasoning",
        default_instruction=NON_REASONING_INITIAL_PROMPT,
        answer_instruction=NON_REASONING_ANSWER_INSTRUCTION,
        enable_thinking=False,
    )


if __name__ == "__main__":
    main()
