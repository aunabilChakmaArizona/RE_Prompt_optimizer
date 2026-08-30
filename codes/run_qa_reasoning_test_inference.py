"""Run reasoning-mode multiple-choice QA test inference."""

from qa_test_inference_common import (
    REASONING_ANSWER_INSTRUCTION,
    run_qa_test_inference,
)


DEFAULT_INSTRUCTION_PROMPT = (
    "Answer the following multiple-choice question. Think step by step carefully and select the best answer."
)


def main() -> None:
    """Run QA inference with model thinking enabled."""
    run_qa_test_inference(
        mode_name="reasoning",
        default_instruction=DEFAULT_INSTRUCTION_PROMPT,
        answer_instruction=REASONING_ANSWER_INSTRUCTION,
        enable_thinking=True,
    )


if __name__ == "__main__":
    main()
