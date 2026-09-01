"""Run reasoning-mode multiple-choice QA test inference."""

from qa_test_inference_common import (
    REASONING_ANSWER_INSTRUCTION,
    REASONING_INITIAL_PROMPT,
    run_qa_test_inference,
)


def main() -> None:
    """Run QA inference with model thinking enabled."""
    run_qa_test_inference(
        mode_name="reasoning",
        default_instruction=REASONING_INITIAL_PROMPT,
        answer_instruction=REASONING_ANSWER_INSTRUCTION,
        enable_thinking=True,
    )


if __name__ == "__main__":
    main()
