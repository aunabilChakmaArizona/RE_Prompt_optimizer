"""Run direct-answer test inference on the official RACE high-school split."""

from qa_test_inference_common import (
    NON_REASONING_ANSWER_INSTRUCTION,
    run_qa_test_inference,
)


RACE_HIGH_INITIAL_PROMPT = (
    "Read the passage carefully and answer the multiple-choice question by "
    "selecting the best answer."
)


def main() -> None:
    """Run RACE-H inference with thinking disabled and exact-label scoring."""
    run_qa_test_inference(
        mode_name="non_reasoning",
        default_instruction=RACE_HIGH_INITIAL_PROMPT,
        answer_instruction=NON_REASONING_ANSWER_INSTRUCTION,
        enable_thinking=False,
        default_max_new_tokens=16,
        default_dataset_path="data/processed/race_high/test.jsonl",
    )


if __name__ == "__main__":
    main()
