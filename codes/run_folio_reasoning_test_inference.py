"""Run reasoning-only test inference on the labeled FOLIO local test split."""

from qa_test_inference_common import run_qa_test_inference


FOLIO_REASONING_INITIAL_PROMPT = (
    "You are given a set of premises and a conclusion. Think step by step carefully "
    "and determine whether the conclusion is true, false, or uncertain based only on "
    "the premises."
)
FOLIO_REASONING_ANSWER_INSTRUCTION = (
    "After you finish reasoning, output only the option label exactly once between "
    "the tags <answer> and </answer>, for example: <answer>B</answer>."
)


def main() -> None:
    """Run FOLIO inference with model thinking enabled."""
    run_qa_test_inference(
        mode_name="reasoning",
        default_instruction=FOLIO_REASONING_INITIAL_PROMPT,
        answer_instruction=FOLIO_REASONING_ANSWER_INSTRUCTION,
        enable_thinking=True,
        default_max_new_tokens=4096,
        default_dataset_path="data/processed/folio/test.jsonl",
    )


if __name__ == "__main__":
    main()
