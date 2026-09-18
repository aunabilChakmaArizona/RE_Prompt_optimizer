"""Run direct three-way classification on the labeled ANLI test rounds."""

from qa_test_inference_common import run_qa_test_inference


ANLI_INITIAL_PROMPT = (
    "Determine whether the hypothesis is entailed by the premise, is neutral with "
    "respect to the premise, or contradicts the premise. Select the best label."
)
ANLI_ANSWER_INSTRUCTION = (
    "Do not think or provide any reasoning. Output only the option label exactly once "
    "between the tags <answer> and </answer>, for example: <answer>entailment</answer>. "
    "Do not output anything else."
)


def main() -> None:
    """Run ANLI with thinking disabled and exact-label accuracy scoring."""
    run_qa_test_inference(
        mode_name="non_reasoning",
        default_instruction=ANLI_INITIAL_PROMPT,
        answer_instruction=ANLI_ANSWER_INSTRUCTION,
        enable_thinking=False,
        default_max_new_tokens=16,
        default_dataset_path="data/processed/anli/test.jsonl",
    )


if __name__ == "__main__":
    main()
