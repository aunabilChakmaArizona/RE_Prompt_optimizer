"""Run direct three-way classification on the labeled ANLI test rounds."""

from anli_task_common import ANLI_ANSWER_INSTRUCTION, ANLI_INITIAL_PROMPT
from qa_test_inference_common import run_qa_test_inference


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
