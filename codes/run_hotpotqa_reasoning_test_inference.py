"""Run reasoning-only test inference on the labeled HotpotQA local test split."""

from qa_test_inference_common import run_qa_test_inference


HOTPOTQA_REASONING_INITIAL_PROMPT = (
    "You are given context passages and a question that may require combining "
    "information from multiple passages. Think step by step carefully and provide "
    "the best answer using the context."
)
HOTPOTQA_REASONING_ANSWER_INSTRUCTION = (
    "After you finish reasoning, output only the final answer between the tags "
    "<answer> and </answer>."
)


def main() -> None:
    """Run HotpotQA inference with model thinking enabled."""
    run_qa_test_inference(
        mode_name="reasoning",
        default_instruction=HOTPOTQA_REASONING_INITIAL_PROMPT,
        answer_instruction=HOTPOTQA_REASONING_ANSWER_INSTRUCTION,
        enable_thinking=True,
        default_max_new_tokens=4096,
        default_dataset_path="data/processed/hotpotqa/test.jsonl",
    )


if __name__ == "__main__":
    main()
