from typing import Optional, List, Tuple

from agent_feedback_samples import FeedbackSamples


class GraphNode:
    def __init__(
        self,
        inference_prompt: str,
        parent: Optional["GraphNode"] = None,
        feedback: str = "",
        raw_feedback_texts: Optional[List[str]] = None,
        feedback_prompt: str = "",
        mutation_prompt: str = "",
        example_generation_prompt: str = "",
        val_score: Optional[float] = None,
        test_score: Optional[float] = None,
    ):
        self.inference_prompt = inference_prompt

        # Graph structure
        self.parent = parent
        self.children: List["GraphNode"] = []

        # Prompt metadata
        self.feedback = feedback
        self.raw_feedback_texts = raw_feedback_texts if raw_feedback_texts is not None else []
        self.feedback_prompt = feedback_prompt
        self.mutation_prompt = mutation_prompt
        self.example_generation_prompt = example_generation_prompt

        # Scores
        self.val_score = val_score
        self.test_score = test_score

        # Each entry: (FeedbackSamples, child_node)
        self.data: List[Tuple[FeedbackSamples, Optional["GraphNode"]]] = []

    def add_child_from_feedback(
        self,
        feedback: FeedbackSamples,
        child: "GraphNode",
    ):
        self.data.append((feedback, child))
        self.children.append(child)

    def __repr__(self) -> str:
        return (
            f"GraphNode("
            f"val_score={self.val_score}, "
            f"test_score={self.test_score}, "
            f"children={len(self.children)}, "
            f"feedback_sets={len(self.data)})"
        )
