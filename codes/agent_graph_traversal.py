import math
import random
import time
from typing import Callable, List, Optional, Sequence

from agent_feedback_samples import FeedbackSamples
from agent_graph_node import GraphNode
from agent_select_feedback import select_feedback_samples


SampleFeedbackFn = Callable[[int], FeedbackSamples]
RunInferenceFn = Callable[[GraphNode, FeedbackSamples], FeedbackSamples]
GenerateFeedbackFn = Callable[[GraphNode, FeedbackSamples], str]
MutatePromptFn = Callable[[GraphNode, FeedbackSamples, str], str]
EvaluateFn = Callable[[GraphNode, str], float]
IterationHook = Callable[[int, GraphNode, GraphNode], None]


class EvolutionarySearch:
    def __init__(
        self,
        root: GraphNode,
        max_iterations: int = 20,
        feedback_sample_size: int = 3,
        temperature: float = 1.0,
        feedback_prompt: str = "",
        mutation_prompt: str = "",
        example_generation_prompt: str = "",
        rng: Optional[random.Random] = None,
    ):
        self.root = root
        self.population: List[GraphNode] = [root]
        self.max_iterations = max_iterations
        self.feedback_sample_size = feedback_sample_size
        self.temperature = temperature
        self.feedback_prompt = feedback_prompt
        self.mutation_prompt = mutation_prompt
        self.example_generation_prompt = example_generation_prompt
        self.rng = rng or random.Random()

    def _score_or_neg_inf(self, node: GraphNode) -> float:
        if node.val_score is None:
            return float("-inf")
        return float(node.val_score)

    def _softmax_weights(self, scores: Sequence[Optional[float]]) -> List[float]:
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        finite_scores = [s for s in scores if s is not None]
        if not finite_scores:
            return [1.0 for _ in scores]
        max_score = max(finite_scores)
        weights = []
        for s in scores:
            if s is None:
                weights.append(0.0)
            else:
                weights.append(math.exp((s - max_score) / self.temperature))
        if sum(weights) == 0.0:
            return [1.0 for _ in scores]
        return weights

    def sample_parent(self) -> GraphNode:
        scores = [node.val_score for node in self.population]
        weights = self._softmax_weights(scores)
        return self.rng.choices(self.population, weights=weights, k=1)[0]

    def best_node(self) -> GraphNode:
        return max(self.population, key=self._score_or_neg_inf)

    def run(
        self,
        sample_feedback_fn: SampleFeedbackFn,
        run_inference_fn: RunInferenceFn,
        generate_feedback_fn: GenerateFeedbackFn,
        mutate_prompt_fn: MutatePromptFn,
        evaluate_fn: EvaluateFn,
        selection_mode: str = "mixed",
        on_iteration_end: Optional[IterationHook] = None,
    ) -> tuple[GraphNode, List[GraphNode]]:
        start_time = time.monotonic()
        print("[evolutionary_search] start")
        if self.root.val_score is None:
            print("[evolutionary_search] evaluate root")
            self.root.val_score = evaluate_fn(self.root, "validation")

        for iteration in range(self.max_iterations):
            iter_start = time.monotonic()
            print(f"[evolutionary_search] iteration {iteration + 1}/{self.max_iterations} start")
            parent = self.sample_parent()
            print("[evolutionary_search] sample feedback")

            feedback_samples = sample_feedback_fn(self.feedback_sample_size)
            print("[evolutionary_search] run inference")
            feedback_samples = run_inference_fn(parent, feedback_samples)
            print("[evolutionary_search] select feedback samples")
            feedback_samples = select_feedback_samples(
                feedback_samples,
                selection_mode=selection_mode,
                rng=self.rng,
            )
            print("[evolutionary_search] generate feedback text")
            feedback_text = generate_feedback_fn(parent, feedback_samples)

            print("[evolutionary_search] mutate prompt")
            new_prompt = mutate_prompt_fn(parent, feedback_samples, feedback_text)
            child = GraphNode(
                inference_prompt=new_prompt,
                parent=parent,
                feedback=feedback_text,
                feedback_prompt=self.feedback_prompt,
                mutation_prompt=self.mutation_prompt,
                example_generation_prompt=self.example_generation_prompt,
            )
            print("[evolutionary_search] evaluate child")
            child.val_score = evaluate_fn(child, "validation")

            print("[evolutionary_search] update graph")
            parent.add_child_from_feedback(feedback_samples, child)
            self.population.append(child)

            if on_iteration_end is not None:
                print("[evolutionary_search] on_iteration_end hook")
                on_iteration_end(iteration, parent, child)

            iter_elapsed = time.monotonic() - iter_start
            total_elapsed = time.monotonic() - start_time
            print(
                f"[evolutionary_search] iteration {iteration + 1} end "
                f"(iter {iter_elapsed:.2f}s, total {total_elapsed:.2f}s)"
            )

        total_elapsed = time.monotonic() - start_time
        print(f"[evolutionary_search] end (total {total_elapsed:.2f}s)")
        
        return self.best_node(), self.population
