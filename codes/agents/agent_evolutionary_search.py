import math
import random
import time
from typing import Callable, List, Optional, Sequence

from agents.agent_feedback_samples import FeedbackSamples
from agents.agent_graph_node import GraphNode
from agents.agent_prompts import INFERENCE_MODE_NON_SEPARATE
from agents.agent_select_feedback import select_feedback_samples
from agents.agent_utils import score_node_or_neg_inf


SampleFeedbackFn = Callable[[int], FeedbackSamples]
RunInferenceFn = Callable[..., FeedbackSamples]
GenerateFeedbackFn = Callable[..., FeedbackSamples]
MutatePromptFn = Callable[..., Optional[tuple[str, str, str, str, str, str]]]
EvaluateFn = Callable[..., float]
IterationHook = Callable[[int, GraphNode, GraphNode], None]

PARENT_SELECTION_MODE_SCORE_WEIGHTED = "score_weighted"
PARENT_SELECTION_MODE_INITIAL_PROMPT_ONLY = "initial_prompt_only"
PARENT_SELECTION_MODE_CHOICES = [
    PARENT_SELECTION_MODE_SCORE_WEIGHTED,
    PARENT_SELECTION_MODE_INITIAL_PROMPT_ONLY,
]


class EvolutionarySearch:
    def __init__(
        self,
        root: GraphNode,
        max_iterations: int = 20,
        population_size: Optional[int] = None,
        feedback_sample_size: int = 100,
        population_sampling_temperature: float = 1.0,
        feedback_prompt: str = "",
        mutation_prompt: str = "",
        mutation_prompts: Optional[List[str]] = None,
        mutation_prompt_keys: Optional[List[str]] = None,
        example_generation_prompt: str = "",
        rng: Optional[random.Random] = None,
    ):
        self.root = root
        if population_size is not None and population_size <= 0:
            raise ValueError("population_size must be > 0 when provided")
        self.population: List[GraphNode] = [root]
        self.population_history: List[GraphNode] = [root]
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.feedback_sample_size = feedback_sample_size
        self.population_sampling_temperature = population_sampling_temperature
        self.feedback_prompt = feedback_prompt
        self.mutation_prompt = mutation_prompt
        self.mutation_prompts = (
            mutation_prompts if mutation_prompts is not None and len(mutation_prompts) > 0
            else [mutation_prompt]
        )
        self.mutation_prompt_keys = (
            mutation_prompt_keys
            if mutation_prompt_keys is not None and len(mutation_prompt_keys) > 0
            else ["" for _ in self.mutation_prompts]
        )
        if len(self.mutation_prompt_keys) != len(self.mutation_prompts):
            raise ValueError("mutation_prompt_keys and mutation_prompts must have same length")
        self.example_generation_prompt = example_generation_prompt
        self.rng = rng or random.Random()
        if self.root.node_id is None:
            self.root.node_id = 0
        self._next_node_id = self.root.node_id + 1

    def _prune_population(self) -> None:
        if self.population_size is None or len(self.population) <= self.population_size:
            return
        self.population.sort(
            key=lambda node: (
                score_node_or_neg_inf(node),
                node.node_id if node.node_id is not None else -1,
            ),
            reverse=True,
        )
        removed_nodes = self.population[self.population_size :]
        self.population = self.population[: self.population_size]
        removed_node_ids = [node.node_id for node in removed_nodes]
        print(
            "[agent_evolutionary_search] pruned active population to "
            f"{self.population_size}; removed node_ids={removed_node_ids}"
        )

    def _format_feedback_text(self, feedback_samples: FeedbackSamples) -> str:
        feedback_texts = getattr(feedback_samples, "feedback_texts", None)
        if feedback_texts is None:
            feedback_texts = [
                getattr(sample, "feedback_text", "") for sample in feedback_samples.selected_samples
            ]
        return "\n---\n".join([t for t in feedback_texts if t.strip()])

    def _softmax_weights(self, scores: Sequence[Optional[float]]) -> List[float]:
        if self.population_sampling_temperature <= 0:
            raise ValueError("population_sampling_temperature must be > 0")
        finite_scores = [s for s in scores if s is not None]
        if not finite_scores:
            return [1.0 for _ in scores]
        max_score = max(finite_scores)
        weights = []
        for s in scores:
            if s is None:
                weights.append(0.0)
            else:
                weights.append(
                    math.exp((s - max_score) / self.population_sampling_temperature)
                )
        if sum(weights) == 0.0:
            return [1.0 for _ in scores]
        return weights

    def _score_weighted_parent(self) -> GraphNode:
        alive_nodes = [node for node in self.population if not node.is_dead]
        if not alive_nodes:
            raise RuntimeError("No active nodes available for mutation")
        scores = []
        for node in alive_nodes:
            val_score = node.val_score
            if val_score is None:
                scores.append(None)
            else:
                scores.append(val_score.get("f1_mean"))
        weights = self._softmax_weights(scores)
        return self.rng.choices(alive_nodes, weights=weights, k=1)[0]

    def sample_parent(
        self,
        parent_selection_mode: str = PARENT_SELECTION_MODE_SCORE_WEIGHTED,
    ) -> GraphNode:
        if parent_selection_mode == PARENT_SELECTION_MODE_INITIAL_PROMPT_ONLY:
            if self.root.is_dead:
                raise RuntimeError("Initial prompt is dead; no parent available for mutation")
            return self.root
        if parent_selection_mode == PARENT_SELECTION_MODE_SCORE_WEIGHTED:
            return self._score_weighted_parent()
        raise ValueError(
            f"Unsupported parent_selection_mode={parent_selection_mode!r}. "
            f"Expected one of {PARENT_SELECTION_MODE_CHOICES}."
        )

    def best_node(self) -> GraphNode:
        return max(self.population_history, key=score_node_or_neg_inf)

    def run(
        self,
        sample_feedback_fn: SampleFeedbackFn,
        run_inference_fn: RunInferenceFn,
        generate_feedback_fn: GenerateFeedbackFn,
        mutate_prompt_fn: MutatePromptFn,
        evaluate_fn: EvaluateFn,
        selection_mode: str = "mixed",
        update_mode: str = "feedback",
        parent_selection_mode: str = PARENT_SELECTION_MODE_SCORE_WEIGHTED,
        overall_start_time: Optional[float] = None,
        on_iteration_end: Optional[IterationHook] = None,
    ) -> tuple[GraphNode, List[GraphNode]]:
        
        start_time = time.monotonic()

        def _log_step(message: str) -> None:
            if overall_start_time is None:
                print(message)
                return
            overall_elapsed = time.monotonic() - overall_start_time
            print(f"{message} (overall {overall_elapsed:.2f}s)")

        print("[agent_evolutionary_search] start")
        if self.root.val_score is None:
            print("[agent_evolutionary_search] evaluate root")
            self.root.val_score = evaluate_fn(self.root, "validation")

        for iteration in range(self.max_iterations):
            iter_start = time.monotonic()
            overall_elapsed = None
            if overall_start_time is not None:
                overall_elapsed = time.monotonic() - overall_start_time
            if overall_elapsed is None:
                print(f"[agent_evolutionary_search] iteration {iteration + 1}/{self.max_iterations} start")
            else:
                print(
                    f"[agent_evolutionary_search] iteration {iteration + 1}/{self.max_iterations} start "
                    f"(overall {overall_elapsed:.2f}s)"
                )
            if parent_selection_mode == PARENT_SELECTION_MODE_INITIAL_PROMPT_ONLY:
                if self.root.is_dead:
                    print(
                        "[agent_evolutionary_search] initial prompt is dead; stopping"
                    )
                    break
            elif not any(not node.is_dead for node in self.population):
                print("[agent_evolutionary_search] no active nodes left; stopping")
                break
            parent = self.sample_parent(parent_selection_mode=parent_selection_mode)
            feedback_text = ""
            if update_mode == "feedback": #todo: this is redundant to run during traces mutation
                _log_step("[agent_evolutionary_search] sample feedback")
                feedback_samples = sample_feedback_fn(self.feedback_sample_size)
                _log_step("[agent_evolutionary_search] run inference")
                feedback_samples = run_inference_fn(
                    parent,
                    feedback_samples,
                    evolution_iteration=iteration + 1,
                    evolution_max_iterations=self.max_iterations,
                )
                _log_step("[agent_evolutionary_search] select feedback samples")
                feedback_samples = select_feedback_samples(
                    feedback_samples,
                    selection_mode=selection_mode,
                    rng=self.rng,
                )
                _log_step("[agent_evolutionary_search] generate feedback text")
                feedback_samples = generate_feedback_fn(parent, feedback_samples)
                feedback_text = self._format_feedback_text(feedback_samples)
            else:
                feedback_samples = FeedbackSamples()

            iteration_mutation_prompt = self.mutation_prompts[
                iteration % len(self.mutation_prompts)
            ]
            iteration_mutation_prompt_key = self.mutation_prompt_keys[
                iteration % len(self.mutation_prompt_keys)
            ]
            _log_step("[agent_evolutionary_search] mutate prompt")
            mutation_result = mutate_prompt_fn(
                parent,
                feedback_samples,
                mutation_prompt_override=iteration_mutation_prompt,
                mutation_prompt_key_override=iteration_mutation_prompt_key,
            )
            if mutation_result is None:
                print("[agent_evolutionary_search] mutation failed; parent marked dead")
                continue
            (
                new_prompt,
                raw_mutation_response,
                mutation_prompt_used,
                raw_differentiation_response,
                differentiation_prompt_used,
                differentiation,
            ) = mutation_result
            child_instruction_prompt = parent.inference_instruction_prompt
            if parent.inference_mode != INFERENCE_MODE_NON_SEPARATE:
                child_instruction_prompt = new_prompt
            child = GraphNode(
                inference_prompt=new_prompt,
                inference_mode=parent.inference_mode,
                inference_instruction_prompt=child_instruction_prompt,
                inference_answer_instruction_prompt=parent.inference_answer_instruction_prompt,
                inference_example_prompt=parent.inference_example_prompt,
                inference_input_prompt=parent.inference_input_prompt,
                parent=parent,
                feedback=feedback_text,
                raw_feedback_texts=getattr(feedback_samples, "raw_feedback_texts", None),
                feedback_prompts_used=getattr(feedback_samples, "feedback_prompts", None),
                feedback_prompt=self.feedback_prompt,
                mutation_prompt=iteration_mutation_prompt,
                example_generation_prompt=self.example_generation_prompt,
                mutation_prompt_used=mutation_prompt_used,
                raw_mutation_response=raw_mutation_response,
                differentiation_prompt_used=differentiation_prompt_used,
                raw_differentiation_response=raw_differentiation_response,
                differentiation=differentiation,
                node_id=self._next_node_id,
            )
            self._next_node_id += 1
            _log_step("[agent_evolutionary_search] evaluate child")
            child.val_score = evaluate_fn(
                child,
                "validation",
                evolution_iteration=iteration + 1,
                evolution_max_iterations=self.max_iterations,
            )

            _log_step("[agent_evolutionary_search] update graph")
            parent.add_child_from_feedback(feedback_samples, child)
            self.population.append(child)
            self.population_history.append(child)
            self._prune_population()

            if on_iteration_end is not None:
                _log_step("[agent_evolutionary_search] on_iteration_end hook")
                on_iteration_end(iteration, parent, child)

            iter_elapsed = time.monotonic() - iter_start
            total_elapsed = time.monotonic() - start_time
            overall_elapsed = None
            if overall_start_time is not None:
                overall_elapsed = time.monotonic() - overall_start_time
            if overall_elapsed is None:
                print(
                    f"[agent_evolutionary_search] iteration {iteration + 1} end "
                    f"(iter {iter_elapsed:.2f}s, total {total_elapsed:.2f}s)"
                )
            else:
                print(
                    f"[agent_evolutionary_search] iteration {iteration + 1} end "
                    f"(iter {iter_elapsed:.2f}s, total {total_elapsed:.2f}s, overall {overall_elapsed:.2f}s)"
                )

        total_elapsed = time.monotonic() - start_time
        if overall_start_time is None:
            print(f"[agent_evolutionary_search] end (total {total_elapsed:.2f}s)")
        else:
            overall_elapsed = time.monotonic() - overall_start_time
            print(
                f"[agent_evolutionary_search] end (total {total_elapsed:.2f}s, "
                f"overall {overall_elapsed:.2f}s)"
            )
        
        return self.best_node(), self.population_history
