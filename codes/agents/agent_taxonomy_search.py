from __future__ import annotations

from typing import Callable, Dict, List, Optional

from agents.agent_cluster_search import (
    _assign_categories,
    _filter_mistakes,
    _format_category_block,
    _log_categories,
)
from agents.agent_feedback_samples import FeedbackSamples
from agents.agent_graph_node import GraphNode
from agents.agent_llm_prompting import run_prompt
from agents.agent_prompts import CLUSTER_MUTATION_PROMPT_V1, INFERENCE_MODE_NON_SEPARATE
from agents.agent_utils import (
    differentiate_prompts,
    extract_tagged_text,
    has_tagged_text,
    score_node_or_neg_inf,
)


RunInferenceFn = Callable[..., FeedbackSamples]
GenerateFeedbackFn = Callable[..., FeedbackSamples]
EvaluateFn = Callable[..., float]
SampleFeedbackFn = Callable[[int], FeedbackSamples]


class TaxonomySearch:
    def __init__(
        self,
        *,
        root: GraphNode,
        model,
        tokenizer,
        feedback_sample_size: int = 500,
        category_min_count: int = 2,
        max_categories: int = 10,
        num_candidates: int = 25,
        top_k: int = 5,
        feedback_examples_per_category: int = 3,
        prompt_open_tag: str = "<p>",
        prompt_close_tag: str = "</p>",
        feedback_batch_size: int = 4,
        feedback_max_new_tokens: int = 5000,
        mutation_max_new_tokens: int = 5000,
        do_sample: bool = True,
    ):
        self.root = root
        self.model = model
        self.tokenizer = tokenizer
        self.feedback_sample_size = feedback_sample_size
        self.category_min_count = category_min_count
        self.max_categories = max_categories
        self.num_candidates = num_candidates
        self.top_k = top_k
        self.feedback_examples_per_category = feedback_examples_per_category
        self.prompt_open_tag = prompt_open_tag
        self.prompt_close_tag = prompt_close_tag
        self.feedback_batch_size = feedback_batch_size
        self.feedback_max_new_tokens = feedback_max_new_tokens
        self.mutation_max_new_tokens = mutation_max_new_tokens
        self.do_sample = do_sample
        self._next_node_id = (self.root.node_id or 0) + 1

    def _next_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _mutate_from_taxonomy(
        self,
        parent: GraphNode,
        categories: List[Dict[str, object]],
        candidate_index: int,
    ) -> Optional[GraphNode]:
        category_block = _format_category_block(
            categories,
            feedback_examples_per_category=self.feedback_examples_per_category,
        )
        prompt = CLUSTER_MUTATION_PROMPT_V1
        prompt = prompt.replace("#INFERENCE_PROMPT#", parent.inference_prompt)
        prompt = prompt.replace("#CATEGORY_BLOCK#", category_block)
        prompt = prompt.replace("#PROMPT_OPEN_TAG#", self.prompt_open_tag)
        prompt = prompt.replace("#PROMPT_CLOSE_TAG#", self.prompt_close_tag)

        print(f"[agent_taxonomy_search] taxonomy mutation prompt:\n{prompt}")

        max_attempts = 10
        for attempt in range(1, max_attempts + 1):
            raw_response = run_prompt(
                prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.mutation_max_new_tokens,
                do_sample=self.do_sample,
            )
            if not has_tagged_text(raw_response, self.prompt_open_tag, self.prompt_close_tag):
                print(
                    "[agent_taxonomy_search] missing prompt tags; "
                    f"retry {attempt}/{max_attempts} for candidate={candidate_index}"
                )
                continue
            candidate_prompt = extract_tagged_text(
                raw_response, self.prompt_open_tag, self.prompt_close_tag
            )
            if not candidate_prompt:
                continue
            print(f"[agent_taxonomy_search] new prompt:\n{candidate_prompt}")

            (
                raw_differentiation_response,
                differentiation,
                differentiation_prompt_used,
            ) = differentiate_prompts(
                parent.inference_prompt,
                candidate_prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.mutation_max_new_tokens,
                do_sample=self.do_sample,
            )

            child_instruction_prompt = parent.inference_instruction_prompt
            if parent.inference_mode != INFERENCE_MODE_NON_SEPARATE:
                child_instruction_prompt = candidate_prompt

            return GraphNode(
                inference_prompt=candidate_prompt,
                inference_mode=parent.inference_mode,
                inference_instruction_prompt=child_instruction_prompt,
                inference_answer_instruction_prompt=parent.inference_answer_instruction_prompt,
                inference_example_prompt=parent.inference_example_prompt,
                inference_input_prompt=parent.inference_input_prompt,
                parent=parent,
                feedback=category_block,
                raw_feedback_texts=[],
                feedback_prompts_used=[],
                feedback_prompt=parent.feedback_prompt,
                mutation_prompt="taxonomy_search_v1",
                example_generation_prompt=parent.example_generation_prompt,
                mutation_prompt_used=prompt,
                raw_mutation_response=raw_response,
                differentiation_prompt_used=differentiation_prompt_used,
                raw_differentiation_response=raw_differentiation_response,
                differentiation=differentiation,
                node_id=self._next_id(),
            )

        return None

    def run(
        self,
        *,
        sample_feedback_fn: SampleFeedbackFn,
        run_inference_fn: RunInferenceFn,
        generate_feedback_fn: GenerateFeedbackFn,
        evaluate_fn: EvaluateFn,
    ) -> Dict[str, object]:
        print("[agent_taxonomy_search] evaluate root")
        if self.root.val_score is None:
            self.root.val_score = evaluate_fn(
                self.root,
                "validation",
                eval_id_override="taxonomy_root",
            )

        print("[agent_taxonomy_search] sample feedback")
        feedback_samples = sample_feedback_fn(self.feedback_sample_size)
        print("[agent_taxonomy_search] run inference on feedback samples")
        feedback_samples = run_inference_fn(self.root, feedback_samples)

        print("[agent_taxonomy_search] collect mistakes")
        mistake_samples = _filter_mistakes(feedback_samples)
        print(
            "[agent_taxonomy_search] mistakes: "
            f"{len(mistake_samples.selected_samples)}/{len(feedback_samples.all_samples)}"
        )

        if not mistake_samples.selected_samples:
            return {
                "root_node": self.root,
                "mistake_count": 0,
                "categories": [],
                "all_candidates": [],
                "candidate_results": [],
                "top_prompts": [],
            }

        print("[agent_taxonomy_search] generate feedback on mistakes")
        mistake_samples = generate_feedback_fn(self.root, mistake_samples)

        print("[agent_taxonomy_search] build feedback categories")
        categories = _assign_categories(
            [sample.feedback_text for sample in mistake_samples.selected_samples],
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.feedback_max_new_tokens,
            do_sample=self.do_sample,
        )
        _log_categories(categories, label="all_categories")
        categories = [
            category
            for category in categories
            if int(category["count"]) >= self.category_min_count
        ]
        categories.sort(
            key=lambda category: (-int(category["count"]), int(category["category_id"]))
        )
        categories = categories[: self.max_categories]
        _log_categories(categories, label="final_categories")

        print(
            f"[agent_taxonomy_search] kept categories={len(categories)} "
            f"(min_count={self.category_min_count}, max_categories={self.max_categories})"
        )

        all_candidates: List[GraphNode] = []
        candidate_results: List[Dict[str, object]] = []

        for candidate_index in range(self.num_candidates):
            print(
                "[agent_taxonomy_search] run candidate prompt "
                f"{candidate_index + 1}/{self.num_candidates}"
            )
            child = self._mutate_from_taxonomy(self.root, categories, candidate_index)
            if child is None:
                continue
            eval_id = f"taxonomy_candidate_{candidate_index}"
            child.val_score = evaluate_fn(
                child,
                "validation",
                eval_id_override=eval_id,
            )
            self.root.add_child_from_feedback(FeedbackSamples(), child)
            all_candidates.append(child)
            candidate_results.append(
                {
                    "candidate_index": candidate_index,
                    "category_ids": [category["category_id"] for category in categories],
                    "node_id": child.node_id,
                    "eval_id": eval_id,
                    "val_score": child.val_score,
                    "inference_prompt": child.inference_prompt,
                    "mutation_prompt_used": child.mutation_prompt_used,
                    "raw_mutation_response": child.raw_mutation_response,
                    "differentiation": child.differentiation,
                }
            )

        sorted_candidates = sorted(
            all_candidates,
            key=score_node_or_neg_inf,
            reverse=True,
        )
        top_nodes = sorted_candidates[: self.top_k]
        top_prompts: List[Dict[str, object]] = []
        for rank, node in enumerate(top_nodes, start=1):
            top_prompts.append(
                {
                    "rank": rank,
                    "node_id": node.node_id,
                    "eval_id": next(
                        (
                            candidate["eval_id"]
                            for candidate in candidate_results
                            if candidate["node_id"] == node.node_id
                        ),
                        None,
                    ),
                    "val_score": node.val_score,
                    "inference_prompt": node.inference_prompt,
                    "mutation_prompt_used": node.mutation_prompt_used,
                    "raw_mutation_response": node.raw_mutation_response,
                    "differentiation": node.differentiation,
                }
            )

        serialized_categories: List[Dict[str, object]] = []
        for category in categories:
            serialized_categories.append(
                {
                    "category_id": category["category_id"],
                    "name": category["name"],
                    "description": category["description"],
                    "count": category["count"],
                    "feedback_texts": list(category["feedback_texts"]),
                }
            )

        return {
            "root_node": self.root,
            "mistake_count": len(mistake_samples.selected_samples),
            "sampled_feedback_count": len(feedback_samples.all_samples),
            "categories": serialized_categories,
            "all_candidates": all_candidates,
            "candidate_results": candidate_results,
            "top_prompts": top_prompts,
            "mistake_samples": mistake_samples,
        }
