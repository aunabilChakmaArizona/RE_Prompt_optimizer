from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence

from agents.agent_feedback_samples import FeedbackSamples
from agents.agent_graph_node import GraphNode
from agents.agent_llm_prompting import run_prompt
from agents.agent_mutate_prompt import _extract_between
from agents.agent_prompts import (
    CLUSTER_CATEGORY_ASSIGNMENT_PROMPT,
    CLUSTER_MUTATION_PROMPT_V1,
    DIFFERENTIATE_PROMPT,
    INFERENCE_MODE_NON_SEPARATE,
)
from agents.agent_utils import extract_tagged_text, has_tagged_text, score_node_or_neg_inf


RunInferenceFn = Callable[..., FeedbackSamples]
GenerateFeedbackFn = Callable[..., FeedbackSamples]
EvaluateFn = Callable[..., float]
SampleFeedbackFn = Callable[[int], FeedbackSamples]


def _filter_mistakes(feedback_samples: FeedbackSamples) -> FeedbackSamples:
    mistakes = FeedbackSamples()
    mistakes.all_samples = list(feedback_samples.all_samples)
    mistakes.selected_samples = [
        sample
        for sample in feedback_samples.all_samples
        if sample.inference and sample.label and sample.inference != sample.label
    ]
    return mistakes


def _format_categories_for_assignment(categories: Sequence[Dict[str, object]]) -> str:
    if not categories:
        return "None"
    blocks = []
    for category in categories:
        blocks.append(
            "\n".join(
                [
                    f"Category ID: {category['category_id']}",
                    f"Name: {category['name']}",
                    f"Description: {category['description']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _parse_assignment(text: str) -> Dict[str, str]:
    content = extract_tagged_text(text, "<assignment>", "</assignment>")
    parsed: Dict[str, str] = {}
    for line in content.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def _assign_categories(
    feedback_texts: Sequence[str],
    *,
    model,
    tokenizer,
    max_new_tokens: int,
    do_sample: bool,
) -> List[Dict[str, object]]:
    categories: List[Dict[str, object]] = []
    next_category_id = 1

    for feedback_text in feedback_texts:
        prompt = CLUSTER_CATEGORY_ASSIGNMENT_PROMPT
        prompt = prompt.replace("#FEEDBACK#", feedback_text)
        prompt = prompt.replace(
            "#CATEGORIES#", _format_categories_for_assignment(categories)
        )
        raw_response = run_prompt(
            prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        parsed = _parse_assignment(raw_response)

        decision = parsed.get("decision", "").lower()
        category_id_text = parsed.get("category_id", "")
        matched_category: Optional[Dict[str, object]] = None
        if decision == "existing":
            for category in categories:
                if str(category["category_id"]) == category_id_text:
                    matched_category = category
                    break

        if matched_category is None:
            matched_category = {
                "category_id": next_category_id,
                "name": parsed.get("name") or f"category_{next_category_id}",
                "description": parsed.get("description") or feedback_text,
                "count": 0,
                "feedback_texts": [],
            }
            categories.append(matched_category)
            next_category_id += 1

        matched_category["count"] = int(matched_category["count"]) + 1
        matched_category["feedback_texts"].append(feedback_text)

    return categories


def _build_cluster_assignments(
    categories: Sequence[Dict[str, object]],
    *,
    num_clusters: int,
    coverage_ratio: float,
    rng: random.Random,
) -> List[Dict[str, object]]:
    if not categories or num_clusters <= 0:
        return []

    categories_per_cluster = max(1, math.ceil(len(categories) * coverage_ratio))
    categories_per_cluster = min(categories_per_cluster, len(categories))
    category_pool = list(categories)
    category_weights = [1.0 for _ in category_pool]

    clusters: List[Dict[str, object]] = []
    for cluster_index in range(num_clusters):
        selected: List[Dict[str, object]] = []
        selected_indices: set[int] = set()

        while len(selected) < categories_per_cluster:
            available_indices = [
                idx for idx in range(len(category_pool)) if idx not in selected_indices
            ]
            if not available_indices:
                break

            available_weights = [category_weights[idx] for idx in available_indices]
            if sum(available_weights) <= 0:
                available_weights = [1.0 for _ in available_indices]

            chosen_index = rng.choices(
                available_indices,
                weights=available_weights,
                k=1,
            )[0]
            selected_indices.add(chosen_index)
            selected.append(category_pool[chosen_index])

        for chosen_index in selected_indices:
            category_weights[chosen_index] *= 0.5

        clusters.append(
            {
                "cluster_id": cluster_index,
                "category_ids": [category["category_id"] for category in selected],
                "categories": selected,
            }
        )
    return clusters


def _format_category_block(categories: Sequence[Dict[str, object]]) -> str:
    blocks = []
    for category in categories:
        feedback_examples = category.get("feedback_texts", [])[:3]
        blocks.append(
            "\n".join(
                [
                    f"- Category ID: {category['category_id']}",
                    f"  Name: {category['name']}",
                    f"  Description: {category['description']}",
                    f"  Count: {category['count']}",
                    "  Representative feedback:",
                    *[f"    - {feedback_text}" for feedback_text in feedback_examples],
                ]
            )
        )
    return "\n\n".join(blocks)


def _differentiate_prompts(
    prompt_1: str,
    prompt_2: str,
    *,
    model,
    tokenizer,
    max_new_tokens: int,
    do_sample: bool,
) -> tuple[str, str]:
    differentiate_prompt = DIFFERENTIATE_PROMPT.replace("#PROMPT1#", prompt_1).replace(
        "#PROMPT2#", prompt_2
    )
    raw_response = run_prompt(
        differentiate_prompt,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
    )
    return raw_response, _extract_between(raw_response, "<d>", "</d>")


class ClusterSearch:
    def __init__(
        self,
        *,
        root: GraphNode,
        model,
        tokenizer,
        rng: Optional[random.Random] = None,
        feedback_sample_size: int = 2000,
        category_min_count: int = 2,
        max_categories: int = 10,
        num_clusters: int = 5,
        candidates_per_cluster: int = 5,
        cluster_coverage_ratio: float = 0.5,
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
        self.rng = rng or random.Random()
        self.feedback_sample_size = feedback_sample_size
        self.category_min_count = category_min_count
        self.max_categories = max_categories
        self.num_clusters = num_clusters
        self.candidates_per_cluster = candidates_per_cluster
        self.cluster_coverage_ratio = cluster_coverage_ratio
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

    def _mutate_for_cluster(
        self,
        parent: GraphNode,
        cluster: Dict[str, object],
        candidate_index: int,
    ) -> Optional[GraphNode]:
        category_block = _format_category_block(cluster["categories"])
        prompt = CLUSTER_MUTATION_PROMPT_V1
        prompt = prompt.replace("#INFERENCE_PROMPT#", parent.inference_prompt)
        prompt = prompt.replace("#CATEGORY_BLOCK#", category_block)
        prompt = prompt.replace("#PROMPT_OPEN_TAG#", self.prompt_open_tag)
        prompt = prompt.replace("#PROMPT_CLOSE_TAG#", self.prompt_close_tag)

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
                    "[agent_cluster_search] missing prompt tags; "
                    f"retry {attempt}/{max_attempts} for cluster={cluster['cluster_id']} candidate={candidate_index}"
                )
                continue
            candidate_prompt = extract_tagged_text(
                raw_response, self.prompt_open_tag, self.prompt_close_tag
            )
            if not candidate_prompt:
                continue

            raw_differentiation_response, differentiation = _differentiate_prompts(
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
                mutation_prompt="cluster_search_v1",
                example_generation_prompt=parent.example_generation_prompt,
                mutation_prompt_used=prompt,
                raw_mutation_response=raw_response,
                differentiation_prompt_used=DIFFERENTIATE_PROMPT,
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
        print("[agent_cluster_search] evaluate root")
        if self.root.val_score is None:
            self.root.val_score = evaluate_fn(
                self.root,
                "validation",
                eval_id_override="cluster_root",
            )

        print("[agent_cluster_search] sample feedback")
        feedback_samples = sample_feedback_fn(self.feedback_sample_size)
        print("[agent_cluster_search] run inference on feedback samples")
        feedback_samples = run_inference_fn(self.root, feedback_samples)

        print("[agent_cluster_search] collect mistakes")
        mistake_samples = _filter_mistakes(feedback_samples)
        print(
            "[agent_cluster_search] mistakes: "
            f"{len(mistake_samples.selected_samples)}/{len(feedback_samples.all_samples)}"
        )

        if not mistake_samples.selected_samples:
            return {
                "root_node": self.root,
                "mistake_count": 0,
                "categories": [],
                "clusters": [],
                "all_candidates": [],
                "best_prompts": [],
            }

        print("[agent_cluster_search] generate feedback on mistakes")
        mistake_samples = generate_feedback_fn(self.root, mistake_samples)

        print("[agent_cluster_search] build feedback categories")
        categories = _assign_categories(
            [sample.feedback_text for sample in mistake_samples.selected_samples],
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.feedback_max_new_tokens,
            do_sample=self.do_sample,
        )
        categories = [
            category
            for category in categories
            if int(category["count"]) >= self.category_min_count
        ]
        categories.sort(
            key=lambda category: (-int(category["count"]), int(category["category_id"]))
        )
        categories = categories[: self.max_categories]

        print(
            f"[agent_cluster_search] kept categories={len(categories)} "
            f"(min_count={self.category_min_count}, max_categories={self.max_categories})"
        )

        clusters = _build_cluster_assignments(
            categories,
            num_clusters=self.num_clusters,
            coverage_ratio=self.cluster_coverage_ratio,
            rng=self.rng,
        )

        all_candidates: List[GraphNode] = []
        candidate_results: List[Dict[str, object]] = []
        best_prompts: List[Dict[str, object]] = []

        for cluster in clusters:
            print(
                "[agent_cluster_search] optimize cluster "
                f"{cluster['cluster_id']} with categories={cluster['category_ids']}"
            )
            cluster_candidates: List[GraphNode] = []

            for candidate_index in range(self.candidates_per_cluster):
                child = self._mutate_for_cluster(self.root, cluster, candidate_index)
                if child is None:
                    continue
                eval_id = f"cluster_{cluster['cluster_id']}_candidate_{candidate_index}"
                child.val_score = evaluate_fn(
                    child,
                    "validation",
                    eval_id_override=eval_id,
                )
                self.root.add_child_from_feedback(FeedbackSamples(), child)
                cluster_candidates.append(child)
                all_candidates.append(child)
                candidate_results.append(
                    {
                        "cluster_id": cluster["cluster_id"],
                        "candidate_index": candidate_index,
                        "category_ids": list(cluster["category_ids"]),
                        "node_id": child.node_id,
                        "eval_id": eval_id,
                        "val_score": child.val_score,
                        "inference_prompt": child.inference_prompt,
                    }
                )

            if not cluster_candidates:
                continue

            best_node = max(cluster_candidates, key=score_node_or_neg_inf)
            best_prompts.append(
                {
                    "cluster_id": cluster["cluster_id"],
                    "category_ids": list(cluster["category_ids"]),
                    "node_id": best_node.node_id,
                    "eval_id": next(
                        (
                            candidate["eval_id"]
                            for candidate in candidate_results
                            if candidate["node_id"] == best_node.node_id
                        ),
                        None,
                    ),
                    "val_score": best_node.val_score,
                    "inference_prompt": best_node.inference_prompt,
                    "mutation_prompt_used": best_node.mutation_prompt_used,
                    "raw_mutation_response": best_node.raw_mutation_response,
                    "differentiation": best_node.differentiation,
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

        serialized_clusters: List[Dict[str, object]] = []
        for cluster in clusters:
            serialized_clusters.append(
                {
                    "cluster_id": cluster["cluster_id"],
                    "category_ids": list(cluster["category_ids"]),
                }
            )

        return {
            "root_node": self.root,
            "mistake_count": len(mistake_samples.selected_samples),
            "sampled_feedback_count": len(feedback_samples.all_samples),
            "categories": serialized_categories,
            "clusters": serialized_clusters,
            "all_candidates": all_candidates,
            "candidate_results": candidate_results,
            "best_prompts": best_prompts,
            "mistake_samples": mistake_samples,
        }
