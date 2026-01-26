from __future__ import annotations

import random
from typing import Dict, List, Optional

from agents.agent_feedback_samples import FeedbackSample, FeedbackSamples


def sample_feedback_fn(
    feedback_pool: Dict[str, List[dict]],
    k: int = 100,
    rng: Optional[random.Random] = None,
) -> FeedbackSamples:
    rng = rng or random.Random()

    if not feedback_pool:
        return FeedbackSamples()

    print(f"sample_feedback_fn: k={k}, relations={len(feedback_pool)}")

    candidate_relations = [
        relation
        for relation, instances in feedback_pool.items()
        if relation != "no_relation" and instances
    ]
    if not candidate_relations:
        print("sample_feedback_fn: no candidate relations found")
        return FeedbackSamples()

    selected_relations = rng.choices(candidate_relations, k=k)

    feedback_samples = FeedbackSamples()
    for relation in selected_relations:
        support_candidates = feedback_pool.get(relation, [])
        if not support_candidates:
            continue

        support_instance = rng.choice(support_candidates)
        other_relations = [
            rel
            for rel, instances in feedback_pool.items()
            if rel != relation and instances
        ]

        choose_same_relation = rng.random() < 0.2
        if choose_same_relation:
            query_relation = relation
            query_candidates = [
                inst for inst in support_candidates if inst is not support_instance
            ]
            query_instance = rng.choice(query_candidates)
        else:
            query_relation = rng.choice(other_relations)
            query_instance = rng.choice(feedback_pool[query_relation])

        sample = FeedbackSample(
            id_1shot=support_instance.get("id_1shot", support_instance.get("id", -1)),
            id_query=query_instance.get("id_query", query_instance.get("id", -1)),
            inference="",
            label="yes" if query_relation == relation else "no",
        )
        feedback_samples.add_to_all_samples(sample)

    print(
        f"sample_feedback_fn: done, "
        f"all_samples={len(feedback_samples.all_samples)}, "
        f"selected_samples={len(feedback_samples.selected_samples)}"
    )
    return feedback_samples
