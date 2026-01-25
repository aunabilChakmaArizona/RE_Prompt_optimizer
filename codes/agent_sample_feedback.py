from __future__ import annotations

import random
from typing import Iterable, List, Optional

from agent_feedback_samples import FeedbackSample, FeedbackSamples


def sample_feedback_fn(
    k: int,
    feedback_pool: Iterable[dict],
    rng: Optional[random.Random] = None,
) -> FeedbackSamples:
    pool_list = list(feedback_pool)
    rng = rng or random.Random()

    if not pool_list:
        return FeedbackSamples()

    selected = rng.sample(pool_list, k=min(k, len(pool_list)))

    feedback_samples = FeedbackSamples()
    for entry in pool_list:
        sample = FeedbackSample(
            id_1shot=entry.get("id_1shot", -1),
            id_query=entry.get("id_query", -1),
            inference="",
            label=entry.get("label", ""),
        )
        sample.relation = entry.get("relation", "")
        sample.relation_description = entry.get("relation_description", "")
        sample.support_sentence = entry.get("support_sentence", "")
        sample.query_sentence = entry.get("query_sentence", "")
        feedback_samples.add_to_all_samples(sample)

    for entry in selected:
        sample = FeedbackSample(
            id_1shot=entry.get("id_1shot", -1),
            id_query=entry.get("id_query", -1),
            inference="",
            label=entry.get("label", ""),
        )
        sample.relation = entry.get("relation", "")
        sample.relation_description = entry.get("relation_description", "")
        sample.support_sentence = entry.get("support_sentence", "")
        sample.query_sentence = entry.get("query_sentence", "")
        feedback_samples.add_to_selected_samples(sample)

    return feedback_samples
