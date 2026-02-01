from __future__ import annotations

import random
from typing import List, Optional

from agents.agent_feedback_samples import FeedbackSample, FeedbackSamples


def _split_by_correctness(
    samples: List[FeedbackSample],
) -> tuple[List[FeedbackSample], List[FeedbackSample]]:
    correct: List[FeedbackSample] = []
    mistakes: List[FeedbackSample] = []
    for sample in samples:
        if getattr(sample, "inference", None) == getattr(sample, "label", None):
            correct.append(sample)
        else:
            mistakes.append(sample)
    return correct, mistakes


def _take_random(
    rng: random.Random,
    samples: List[FeedbackSample],
    k: int,
) -> List[FeedbackSample]:
    if k <= 0 or not samples:
        return []
    if len(samples) <= k:
        return list(samples)
    return rng.sample(samples, k)


def _select_mixed(
    rng: random.Random,
    correct: List[FeedbackSample],
    mistakes: List[FeedbackSample],
    k: int,
) -> List[FeedbackSample]:
    if k <= 0:
        return []
    if mistakes and correct:
        selected = [
            rng.choice(mistakes),
            rng.choice(correct),
        ]
        remaining_pool = [s for s in mistakes + correct if s not in selected]
        selected += _take_random(rng, remaining_pool, k - 2)
        rng.shuffle(selected)
        return selected

    if mistakes:
        return _take_random(rng, mistakes, k)
    return _take_random(rng, correct, k)

def select_feedback_samples(
    feedback_samples: FeedbackSamples,
    *,
    selection_mode: str = "mixed",
    k: int = 3,
    rng: Optional[random.Random] = None,
) -> FeedbackSamples:
    rng = rng or random.Random()

    samples = feedback_samples.all_samples
    if not samples:
        feedback_samples.selected_samples = []
        return feedback_samples

    correct, mistakes = _split_by_correctness(samples)

    if selection_mode == "correct":
        selected = _take_random(rng, correct, k)
    elif selection_mode == "mistakes":
        selected = _take_random(rng, mistakes, k)
    else:
        selected = _select_mixed(rng, correct, mistakes, k)

    feedback_samples.selected_samples = selected
    return feedback_samples
