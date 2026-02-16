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
    if k == 1:
        return _take_random(rng, mistakes + correct, 1)

    if mistakes and correct:
        selected = [
            rng.choice(mistakes),
            rng.choice(correct),
        ]

        remaining_mistakes = [s for s in mistakes if s is not selected[0]]
        remaining_correct = [s for s in correct if s is not selected[1]]

        # Balance the random pool so one class cannot dominate extra picks.
        balanced_size = min(len(remaining_mistakes), len(remaining_correct))
        balanced_mistakes = _take_random(rng, remaining_mistakes, balanced_size)
        balanced_correct = _take_random(rng, remaining_correct, balanced_size)
        balanced_pool = balanced_mistakes + balanced_correct

        extra_selected = _take_random(rng, balanced_pool, k - 2)
        if len(extra_selected) < (k - 2):
            leftovers = [
                s
                for s in remaining_mistakes + remaining_correct
                if s not in extra_selected
            ]
            extra_selected += _take_random(rng, leftovers, (k - 2) - len(extra_selected))

        selected += extra_selected
        rng.shuffle(selected)
        return selected[:k]

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
