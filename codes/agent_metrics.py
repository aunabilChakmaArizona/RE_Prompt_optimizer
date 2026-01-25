from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from scorer import score


def _chunk_indices(total: int, n_chunks: int) -> List[Tuple[int, int]]:
    if n_chunks <= 0:
        raise ValueError("n_chunks must be >= 1")
    base = total // n_chunks
    remainder = total % n_chunks
    indices = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < remainder else 0)
        end = start + size
        indices.append((start, end))
        start = end
    return indices


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var**0.5


def compute_prf_stats(
    labels: Sequence[str],
    predictions: Sequence[str],
    *,
    n_chunks: int = 1,
) -> Dict[str, float]:
    if not labels:
        return {
            "precision_mean": 0.0,
            "precision_std": 0.0,
            "recall_mean": 0.0,
            "recall_std": 0.0,
            "f1_mean": 0.0,
            "f1_std": 0.0,
        }

    n_chunks = max(1, min(n_chunks, len(labels)))
    chunk_bounds = _chunk_indices(len(labels), n_chunks)

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for start, end in chunk_bounds:
        if start == end:
            continue
        p, r, f1 = score(labels[start:end], predictions[start:end], verbose=False)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    p_mean, p_std = _mean_std(precisions)
    r_mean, r_std = _mean_std(recalls)
    f1_mean, f1_std = _mean_std(f1s)

    return {
        "precision_mean": p_mean,
        "precision_std": p_std,
        "recall_mean": r_mean,
        "recall_std": r_std,
        "f1_mean": f1_mean,
        "f1_std": f1_std,
    }
