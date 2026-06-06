"""Core evaluation metrics for traceability recovery experiments."""

from typing import Dict, List, Sequence, Set

import numpy as np


def average_precision(ranked: Sequence[str], relevant: Set[str]) -> float:
    """Compute average precision for one query.

    Args:
        ranked: Ranked retrieved code ids.
        relevant: Relevant code ids.

    Returns:
        Average precision score.
    """
    hits = 0
    score = 0.0
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    if len(relevant) == 0:
        return 0.0
    return score / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Set[str]) -> float:
    """Compute reciprocal rank for one query.

    Args:
        ranked: Ranked retrieved code ids.
        relevant: Relevant code ids.

    Returns:
        Reciprocal rank score.
    """
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """Compute recall@k for one query.

    Args:
        ranked: Ranked retrieved code ids.
        relevant: Relevant code ids.
        k: Cutoff rank.

    Returns:
        Recall@k score.
    """
    retrieved = ranked[:k]
    hits = len(set(retrieved).intersection(relevant))
    if len(relevant) == 0:
        return 0.0
    return hits / len(relevant)


def evaluate_rankings(rankings: Dict[str, List[str]], ground_truth: Dict[str, Set[str]]) -> Dict[str, float | List[float]]:
    """Evaluate rankings against many-to-many ground truth.

    Args:
        rankings: Requirement -> ranked code ids.
        ground_truth: Requirement -> relevant code ids.

    Returns:
        Dictionary containing MAP, MRR, Recall@5, Recall@10, and AP_scores.

    Raises:
        ValueError: If no valid query could be evaluated.
    """
    aps: List[float] = []
    rrs: List[float] = []
    recalls5: List[float] = []
    recalls10: List[float] = []

    for req_id, relevant in ground_truth.items():
        if req_id not in rankings:
            continue
        ranked = rankings[req_id]
        aps.append(average_precision(ranked, relevant))
        rrs.append(reciprocal_rank(ranked, relevant))
        recalls5.append(recall_at_k(ranked, relevant, 5))
        recalls10.append(recall_at_k(ranked, relevant, 10))

    if not aps:
        raise ValueError("Metrics cannot be computed with zero valid queries")

    return {
        "MAP": float(np.mean(aps)),
        "MRR": float(np.mean(rrs)),
        "Recall@5": float(np.mean(recalls5)),
        "Recall@10": float(np.mean(recalls10)),
        "AP_scores": aps,
    }
