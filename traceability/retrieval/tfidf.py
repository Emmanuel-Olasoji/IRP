"""TF-IDF retrieval baseline for requirements-to-code ranking."""

from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from traceability.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_tfidf(req_docs: Dict[str, str], code_docs: Dict[str, str]) -> Tuple[Dict[str, List[str]], np.ndarray, List[str], List[str]]:
    """Run TF-IDF retrieval and produce rankings.

    Args:
        req_docs: Requirement corpus.
        code_docs: Code corpus.

    Returns:
        Rankings, similarity matrix, requirement ids, code ids.
    """
    logger.info("Model being executed: tfidf")

    req_ids = list(req_docs.keys())
    code_ids = list(code_docs.keys())
    req_texts = list(req_docs.values())
    code_texts = list(code_docs.values())

    vectorizer = TfidfVectorizer()
    code_matrix = vectorizer.fit_transform(code_texts)
    req_matrix = vectorizer.transform(req_texts)
    similarities = cosine_similarity(req_matrix, code_matrix)

    rankings: Dict[str, List[str]] = {}
    for i, req_id in enumerate(req_ids):
        ranked_indices = np.argsort(similarities[i])[::-1]
        rankings[req_id] = [code_ids[idx] for idx in ranked_indices]

    return rankings, similarities, req_ids, code_ids
