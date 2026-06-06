"""BM25 lexical retrieval model for traceability ranking."""

from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from traceability.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_bm25(req_docs: Dict[str, str], code_docs: Dict[str, str]) -> Tuple[Dict[str, List[str]], np.ndarray, List[str], List[str]]:
    """Run BM25 retrieval and produce rankings.

    Args:
        req_docs: Requirement corpus.
        code_docs: Code corpus.

    Returns:
        Rankings, similarity matrix, requirement ids, code ids.
    """
    logger.info("Model being executed: bm25")

    req_ids = list(req_docs.keys())
    code_ids = list(code_docs.keys())

    tokenized_code = [doc.split() for doc in code_docs.values()]
    bm25 = BM25Okapi(tokenized_code)

    all_scores = []
    rankings: Dict[str, List[str]] = {}

    for req_id in req_ids:
        query = req_docs[req_id].split()
        scores = bm25.get_scores(query)
        all_scores.append(scores)

        ranked_indices = np.argsort(scores)[::-1]
        rankings[req_id] = [code_ids[idx] for idx in ranked_indices]

    similarity_matrix = np.vstack(all_scores)
    return rankings, similarity_matrix, req_ids, code_ids
