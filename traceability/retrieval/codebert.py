"""CodeBERT-based dense retrieval for traceability ranking."""

from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import EMBEDDING_MODEL_NAMES
from traceability.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_codebert(req_docs: Dict[str, str], code_docs: Dict[str, str]) -> Tuple[Dict[str, List[str]], np.ndarray, List[str], List[str]]:
    """Run CodeBERT retrieval and produce rankings.

    Args:
        req_docs: Requirement corpus.
        code_docs: Code corpus.

    Returns:
        Rankings, similarity matrix, requirement ids, code ids.

    Raises:
        RuntimeError: If model loading or embedding fails.
    """
    logger.info("Model being executed: codebert")

    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAMES["codebert"])
    except Exception as exc:
        raise RuntimeError(f"Failed model loading: CodeBERT ({exc})") from exc

    req_ids = list(req_docs.keys())
    code_ids = list(code_docs.keys())

    try:
        req_embeddings = model.encode(list(req_docs.values()), convert_to_numpy=True)
        code_embeddings = model.encode(list(code_docs.values()), convert_to_numpy=True)
    except Exception as exc:
        raise RuntimeError(f"Failed model loading: CodeBERT encoding ({exc})") from exc

    similarities = cosine_similarity(req_embeddings, code_embeddings)

    rankings: Dict[str, List[str]] = {}
    for i, req_id in enumerate(req_ids):
        ranked_indices = np.argsort(similarities[i])[::-1]
        rankings[req_id] = [code_ids[idx] for idx in ranked_indices]

    return rankings, similarities, req_ids, code_ids
