"""Ranking export utilities for reproducible model output artifacts."""

import csv
from pathlib import Path
from typing import List

import numpy as np

from traceability.utils.file_utils import ensure_dir
from traceability.utils.logging_utils import get_logger

logger = get_logger(__name__)


def export_rankings(
    output_dir: Path,
    model_name: str,
    similarity_matrix: np.ndarray,
    requirement_ids: List[str],
    code_ids: List[str],
    top_k: int = 10,
) -> Path:
    """Export top-k rankings for each requirement.

    Args:
        output_dir: Rankings output directory.
        model_name: Model identifier.
        similarity_matrix: Query-document score matrix.
        requirement_ids: Ordered query ids.
        code_ids: Ordered document ids.
        top_k: Number of top ranked items to export.

    Returns:
        Path to exported ranking CSV.

    Raises:
        ValueError: If no ranking rows are generated.
        RuntimeError: If export fails.
    """
    ensure_dir(output_dir)
    output_file = output_dir / f"{model_name}_rankings.csv"

    row_count = 0
    try:
        with output_file.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["requirement_id", "rank", "code_file", "score"])

            for req_idx, req_id in enumerate(requirement_ids):
                scores = similarity_matrix[req_idx]
                ranked_indices = np.argsort(scores)[::-1][:top_k]
                for rank, code_idx in enumerate(ranked_indices, start=1):
                    writer.writerow([req_id, rank, code_ids[code_idx], float(scores[code_idx])])
                    row_count += 1
    except Exception as exc:
        raise RuntimeError(f"Failed ranking export: {output_file} ({exc})") from exc

    if row_count == 0:
        raise ValueError(f"Ranking outputs are empty: {output_file}")

    logger.info("Ranking export path: %s", output_file)
    return output_file
