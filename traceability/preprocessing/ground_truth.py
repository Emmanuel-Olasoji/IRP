"""Ground-truth loading and normalization for trace link evaluation."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Set

import pandas as pd

from traceability.utils.logging_utils import get_logger
from traceability.utils.text_utils import normalize_id

logger = get_logger(__name__)


def load_ground_truth(ground_truth_file: Path) -> Dict[str, Set[str]]:
    """Load trace links from CSV and normalize identifiers.

    Args:
        ground_truth_file: Path to ground truth CSV.

    Returns:
        Mapping requirement_id -> set of code_ids.

    Raises:
        FileNotFoundError: If ground truth file is missing.
        ValueError: If expected columns are not present.
    """
    if not ground_truth_file.exists():
        raise FileNotFoundError(f"Missing ground truth file: {ground_truth_file}")

    df = pd.read_csv(ground_truth_file)
    df.columns = [column.strip().lower() for column in df.columns]

    if "requirement_id" not in df.columns:
        raise ValueError("Ground truth is missing 'requirement_id' column")

    code_column = "code_file" if "code_file" in df.columns else "code_id" if "code_id" in df.columns else None
    if code_column is None:
        raise ValueError("Ground truth requires one of: 'code_file' or 'code_id'")

    trace_map: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        req_id = normalize_id(str(row["requirement_id"]))
        code_id = normalize_id(str(row[code_column]))
        if req_id and code_id:
            trace_map[req_id].add(code_id)

    trace_count = sum(len(code_set) for code_set in trace_map.values())
    if trace_count == 0:
        raise ValueError("No matching trace links were loaded from ground truth")

    logger.info("Trace link count: %d", trace_count)
    return dict(trace_map)


def validate_ground_truth_alignment(
    ground_truth: Dict[str, Set[str]],
    requirement_ids: Iterable[str],
    code_ids: Iterable[str],
) -> None:
    """Validate id overlap between processed corpora and ground truth.

    Args:
        ground_truth: Mapping requirement_id -> relevant code_ids.
        requirement_ids: Processed requirement IDs.
        code_ids: Processed code IDs.

    Raises:
        ValueError: If no requirement or code ids overlap ground truth.
    """
    req_id_set = set(requirement_ids)
    code_id_set = set(code_ids)

    overlapping_requirements = req_id_set.intersection(ground_truth.keys())
    if not overlapping_requirements:
        raise ValueError("Ground truth IDs do not match processed requirement IDs")

    overlapping_codes = set()
    for requirement in overlapping_requirements:
        overlapping_codes.update(code_id_set.intersection(ground_truth[requirement]))

    if not overlapping_codes:
        raise ValueError("Ground truth IDs do not match processed code IDs")
