"""Statistical testing for comparing traceability retrieval models."""

from pathlib import Path
from typing import List

import pandas as pd
from scipy.stats import ttest_rel

from traceability.utils.file_utils import ensure_dir
from traceability.utils.logging_utils import get_logger

logger = get_logger(__name__)


def paired_ttest(scores_a: List[float], scores_b: List[float], name_a: str, name_b: str) -> dict:
    """Compute paired t-test over aligned AP score vectors.

    Args:
        scores_a: AP scores for model A.
        scores_b: AP scores for model B.
        name_a: Model A name.
        name_b: Model B name.

    Returns:
        Dictionary containing test result summary.
    """
    t_stat, p_value = ttest_rel(scores_a, scores_b)
    result = {
        "model_a": name_a,
        "model_b": name_b,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
    }
    logger.info("Statistical testing output: %s vs %s, p=%.6f", name_a, name_b, p_value)
    return result


def save_statistics(output_dir: Path, dataset_name: str, tests: List[dict]) -> Path:
    """Save statistical test results as CSV.

    Args:
        output_dir: Statistics output directory.
        dataset_name: Dataset name.
        tests: List of test result rows.

    Returns:
        Path to saved statistics file.
    """
    ensure_dir(output_dir)
    output_file = output_dir / f"{dataset_name}_paired_ttests.csv"
    pd.DataFrame(tests).to_csv(output_file, index=False)
    return output_file
