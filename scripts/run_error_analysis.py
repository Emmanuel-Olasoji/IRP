"""CLI entrypoint for per-dataset error analysis."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.dataset_config import get_dataset_paths
from config.settings import DEFAULT_TOP_K, MODEL_NAMES
from traceability.analysis.error_analysis import run_error_analysis
from traceability.utils.logging_utils import setup_logging


def main() -> None:
    """Run error analysis for selected dataset and models."""
    parser = argparse.ArgumentParser(description="Run traceability error analysis for a dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name: itrust, etour, eanci, smos")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, help="Model list to analyze")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k threshold")
    args = parser.parse_args()

    setup_logging()
    dataset_paths = get_dataset_paths(args.dataset)
    run_error_analysis(dataset_paths["root"], [model.lower() for model in args.models], args.top_k)


if __name__ == "__main__":
    main()
