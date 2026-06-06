"""Backward-compatible wrapper for error analysis script."""

import argparse

from config.dataset_config import get_dataset_paths
from traceability.analysis.error_analysis import run_error_analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run traceability error analysis")
    parser.add_argument("--dataset", default="smos", help="Dataset key: itrust, etour, eanci, smos")
    args = parser.parse_args()

    paths = get_dataset_paths(args.dataset)
    run_error_analysis(paths["root"], ["tfidf", "bm25", "sbert", "codebert"], 10)
