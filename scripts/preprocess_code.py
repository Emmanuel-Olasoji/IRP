"""Backward-compatible wrapper for code preprocessing."""

import argparse

from config.dataset_config import get_dataset_paths
from traceability.preprocessing.code import preprocess_code
from traceability.utils.file_utils import validate_dataset_paths
from traceability.utils.logging_utils import setup_logging


def main() -> None:
    """Run code preprocessing.

    Supports both legacy positional path and dataset-based execution.
    """
    parser = argparse.ArgumentParser(description="Preprocess code")
    parser.add_argument("dataset_path", nargs="?", default=None, help="Legacy dataset path")
    parser.add_argument("--dataset", default=None, help="Dataset key: itrust, etour, eanci, smos")
    args = parser.parse_args()

    setup_logging()

    if args.dataset:
        paths = get_dataset_paths(args.dataset)
    elif args.dataset_path:
        from pathlib import Path

        root = Path(args.dataset_path)
        paths = {
            "root": root,
            "code": root / "code",
            "code_processed": root / "code_processed",
            "metadata": root / "metadata",
            "req": root / "req",
            "traces": root / "traces",
            "ground_truth": root / "traces" / "ground_tracelinks.csv",
        }
    else:
        raise ValueError("Provide --dataset or legacy dataset_path")

    validate_dataset_paths(paths)
    preprocess_code(paths["code"], paths["code_processed"], paths["metadata"])


if __name__ == "__main__":
    main()
