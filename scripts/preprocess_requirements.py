"""Backward-compatible wrapper for requirements preprocessing."""

import argparse

from config.dataset_config import get_dataset_paths
from traceability.preprocessing.requirements import preprocess_requirements
from traceability.utils.file_utils import validate_dataset_paths
from traceability.utils.logging_utils import setup_logging


def main() -> None:
    """Run requirement preprocessing.

    Supports both legacy positional path and dataset-based execution.
    """
    parser = argparse.ArgumentParser(description="Preprocess requirements")
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
            "req": root / "req",
            "req_processed": root / "req_processed",
            "metadata": root / "metadata",
            "code": root / "code",
            "traces": root / "traces",
            "ground_truth": root / "traces" / "ground_tracelinks.csv",
        }
    else:
        raise ValueError("Provide --dataset or legacy dataset_path")

    validate_dataset_paths(paths)
    preprocess_requirements(paths["req"], paths["req_processed"], paths["metadata"])


if __name__ == "__main__":
    main()
