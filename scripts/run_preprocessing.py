"""CLI entrypoint for dataset preprocessing."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.dataset_config import get_dataset_paths
from traceability.preprocessing.code import preprocess_code
from traceability.preprocessing.requirements import preprocess_requirements
from traceability.utils.file_utils import validate_dataset_paths
from traceability.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Run preprocessing for one dataset."""
    parser = argparse.ArgumentParser(description="Run requirements and code preprocessing for one dataset.")
    parser.add_argument("--dataset", required=True, help="Dataset name: itrust, etour, eanci, smos")
    args = parser.parse_args()

    setup_logging()

    dataset_paths = get_dataset_paths(args.dataset)
    validate_dataset_paths(dataset_paths)

    preprocess_requirements(dataset_paths["req"], dataset_paths["req_processed"], dataset_paths["metadata"])
    preprocess_code(dataset_paths["code"], dataset_paths["code_processed"], dataset_paths["metadata"])

    logger.info("Preprocessing completed for dataset: %s", args.dataset)


if __name__ == "__main__":
    main()
