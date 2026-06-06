"""Run full reproducible traceability experiment across all datasets."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.dataset_config import get_dataset_paths
from config.settings import MODEL_NAMES, OUTPUT_DIRECTORIES
from scripts.run_pipeline import run_pipeline
from traceability.analysis.error_analysis import run_error_analysis
from traceability.preprocessing.code import preprocess_code
from traceability.preprocessing.requirements import preprocess_requirements
from traceability.utils.file_utils import validate_dataset_paths
from traceability.visualization.plots import generate_visualizations
from traceability.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Run pipeline for all supported datasets, then generate visualizations."""
    setup_logging()

    for dataset in ["iTrust", "eTour", "eANCI", "SMOS"]:
        logger.info("Running preprocessing for dataset: %s", dataset)
        paths = get_dataset_paths(dataset)
        validate_dataset_paths(paths)
        preprocess_requirements(paths["req"], paths["req_processed"], paths["metadata"])
        preprocess_code(paths["code"], paths["code_processed"], paths["metadata"])

        logger.info("Running pipeline for dataset: %s", dataset)
        run_pipeline(dataset, MODEL_NAMES)

        logger.info("Running error analysis for dataset: %s", dataset)
        run_error_analysis(paths["root"], MODEL_NAMES)

    generate_visualizations(OUTPUT_DIRECTORIES["metrics"].parent, OUTPUT_DIRECTORIES["visualizations"])
    logger.info("All datasets completed.")


if __name__ == "__main__":
    main()
