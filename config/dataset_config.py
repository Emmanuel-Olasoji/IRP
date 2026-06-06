"""Dataset configuration and path resolution for supported datasets."""

from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "datasets"

_DATASET_PATHS: Dict[str, str] = {
    "itrust": "iTrust",
    "etour": "eTour",
    "eanci": "eANCI",
    "smos": "SMOS",
}


def normalize_dataset_name(dataset_name: str) -> str:
    """Normalize user dataset input to the internal key.

    Args:
        dataset_name: User-provided dataset identifier.

    Returns:
        Lowercased normalized dataset key.
    """
    return dataset_name.strip().lower()


def get_dataset_root(dataset_name: str) -> Path:
    """Return canonical dataset root path.

    Args:
        dataset_name: Dataset name, case-insensitive.

    Returns:
        Absolute path to dataset folder.

    Raises:
        ValueError: If dataset name is not supported.
    """
    key = normalize_dataset_name(dataset_name)
    if key not in _DATASET_PATHS:
        supported = ", ".join(sorted(_DATASET_PATHS.keys()))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}")
    return DATASETS_ROOT / _DATASET_PATHS[key]


def get_dataset_paths(dataset_name: str) -> Dict[str, Path]:
    """Get all standard paths for a dataset.

    Args:
        dataset_name: Dataset name, case-insensitive.

    Returns:
        Mapping with dataset root and sub-paths.
    """
    root = get_dataset_root(dataset_name)
    return {
        "root": root,
        "req": root / "req",
        "req_processed": root / "req_processed",
        "code": root / "code",
        "code_processed": root / "code_processed",
        "traces": root / "traces",
        "ground_truth": root / "traces" / "ground_tracelinks.csv",
        "metadata": root / "metadata",
        "results": root / "results",
    }
