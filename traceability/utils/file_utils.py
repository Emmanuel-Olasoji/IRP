"""File and directory utilities for dataset validation and IO."""

import json
from pathlib import Path
from typing import Iterable, List

from traceability.utils.logging_utils import get_logger

logger = get_logger(__name__)


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not exist.

    Args:
        path: Directory path.
    """
    path.mkdir(parents=True, exist_ok=True)


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Create multiple directories.

    Args:
        paths: Iterable of directory paths.
    """
    for path in paths:
        ensure_dir(path)


def read_text(path: Path) -> str:
    """Read UTF-8 text from file with error-tolerant decoding.

    Args:
        path: File path.

    Returns:
        File content.
    """
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: Path, content: str) -> None:
    """Write UTF-8 text and create parent directories.

    Args:
        path: Target file path.
        content: Text to persist.
    """
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, data: dict) -> None:
    """Write dictionary as pretty JSON.

    Args:
        path: Target file path.
        data: Serializable dictionary.
    """
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def list_files_recursive(root: Path, extensions: Iterable[str]) -> List[Path]:
    """Collect files under root with selected extensions.

    Args:
        root: Root directory.
        extensions: Allowed suffixes.

    Returns:
        Sorted file list.
    """
    ext_set = {ext.lower() for ext in extensions}
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in ext_set
    ]
    return sorted(files)


def validate_dataset_paths(dataset_paths: dict) -> None:
    """Validate mandatory dataset directories and files.

    Args:
        dataset_paths: Dataset path mapping.

    Raises:
        FileNotFoundError: If required folders or trace file are missing.
    """
    root = dataset_paths["root"]
    if not root.exists():
        raise FileNotFoundError(f"Missing dataset folder: {root}")

    required_dirs = [dataset_paths["req"], dataset_paths["code"], dataset_paths["traces"]]
    for directory in required_dirs:
        if not directory.exists():
            raise FileNotFoundError(f"Missing required folder: {directory}")

    if not dataset_paths["ground_truth"].exists():
        raise FileNotFoundError(f"Missing ground truth file: {dataset_paths['ground_truth']}")

    logger.info("Dataset loaded: %s", root)
