"""Text normalization helpers shared across preprocessing and evaluation."""

import re
from pathlib import Path


EXTENSIONS_TO_STRIP = [
    ".txt",
    ".java",
    ".py",
    ".cpp",
    ".cs",
    ".js",
    ".php",
    ".xml",
    ".json",
]


def split_identifiers(text: str) -> str:
    """Split camelCase, PascalCase, and snake_case identifiers.

    Args:
        text: Raw identifier text.

    Returns:
        Space-separated text.
    """
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"[_\-]", " ", text)
    return text


def normalize_id(value: str) -> str:
    """Normalize requirement/code ids for robust matching.

    Args:
        value: Raw id, filename, or path.

    Returns:
        Lowercased extension-free id token.
    """
    x = str(value).strip().lower().replace("\\", "/").split("/")[-1]
    for ext in EXTENSIONS_TO_STRIP:
        while x.endswith(ext):
            x = x[: -len(ext)]
    return x


def canonical_id(relative_path: Path) -> str:
    """Generate canonical id from a relative path.

    Args:
        relative_path: Relative source path.

    Returns:
        Normalized id string.
    """
    x = str(relative_path).replace("/", "_").replace("\\", "_")
    x = re.sub(r"\.[a-zA-Z0-9]+$", "", x)
    return x.lower()
