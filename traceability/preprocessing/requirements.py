"""Requirements preprocessing with reproducible token normalization."""

import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from config.settings import RANDOM_SEED
from traceability.utils.file_utils import ensure_dir, write_json, write_text
from traceability.utils.logging_utils import get_logger
from traceability.utils.text_utils import canonical_id, split_identifiers

logger = get_logger(__name__)

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

USE_LEMMATIZATION = True
MIN_TOKEN_LENGTH = 3
MAX_TOKEN_FREQ = 3

CUSTOM_KEEP_WORDS = {
    "user",
    "system",
    "login",
    "password",
    "authentication",
    "booking",
    "reservation",
    "payment",
    "customer",
    "employee",
    "administrator",
    "admin",
    "access",
    "permission",
    "verification",
    "schedule",
    "service",
    "email",
    "account",
    "database",
    "security",
    "session",
    "notification",
    "transaction",
}

STRUCTURE_STOPWORDS = {
    "use",
    "case",
    "name",
    "participating",
    "actor",
    "actors",
    "flow",
    "event",
    "events",
    "entry",
    "exit",
    "condition",
    "conditions",
    "quality",
    "requirement",
    "requirements",
}

NOISE_WORDS = {
    "contains",
    "character",
    "characters",
    "length",
    "empty",
    "well",
    "formed",
    "present",
    "specified",
    "called",
    "call",
    "less",
}


def clean_text(text: str) -> str:
    """Normalize free text into lexical tokens.

    Args:
        text: Raw requirement text.

    Returns:
        Cleaned plain text.
    """
    text = split_identifiers(text).lower()
    text = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Tokenize alphabetic words.

    Args:
        text: Pre-cleaned text.

    Returns:
        List of alphabetic tokens.
    """
    return re.findall(r"\b[a-zA-Z]+\b", text)


def _remove_stopwords(tokens: List[str], stop_words: set[str]) -> List[str]:
    """Remove structural and semantic noise tokens.

    Args:
        tokens: Input token list.
        stop_words: NLTK stopword set.

    Returns:
        Filtered token list.
    """
    cleaned: List[str] = []
    for token in tokens:
        if len(token) < MIN_TOKEN_LENGTH:
            continue
        if token in stop_words and token not in CUSTOM_KEEP_WORDS:
            continue
        if token in STRUCTURE_STOPWORDS or token in NOISE_WORDS:
            continue
        cleaned.append(token)
    return cleaned


def _lemmatize(tokens: List[str], lemmatizer: WordNetLemmatizer) -> List[str]:
    """Lemmatize a token sequence.

    Args:
        tokens: Input token list.
        lemmatizer: NLTK lemmatizer instance.

    Returns:
        Lemmatized tokens.
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


def _control_token_frequency(tokens: List[str]) -> List[str]:
    """Apply frequency cap used in the original scripts.

    Args:
        tokens: Input token list.

    Returns:
        Frequency-controlled token list.
    """
    counts: Counter[str] = Counter()
    filtered: List[str] = []
    for token in tokens:
        if token == "system" and counts[token] >= 1:
            counts[token] += 1
            continue
        if counts[token] < MAX_TOKEN_FREQ:
            filtered.append(token)
        counts[token] += 1
    return filtered


def _semantic_deduplicate(tokens: List[str]) -> List[str]:
    """Drop duplicates while preserving order.

    Args:
        tokens: Input token list.

    Returns:
        De-duplicated token list.
    """
    seen: set[str] = set()
    result: List[str] = []
    for token in tokens:
        if token not in seen:
            result.append(token)
            seen.add(token)
    return result


def preprocess_text(text: str, stop_words: set[str], lemmatizer: WordNetLemmatizer) -> List[str]:
    """Preprocess one requirement document.

    Args:
        text: Raw requirement text.
        stop_words: NLTK stopword set.
        lemmatizer: NLTK lemmatizer instance.

    Returns:
        Final token list.
    """
    tokens = tokenize(clean_text(text))
    tokens = _remove_stopwords(tokens, stop_words)
    if USE_LEMMATIZATION:
        tokens = _lemmatize(tokens, lemmatizer)
    tokens = _control_token_frequency(tokens)
    return _semantic_deduplicate(tokens)


def preprocess_requirements(req_dir: Path, req_processed_dir: Path, metadata_dir: Path) -> Dict[str, str]:
    """Preprocess all requirement files for a dataset.

    Args:
        req_dir: Raw requirements folder.
        req_processed_dir: Output folder for processed requirements.
        metadata_dir: Metadata output folder.

    Returns:
        Mapping from normalized requirement id to processed text.

    Raises:
        ValueError: If requirement corpus is empty.
    """
    start = time.time()

    ensure_dir(req_processed_dir)
    ensure_dir(metadata_dir)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    files = sorted(req_dir.rglob("*.txt"))
    if not files:
        raise ValueError(f"Empty requirement corpus: {req_dir}")

    processed_map: Dict[str, str] = {}
    summary_rows: List[dict] = []
    total_before = 0
    total_after = 0

    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        rel_path = file_path.relative_to(req_dir)

        before_tokens = tokenize(text)
        after_tokens = preprocess_text(text, stop_words, lemmatizer)

        cid = canonical_id(rel_path)
        final_text = " ".join(after_tokens).strip()

        output_path = req_processed_dir / rel_path
        write_text(output_path, final_text)

        processed_map[cid] = final_text
        total_before += len(before_tokens)
        total_after += len(after_tokens)

        summary_rows.append(
            {
                "canonical_id": cid,
                "file": str(rel_path),
                "tokens_before": len(before_tokens),
                "tokens_after": len(after_tokens),
            }
        )

    runtime = time.time() - start
    stats = {
        "random_seed": RANDOM_SEED,
        "files_processed": len(files),
        "avg_tokens_before": total_before / len(files),
        "avg_tokens_after": total_after / len(files),
        "runtime_seconds": runtime,
    }

    write_json(metadata_dir / "requirements_stats.json", stats)
    pd.DataFrame(summary_rows).to_csv(metadata_dir / "requirements_summary.csv", index=False)

    logger.info("Requirement count: %d", len(processed_map))
    return processed_map
