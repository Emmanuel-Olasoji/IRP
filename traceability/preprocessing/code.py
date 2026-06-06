"""Code preprocessing based on identifier, structure, and comment extraction."""

import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer

from config.settings import RANDOM_SEED, SUPPORTED_FILE_EXTENSIONS
from traceability.utils.file_utils import ensure_dir, write_json, write_text
from traceability.utils.logging_utils import get_logger
from traceability.utils.text_utils import canonical_id

logger = get_logger(__name__)

nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()

PROGRAMMING_STOPWORDS = {
    "public",
    "private",
    "protected",
    "class",
    "void",
    "return",
    "this",
    "null",
    "static",
    "final",
    "new",
    "true",
    "false",
    "package",
    "import",
    "extends",
    "implements",
    "try",
    "catch",
    "throw",
    "throws",
    "string",
    "int",
    "float",
    "double",
    "boolean",
    "main",
    "args",
    "system",
    "out",
    "println",
    "object",
    "interface",
    "enum",
    "serializable",
    "override",
}

COMMENT_STOPWORDS = {
    "method",
    "returns",
    "return",
    "sets",
    "constructor",
    "param",
    "author",
    "empty",
    "dependencies",
    "dependency",
    "information",
    "string",
    "logs",
    "log",
    "into",
    "that",
    "who",
    "the",
    "allows",
    "allow",
    "convert",
    "converts",
    "management",
    "user",
    "users",
    "bean",
    "object",
    "class",
}

GENERIC_IDENTIFIERS = {"tmp", "obj", "var", "val", "data", "test"}

MIN_TOKEN_LENGTH = 3
MAX_TOKEN_FREQUENCY = 2
PRESERVE_COMMENTS = True


def split_identifiers(text: str) -> str:
    """Split code identifiers into lexical tokens.

    Args:
        text: Raw identifier text.

    Returns:
        Split plain text.
    """
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_\-]", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\b[a-zA-Z]{1,2}\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"\b[a-zA-Z]+o\b",
        lambda match: match.group(0)[:-1] if len(match.group(0)) > 6 else match.group(0),
        text,
    )
    return text.strip()


def normalize_token(token: str) -> str:
    """Normalize token with lowercasing and lemmatization.

    Args:
        token: Raw token.

    Returns:
        Normalized token.
    """
    return lemmatizer.lemmatize(token.lower().strip())


def extract_comments(text: str) -> List[str]:
    """Extract and clean comments from code.

    Args:
        text: Raw code content.

    Returns:
        Filtered comment token list.
    """
    comments: List[str] = []
    comments.extend(re.findall(r"//(.*)", text))
    comments.extend(re.findall(r"/\*([\s\S]*?)\*/", text))
    comments.extend(re.findall(r"#(.*)", text))

    cleaned: List[str] = []
    for comment in comments:
        if re.search(r"author\s*[:\-]?", comment.lower()):
            continue
        chunk = split_identifiers(comment)
        chunk = re.sub(r"[^a-zA-Z\s]", " ", chunk.lower())
        chunk = re.sub(r"\s+", " ", chunk)
        for token in chunk.split():
            token = normalize_token(token)
            if token in COMMENT_STOPWORDS or len(token) < MIN_TOKEN_LENGTH:
                continue
            cleaned.append(token)
    return cleaned[:50]


def extract_package(text: str) -> List[str]:
    """Extract package name tokens.

    Args:
        text: Raw code content.

    Returns:
        Package tokens.
    """
    packages = re.findall(r"package\s+([A-Za-z0-9_.]+)", text)
    output: List[str] = []
    for package in packages:
        output.extend(package.replace(".", " ").split())
    return output


def extract_classes(text: str) -> List[str]:
    """Extract class names from code.

    Args:
        text: Raw code content.

    Returns:
        Split class identifiers.
    """
    return [split_identifiers(match) for match in re.findall(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", text)]


def extract_methods(text: str) -> List[str]:
    """Extract non-trivial method names.

    Args:
        text: Raw code content.

    Returns:
        Split method identifiers.
    """
    matches = re.findall(
        r"(?:public|private|protected|static|\s)+[\w\<\>\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        text,
    )
    skip_methods = {"get", "set", "is", "toString", "main", "equals", "hashCode", "clone", "finalize"}
    return [split_identifiers(match) for match in matches if match.lower() not in {m.lower() for m in skip_methods}]


def extract_variables(text: str) -> List[str]:
    """Extract pseudo-variable signals from comments.

    This function intentionally preserves legacy behavior from the original script,
    where only the first qualifying comment block contributes tokens.

    Args:
        text: Raw code content.

    Returns:
        Comment-derived tokens from the first qualifying block.
    """
    comments: List[str] = []
    comments.extend(re.findall(r"//(.*)", text))
    comments.extend(re.findall(r"/\*([\s\S]*?)\*/", text))
    comments.extend(re.findall(r"#(.*)", text))

    cleaned: List[str] = []
    for comment in comments:
        if re.search(r"author|federic|francesc|cinque", comment.lower()):
            continue
        chunk = split_identifiers(comment)
        chunk = re.sub(r"[^a-zA-Z\s]", " ", chunk.lower())
        chunk = re.sub(r"\s+", " ", chunk)
        for token in chunk.split():
            token = normalize_token(token)
            if token in COMMENT_STOPWORDS or len(token) < MIN_TOKEN_LENGTH:
                continue
            cleaned.append(token)
        return cleaned[:50]
    return cleaned[:50]


def clean_and_filter(tokens: List[str]) -> List[str]:
    """Filter lexical noise from token list.

    Args:
        tokens: Candidate tokens.

    Returns:
        Cleaned token list.
    """
    cleaned: List[str] = []
    for token in tokens:
        for part in token.split():
            part = normalize_token(part)
            if not part.isalpha():
                continue
            if len(part) < MIN_TOKEN_LENGTH:
                continue
            if part in PROGRAMMING_STOPWORDS or part in COMMENT_STOPWORDS or part in GENERIC_IDENTIFIERS:
                continue
            cleaned.append(part)
    return cleaned


def control_token_frequency(tokens: List[str]) -> List[str]:
    """Limit repeated terms using legacy frequency cap and de-duplication.

    Args:
        tokens: Clean token list.

    Returns:
        Final token list.
    """
    counts: Counter[str] = Counter()
    filtered: List[str] = []
    for token in tokens:
        if counts[token] < MAX_TOKEN_FREQUENCY:
            filtered.append(token)
        counts[token] += 1
    return list(dict.fromkeys(filtered))


def process_code(file_name: str, content: str) -> str:
    """Process one code file into a document string.

    Args:
        file_name: Source file name.
        content: Raw code content.

    Returns:
        Processed lexical representation.
    """
    class_name = split_identifiers(Path(file_name).stem)
    comments = extract_comments(content) if PRESERVE_COMMENTS else []
    packages = extract_package(content)
    classes = extract_classes(content)
    methods = extract_methods(content)
    variables = extract_variables(content)

    weighted_tokens = packages + [class_name] + classes * 2 + methods + variables + comments
    cleaned = clean_and_filter(weighted_tokens)
    cleaned = control_token_frequency(cleaned)
    return re.sub(r"\s+", " ", " ".join(cleaned)).strip()


def preprocess_code(code_dir: Path, code_processed_dir: Path, metadata_dir: Path) -> Dict[str, str]:
    """Preprocess all supported source files in the code corpus.

    Args:
        code_dir: Raw code folder.
        code_processed_dir: Output folder for processed code.
        metadata_dir: Metadata output folder.

    Returns:
        Mapping from normalized code id to processed text.

    Raises:
        ValueError: If code corpus is empty.
    """
    start = time.time()

    ensure_dir(code_processed_dir)
    ensure_dir(metadata_dir)

    files = sorted(
        [path for path in code_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_FILE_EXTENSIONS]
    )
    if not files:
        raise ValueError(f"Empty code corpus: {code_dir}")

    processed_map: Dict[str, str] = {}
    summary_rows: List[dict] = []
    total_before = 0
    total_after = 0

    for file_path in files:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        rel_path = file_path.relative_to(code_dir)

        before_tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", text)
        processed_text = process_code(rel_path.name, text)
        after_tokens = processed_text.split()

        cid = canonical_id(rel_path)
        write_text(code_processed_dir / rel_path, processed_text)

        processed_map[cid] = processed_text
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

    write_json(metadata_dir / "code_stats.json", stats)
    pd.DataFrame(summary_rows).to_csv(metadata_dir / "code_summary.csv", index=False)

    logger.info("Code file count: %d", len(processed_map))
    return processed_map
