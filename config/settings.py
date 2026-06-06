"""Global framework settings for reproducible traceability experiments."""

from pathlib import Path

RANDOM_SEED = 42

SUPPORTED_FILE_EXTENSIONS = {
    ".java",
    ".py",
    ".js",
    ".ts",
    ".cpp",
    ".c",
    ".cs",
    ".go",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".xml",
    ".json",
    ".yaml",
    ".yml",
}

TOP_K_VALUES = [5, 10]
DEFAULT_TOP_K = 10

MODEL_NAMES = ["tfidf", "bm25", "sbert", "codebert"]

EMBEDDING_MODEL_NAMES = {
    "sbert": "all-MiniLM-L6-v2",
    "codebert": "microsoft/codebert-base",
}

LOGGING_LEVEL = "INFO"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIRECTORIES = {
    "rankings": RESULTS_ROOT / "rankings",
    "metrics": RESULTS_ROOT / "metrics",
    "statistics": RESULTS_ROOT / "statistics",
    "error_analysis": RESULTS_ROOT / "error_analysis",
    "visualizations": RESULTS_ROOT / "visualizations",
}
