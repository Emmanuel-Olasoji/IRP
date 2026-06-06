"""Backward-compatible wrapper for the modular run pipeline script."""

from scripts.run_pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline("eanci", ["tfidf", "bm25", "sbert", "codebert"], 10)
