"""CLI entrypoint for end-to-end retrieval and evaluation pipeline."""

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config.dataset_config import get_dataset_paths, normalize_dataset_name
from config.settings import DEFAULT_TOP_K, MODEL_NAMES, OUTPUT_DIRECTORIES, RANDOM_SEED
from traceability.evaluation.metrics import evaluate_rankings
from traceability.evaluation.rankings import export_rankings
from traceability.evaluation.statistics import paired_ttest, save_statistics
from traceability.preprocessing.ground_truth import load_ground_truth, validate_ground_truth_alignment
from traceability.retrieval.bm25 import run_bm25
from traceability.retrieval.codebert import run_codebert
from traceability.retrieval.sbert import run_sbert
from traceability.retrieval.tfidf import run_tfidf
from traceability.utils.file_utils import ensure_dirs, validate_dataset_paths
from traceability.utils.logging_utils import get_logger, setup_logging
from traceability.utils.text_utils import normalize_id

logger = get_logger(__name__)


def set_reproducible_seed(seed: int = RANDOM_SEED) -> None:
    """Set deterministic seeds across supported libraries.

    Args:
        seed: Integer seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        logger.warning("PyTorch not available during seeding; continuing without torch seed")


def load_processed_docs(directory: Path) -> Dict[str, str]:
    """Load processed text files as model input corpus.

    Args:
        directory: Processed corpus directory.

    Returns:
        Mapping normalized id -> text.

    Raises:
        ValueError: If corpus is empty.
    """
    docs: Dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        doc_id = normalize_id(path.stem)
        docs[doc_id] = text

    if not docs:
        raise ValueError(f"Empty processed corpus: {directory}")
    return docs


def _save_dataset_result(dataset_results_dir: Path, model_name: str, metrics: Dict[str, float | List[float]]) -> Path:
    """Save per-model metric JSON in dataset folder.

    Args:
        dataset_results_dir: Dataset-specific results directory.
        model_name: Model name.
        metrics: Metric dictionary.

    Returns:
        Path to JSON artifact.
    """
    dataset_results_dir.mkdir(parents=True, exist_ok=True)
    out_file = dataset_results_dir / f"{model_name}_results.json"
    payload = {key: value for key, value in metrics.items() if key != "AP_scores"}
    out_file.write_text(json.dumps(payload, indent=4), encoding="utf-8")
    return out_file


def run_pipeline(dataset: str, models: List[str], top_k: int = DEFAULT_TOP_K) -> None:
    """Execute retrieval, ranking export, evaluation, and statistics.

    Args:
        dataset: Dataset name.
        models: Model list.
        top_k: Top-k ranking export cutoff.
    """
    setup_logging()
    set_reproducible_seed()

    dataset_key = normalize_dataset_name(dataset)
    dataset_paths = get_dataset_paths(dataset_key)
    validate_dataset_paths(dataset_paths)

    ensure_dirs(OUTPUT_DIRECTORIES.values())
    (dataset_paths["results"] / "rankings").mkdir(parents=True, exist_ok=True)

    req_docs = load_processed_docs(dataset_paths["req_processed"])
    code_docs = load_processed_docs(dataset_paths["code_processed"])
    logger.info("Requirement count: %d", len(req_docs))
    logger.info("Code file count: %d", len(code_docs))

    ground_truth = load_ground_truth(dataset_paths["ground_truth"])
    validate_ground_truth_alignment(ground_truth, req_docs.keys(), code_docs.keys())

    model_runners: Dict[str, Callable] = {
        "tfidf": run_tfidf,
        "bm25": run_bm25,
        "sbert": run_sbert,
        "codebert": run_codebert,
    }

    selected_models = [model.lower() for model in models]
    for model in selected_models:
        if model not in model_runners:
            raise ValueError(f"Unsupported model: {model}. Supported: {', '.join(MODEL_NAMES)}")

    all_metrics: Dict[str, Dict[str, float | List[float]]] = {}

    for model in selected_models:
        rankings, similarity_matrix, req_ids, code_ids = model_runners[model](req_docs, code_docs)

        if not rankings:
            raise ValueError(f"Ranking outputs are empty for model: {model}")

        export_rankings(dataset_paths["results"] / "rankings", model, similarity_matrix, req_ids, code_ids, top_k)
        export_rankings(OUTPUT_DIRECTORIES["rankings"], f"{dataset_key}_{model}", similarity_matrix, req_ids, code_ids, top_k)

        metrics = evaluate_rankings(rankings, ground_truth)
        all_metrics[model] = metrics

        _save_dataset_result(dataset_paths["results"], model, metrics)
        logger.info(
            "Metric output | %s | MAP=%.4f MRR=%.4f Recall@5=%.4f Recall@10=%.4f",
            model,
            metrics["MAP"],
            metrics["MRR"],
            metrics["Recall@5"],
            metrics["Recall@10"],
        )

    metrics_rows = []
    for model_name, metrics in all_metrics.items():
        metrics_rows.append(
            {
                "dataset": dataset_key,
                "model": model_name,
                "MAP": metrics["MAP"],
                "MRR": metrics["MRR"],
                "Recall@5": metrics["Recall@5"],
                "Recall@10": metrics["Recall@10"],
            }
        )

    dataset_metrics_file = dataset_paths["results"] / "metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(dataset_metrics_file, index=False)

    global_metrics_file = OUTPUT_DIRECTORIES["metrics"] / f"{dataset_key}_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(global_metrics_file, index=False)

    tests = []
    pairs = [("sbert", "bm25"), ("codebert", "sbert"), ("tfidf", "bm25")]
    for a_name, b_name in pairs:
        if a_name in all_metrics and b_name in all_metrics:
            tests.append(
                paired_ttest(
                    all_metrics[a_name]["AP_scores"],
                    all_metrics[b_name]["AP_scores"],
                    a_name,
                    b_name,
                )
            )

    if tests:
        dataset_stats_file = save_statistics(dataset_paths["results"], dataset_key, tests)
        global_stats_file = save_statistics(OUTPUT_DIRECTORIES["statistics"], dataset_key, tests)
        logger.info("Statistical testing output saved: %s", dataset_stats_file)
        logger.info("Statistical testing output saved: %s", global_stats_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval pipeline for one dataset")
    parser.add_argument("--dataset", required=True, help="Dataset name: itrust, etour, eanci, smos")
    parser.add_argument("--models", nargs="+", default=MODEL_NAMES, help="Models to run")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-k ranking cutoff")
    args = parser.parse_args()

    run_pipeline(args.dataset, args.models, args.top_k)
