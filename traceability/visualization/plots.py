"""Plot generation for cross-dataset and cross-model metric comparisons."""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from traceability.utils.file_utils import ensure_dir
from traceability.utils.logging_utils import get_logger

logger = get_logger(__name__)


METRIC_COLUMNS = ["MAP", "MRR", "Recall@5", "Recall@10"]


def _build_master_dataframe(results_root: Path) -> pd.DataFrame:
    """Build a master metric table from per-dataset metric CSV files.

    Args:
        results_root: Project-level results root.

    Returns:
        Master DataFrame with Dataset, Model, and metric columns.

    Raises:
        ValueError: If no metric rows are available.
    """
    metrics_dir = results_root / "metrics"
    rows: List[dict] = []

    for metric_file in sorted(metrics_dir.glob("*_metrics.csv")):
        dataset_name = metric_file.stem.replace("_metrics", "")
        frame = pd.read_csv(metric_file)
        for _, row in frame.iterrows():
            rows.append(
                {
                    "Dataset": dataset_name,
                    "Model": row["model"],
                    "MAP": row["MAP"],
                    "MRR": row["MRR"],
                    "Recall@5": row["Recall@5"],
                    "Recall@10": row["Recall@10"],
                }
            )

    if not rows:
        raise ValueError("No metrics found. Run pipeline before generating visualizations.")

    return pd.DataFrame(rows)


def generate_visualizations(results_root: Path, output_dir: Path) -> List[Path]:
    """Generate standard dissertation plots from experiment metrics.

    Args:
        results_root: Project-level results root.
        output_dir: Visualization output directory.

    Returns:
        List of generated artifact paths.
    """
    ensure_dir(output_dir)
    df = _build_master_dataframe(results_root)

    outputs: List[Path] = []

    master_csv = output_dir / "master_results.csv"
    df.to_csv(master_csv, index=False)
    outputs.append(master_csv)

    pivot_map = df.pivot(index="Dataset", columns="Model", values="MAP")
    ax = pivot_map.plot(kind="bar", figsize=(10, 6))
    ax.set_title("MAP Comparison Across Datasets")
    ax.set_ylabel("MAP")
    ax.set_xlabel("Dataset")
    plt.xticks(rotation=0)
    plt.tight_layout()
    map_file = output_dir / "map_comparison.png"
    plt.savefig(map_file, dpi=300)
    plt.close()
    outputs.append(map_file)

    pivot_mrr = df.pivot(index="Dataset", columns="Model", values="MRR")
    ax = pivot_mrr.plot(kind="bar", figsize=(10, 6))
    ax.set_title("MRR Comparison Across Datasets")
    ax.set_ylabel("MRR")
    ax.set_xlabel("Dataset")
    plt.xticks(rotation=0)
    plt.tight_layout()
    mrr_file = output_dir / "mrr_comparison.png"
    plt.savefig(mrr_file, dpi=300)
    plt.close()
    outputs.append(mrr_file)

    pivot_r5 = df.pivot(index="Dataset", columns="Model", values="Recall@5")
    ax = pivot_r5.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Recall@5 Comparison Across Datasets")
    ax.set_ylabel("Recall@5")
    ax.set_xlabel("Dataset")
    plt.xticks(rotation=0)
    plt.tight_layout()
    recall5_file = output_dir / "recall5_comparison.png"
    plt.savefig(recall5_file, dpi=300)
    plt.close()
    outputs.append(recall5_file)

    pivot_r10 = df.pivot(index="Dataset", columns="Model", values="Recall@10")
    ax = pivot_r10.plot(kind="bar", figsize=(10, 6))
    ax.set_title("Recall@10 Comparison Across Datasets")
    ax.set_ylabel("Recall@10")
    ax.set_xlabel("Dataset")
    plt.xticks(rotation=0)
    plt.tight_layout()
    recall10_file = output_dir / "recall10_comparison.png"
    plt.savefig(recall10_file, dpi=300)
    plt.close()
    outputs.append(recall10_file)

    best_models = df.loc[df.groupby("Dataset")["MAP"].idxmax()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(best_models["Dataset"], best_models["MAP"])
    ax.set_title("Best Performing Model Per Dataset")
    ax.set_ylabel("Best MAP Score")
    ax.set_xlabel("Dataset")

    for idx, row in best_models.iterrows():
        x = list(best_models.index).index(idx)
        ax.text(x=x, y=row["MAP"] + 0.005, s=row["Model"], ha="center")

    plt.tight_layout()
    best_file = output_dir / "best_model_per_dataset.png"
    plt.savefig(best_file, dpi=300)
    plt.close()
    outputs.append(best_file)

    logger.info("Generated %d visualization artifacts", len(outputs))
    return outputs
