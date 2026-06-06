"""Post-hoc error analysis for ranking outputs against trace links."""

from pathlib import Path
from typing import Dict, List

import pandas as pd

from traceability.utils.file_utils import ensure_dir
from traceability.utils.logging_utils import get_logger
from traceability.utils.text_utils import normalize_id

logger = get_logger(__name__)


def _normalize_requirement_id(value: str) -> str:
    return normalize_id(value)


def _normalize_code_id(value: str) -> str:
    return normalize_id(value)


def analyze_model_rankings(
    model_name: str,
    rankings_file: Path,
    ground_truth_df: pd.DataFrame,
    output_dir: Path,
    top_k: int,
) -> Dict[str, Path]:
    """Analyze one model ranking file.

    Args:
        model_name: Model identifier.
        rankings_file: Ranking CSV path.
        ground_truth_df: Normalized ground-truth DataFrame.
        output_dir: Output folder for analysis CSVs.
        top_k: Top-k threshold used in evaluation.

    Returns:
        Mapping of generated artifact names to paths.
    """
    rankings = pd.read_csv(rankings_file)
    rankings.columns = [column.strip().lower() for column in rankings.columns]

    rankings["requirement_id_norm"] = rankings["requirement_id"].apply(_normalize_requirement_id)
    rankings["code_file_norm"] = rankings["code_file"].apply(_normalize_code_id)

    analysis_rows: List[dict] = []
    for _, gt_row in ground_truth_df.iterrows():
        req_id = str(gt_row["requirement_id"]).strip()
        req_id_norm = str(gt_row["requirement_id_norm"]).strip()
        code_id = str(gt_row["code_id"]).strip()
        code_id_norm = str(gt_row["code_id_norm"]).strip()

        req_rankings = rankings[rankings["requirement_id_norm"] == req_id_norm]

        found = False
        found_rank = None
        top_prediction = None
        top_score = None

        if len(req_rankings) > 0:
            top_prediction = req_rankings.iloc[0]["code_file"]
            top_score = req_rankings.iloc[0]["score"]

        for _, row in req_rankings.iterrows():
            predicted_file = str(row["code_file_norm"]).strip()
            if predicted_file == code_id_norm:
                found = True
                found_rank = int(row["rank"])
                break

        analysis_rows.append(
            {
                "requirement_id": req_id,
                "requirement_id_norm": req_id_norm,
                "correct_code_file": code_id,
                "correct_code_file_norm": code_id_norm,
                "found": found,
                "rank_found": found_rank,
                "top_prediction": top_prediction,
                "top_score": top_score,
                "retrieved_in_top_k": found_rank is not None and found_rank <= top_k,
            }
        )

    analysis_df = pd.DataFrame(analysis_rows)
    outputs: Dict[str, Path] = {}

    outputs["full"] = output_dir / f"{model_name}_full_analysis.csv"
    analysis_df.to_csv(outputs["full"], index=False)

    outputs["best"] = output_dir / f"{model_name}_best_cases.csv"
    analysis_df[analysis_df["rank_found"] == 1].to_csv(outputs["best"], index=False)

    outputs["worst"] = output_dir / f"{model_name}_worst_cases.csv"
    analysis_df[analysis_df["found"] == False].to_csv(outputs["worst"], index=False)

    outputs["topk_failures"] = output_dir / f"{model_name}_topk_failures.csv"
    analysis_df[(analysis_df["found"] == True) & (analysis_df["rank_found"] > top_k)].to_csv(
        outputs["topk_failures"], index=False
    )

    repeated_false_positives = analysis_df[(analysis_df["found"] == False) & (analysis_df["top_prediction"].notna())]
    repeated_false_positives = repeated_false_positives[
        repeated_false_positives["top_prediction"].astype(str).str.strip() != ""
    ]
    repeated_false_positives = (
        repeated_false_positives.groupby("top_prediction", as_index=False)
        .size()
        .rename(columns={"size": "times_as_false_positive_top1"})
        .sort_values("times_as_false_positive_top1", ascending=False)
    )

    outputs["repeated_false_positives"] = output_dir / f"{model_name}_repeated_false_positives.csv"
    repeated_false_positives.to_csv(outputs["repeated_false_positives"], index=False)

    requirement_difficulty = (
        analysis_df.groupby("requirement_id", as_index=False)
        .agg(
            total_links=("found", "size"),
            found_links=("found", "sum"),
            top1_hits=("rank_found", lambda series: (series == 1).sum()),
            top5_hits=("rank_found", lambda series: (series <= 5).sum()),
            top10_hits=("rank_found", lambda series: (series <= 10).sum()),
            best_rank=("rank_found", "min"),
            avg_rank_found=("rank_found", "mean"),
        )
    )

    requirement_difficulty["found_rate"] = (
        requirement_difficulty["found_links"] / requirement_difficulty["total_links"]
    ).round(4)
    requirement_difficulty["difficulty_label"] = "mixed"
    requirement_difficulty.loc[requirement_difficulty["found_rate"] == 0, "difficulty_label"] = "hard"
    requirement_difficulty.loc[
        (requirement_difficulty["found_rate"] == 1) & (requirement_difficulty["best_rank"] == 1),
        "difficulty_label",
    ] = "easy"

    outputs["difficulty"] = output_dir / f"{model_name}_requirement_difficulty.csv"
    requirement_difficulty.to_csv(outputs["difficulty"], index=False)

    outputs["easy"] = output_dir / f"{model_name}_easy_requirements.csv"
    requirement_difficulty[requirement_difficulty["difficulty_label"] == "easy"].sort_values(
        ["top1_hits", "total_links"], ascending=False
    ).to_csv(outputs["easy"], index=False)

    outputs["hard"] = output_dir / f"{model_name}_hard_requirements.csv"
    requirement_difficulty[requirement_difficulty["difficulty_label"] == "hard"].sort_values(
        ["total_links", "requirement_id"], ascending=[False, True]
    ).to_csv(outputs["hard"], index=False)

    total = len(analysis_df)
    top1 = len(analysis_df[analysis_df["rank_found"] == 1])
    top5 = len(analysis_df[analysis_df["rank_found"] <= 5])
    top10 = len(analysis_df[analysis_df["rank_found"] <= 10])
    found_total = int(analysis_df["found"].sum())

    summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "total_links": total,
                "found_links": found_total,
                "top1_hits": top1,
                "top5_hits": top5,
                "top10_hits": top10,
                "top1_rate": round(top1 / total, 4) if total else 0,
                "top5_rate": round(top5 / total, 4) if total else 0,
                "top10_rate": round(top10 / total, 4) if total else 0,
            }
        ]
    )
    outputs["summary"] = output_dir / f"{model_name}_summary.csv"
    summary.to_csv(outputs["summary"], index=False)

    return outputs


def run_error_analysis(dataset_root: Path, models: List[str], top_k: int = 10) -> Dict[str, Dict[str, Path]]:
    """Run full error analysis over selected models.

    Args:
        dataset_root: Dataset root path.
        models: Model names to analyze.
        top_k: Top-k threshold.

    Returns:
        Nested mapping model -> artifact key -> path.
    """
    rankings_dir = dataset_root / "results" / "rankings"
    ground_truth_file = dataset_root / "traces" / "ground_tracelinks.csv"
    output_dir = dataset_root / "results" / "error_analysis"
    ensure_dir(output_dir)

    ground_truth = pd.read_csv(ground_truth_file)
    ground_truth.columns = [column.strip().lower() for column in ground_truth.columns]

    if "code_id" not in ground_truth.columns and "code_file" in ground_truth.columns:
        ground_truth["code_id"] = ground_truth["code_file"]

    ground_truth["requirement_id_norm"] = ground_truth["requirement_id"].apply(_normalize_requirement_id)
    ground_truth["code_id_norm"] = ground_truth["code_id"].apply(_normalize_code_id)
    ground_truth = ground_truth[ground_truth["requirement_id_norm"].astype(str).str.strip() != ""].copy()

    logger.info("Trace link count: %d", len(ground_truth))

    results: Dict[str, Dict[str, Path]] = {}
    for model_name in models:
        rankings_file = rankings_dir / f"{model_name}_rankings.csv"
        if not rankings_file.exists():
            logger.warning("Skipping missing ranking file: %s", rankings_file)
            continue
        results[model_name] = analyze_model_rankings(model_name, rankings_file, ground_truth, output_dir, top_k)

    return results
