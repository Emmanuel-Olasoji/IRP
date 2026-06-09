## Project Aim
This repository implements the MSc IRP project titled **"Automating Requirements-to-Code Traceability Using Embedding-Based NLP Approaches"**. Its aim is to provide a rigorous, reproducible research framework for automatically linking natural-language requirements to source-code artefacts using both lexical and embedding-based retrieval models.

The framework evaluates four retrieval models across four benchmark datasets and produces reproducible outputs for rankings, evaluation metrics, statistical testing, error analysis, and visualisations.

## Datasets
- iTrust
- eTour
- eANCI
- SMOS

## Retrieval Models
- TF-IDF
- BM25
- SBERT (`all-MiniLM-L6-v2`)
- CodeBERT (`microsoft/codebert-base`)

## Evaluation Metrics
- MAP
- MRR
- Recall@5
- Recall@10

## Framework Design
The implementation is modular and organized by research activity:
- `traceability/preprocessing`: requirements/code preprocessing and ground-truth normalization
- `traceability/retrieval`: model-specific ranking generation
- `traceability/evaluation`: metrics, ranking export, statistical testing
- `traceability/analysis`: post-hoc error analysis
- `traceability/visualization`: publication-style plots
- `config`: global settings and dataset mapping

## Folder Structure
```text
MSC-RESEARCH/
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── dataset_config.py
├── traceability/
│   ├── __init__.py
│   ├── preprocessing/
│   ├── retrieval/
│   ├── evaluation/
│   ├── analysis/
│   ├── visualization/
│   └── utils/
├── scripts/
│   ├── run_preprocessing.py
│   ├── run_pipeline.py
│   ├── run_error_analysis.py
│   └── generate_visualizations.py
├── datasets/
├── results/
│   ├── rankings/
│   ├── metrics/
│   ├── statistics/
│   ├── error_analysis/
│   └── visualizations/
├── requirements.txt
└── run_all.py
```

## Setup
1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start (End-to-End)
To execute the complete experimental workflow across all datasets, run:

```bash
python3 run_all.py
```

This single command runs preprocessing, retrieval, evaluation, statistical testing, error analysis, and visualization generation for iTrust, eTour, eANCI, and SMOS.

## Execution Commands
Run from project root (`MSC-RESEARCH`):

```bash
python3 scripts/run_preprocessing.py --dataset smos
python3 scripts/run_pipeline.py --dataset smos --models tfidf bm25 sbert codebert
python3 scripts/run_error_analysis.py --dataset smos
python3 scripts/generate_visualizations.py
python3 run_all.py
```

## End-to-End Flow
`run_all.py` executes the following sequence for each dataset:
1. Validate dataset structure and required files.
2. Preprocess requirements and code corpora.
3. Run all configured retrieval models.
4. Export rankings and compute MAP, MRR, Recall@5, Recall@10.
5. Run paired t-tests and export statistical outputs.
6. Run error analysis over ranking outputs.
7. Generate consolidated result visualizations after all datasets finish.

Note: First-time SBERT/CodeBERT execution may download pretrained model files, so the first run can take longer.

## Output Artifacts
### Dataset-local artifacts (`datasets/<dataset>/results/`)
- `<model>_results.json`: per-model metric summary
- `rankings/<model>_rankings.csv`: top-k ranked links
- `error_analysis/*.csv`: full and sliced error-analysis reports
- `<dataset>_paired_ttests.csv`: paired t-tests on AP vectors

### Project-level artifacts (`results/`)
- `metrics/<dataset>_metrics.csv`: normalized metric tables used for cross-dataset comparison
- `statistics/<dataset>_paired_ttests.csv`: global stats export
- `rankings/<dataset>_<model>_rankings.csv`: global ranking snapshots
- `visualizations/*.png`: dissertation figures
- `visualizations/master_results.csv`: consolidated plotting table

## Reproducibility Notes
- Deterministic seed is centralized in `config/settings.py`.
- Dataset path validation is enforced before preprocessing/pipeline execution.
- Ground-truth ID alignment is validated against processed requirement and code IDs.
- Metric computation fails explicitly if zero valid queries are available.
- Ranking export fails explicitly if no rows are produced.
- All pipeline stages use structured logging (`info`, `warning`, `error`) for auditability.

## Dissertation Quality Considerations
This framework emphasizes:
- transparent computational workflow
- reproducible experiment execution
- modular software engineering design
- maintainable code organization
- explicit validation and failure handling
- clean separation between preprocessing, retrieval, evaluation, and analysis

## Legacy Script Compatibility
Original script names are retained in `scripts/` as wrappers, so prior invocation styles remain usable while delegating to the new modular framework.
