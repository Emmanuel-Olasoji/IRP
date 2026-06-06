"""Generate plots from framework metric outputs."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import OUTPUT_DIRECTORIES
from traceability.visualization.plots import generate_visualizations


if __name__ == "__main__":
    generate_visualizations(OUTPUT_DIRECTORIES["metrics"].parent, OUTPUT_DIRECTORIES["visualizations"])
