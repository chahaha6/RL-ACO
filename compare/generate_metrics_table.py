from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_postprocess import generate_metrics_table


EXPERIMENT_DIR = Path(__file__).resolve().parent


TAG_ALIASES = {
    "cg_moaco": ("cg_moaco", "CG-MOACO"),
    "modbo": ("modbo", "MODBO"),
    "sfmodbo": ("modbo", "MODBO"),
    "mopso": ("mopso", "MOPSO"),
    "spea2": ("spea2", "SPEA2"),
    "moaco": ("moaco", "MOACO"),
    "nsga2": ("nsga2", "NSGA-II"),
}


if __name__ == "__main__":
    generate_metrics_table(
        input_dir=EXPERIMENT_DIR,
        output_dir=EXPERIMENT_DIR,
        tag_aliases=TAG_ALIASES,
        title="Comparison experiment metrics",
    )
