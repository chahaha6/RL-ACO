from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_postprocess import generate_metrics_table


EXPERIMENT_DIR = Path(__file__).resolve().parent


TAG_ALIASES = {
    "cg_moaco": ("cg_moaco", "CG-MOACO"),
    "cg_moaco_no_gouzao": ("cg_moaco_no_gouzao", "CG-MOACO-NO-GOUZAO"),
    "cg_moaco_no_message": ("cg_moaco_no_message", "CG-MOACO-NO-MESSAGE"),
    "cg_moaco_insert": ("cg_moaco_insert", "CG-MOACO-INSERT"),
    "cg_moaco_no_search": ("cg_moaco_no_search", "CG-MOACO-NO-SEARCH"),
    "cg_moaco_tihuan": ("cg_moaco_tihuan", "CG-MOACO-TIHUAN"),
}


if __name__ == "__main__":
    generate_metrics_table(
        input_dir=EXPERIMENT_DIR,
        output_dir=EXPERIMENT_DIR,
        tag_aliases=TAG_ALIASES,
        title="Ablation experiment metrics",
    )
