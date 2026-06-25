from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_postprocess import generate_pareto_plots


EXPERIMENT_DIR = Path(__file__).resolve().parent


TAG_ALIASES = {
    "cg_moaco": ("cg_moaco", "CG-MOACO"),
    "param_lambda_conflict_0_5": ("param_lambda_conflict_0_5", "lambda_conflict=0.5"),
    "param_lambda_conflict_2_0": ("param_lambda_conflict_2_0", "lambda_conflict=2.0"),
    "param_rho_0_05": ("param_rho_0_05", "rho=0.05"),
    "param_replacement_attempts_60": (
        "param_replacement_attempts_60",
        "replacement_attempts=60",
    ),
}


if __name__ == "__main__":
    generate_pareto_plots(
        input_dir=EXPERIMENT_DIR,
        output_dir=EXPERIMENT_DIR,
        tag_aliases=TAG_ALIASES,
        title="Parameter analysis Pareto fronts",
    )
