from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_postprocess import generate_pareto_plots
from xiaorong.experiment_selection import SELECTED_CASE_NAMES


EXPERIMENT_DIR = Path(__file__).resolve().parent


TAG_ALIASES = {
    "rl_cg_moaco": ("rl_cg_moaco", "RL-CG-MOACO"),
    "rl_cg_moaco_no_ddqn": (
        "rl_cg_moaco_no_ddqn",
        "RL-CG-MOACO without DDQN",
    ),
    "rl_cg_moaco_no_block_penalty": (
        "rl_cg_moaco_no_block_penalty",
        "RL-CG-MOACO without BlockPenalty",
    ),
    "rl_cg_moaco_no_final_reward": (
        "rl_cg_moaco_no_final_reward",
        "RL-CG-MOACO without final reward",
    ),
    "rl_cg_moaco_no_search": (
        "rl_cg_moaco_no_search",
        "RL-CG-MOACO without local search",
    ),
    "rl_cg_moaco_no_graph_pheromone": (
        "rl_cg_moaco_no_graph_pheromone",
        "RL-CG-MOACO without graph pheromone",
    ),
}


if __name__ == "__main__":
    generate_pareto_plots(
        input_dir=EXPERIMENT_DIR,
        output_dir=EXPERIMENT_DIR,
        tag_aliases=TAG_ALIASES,
        title="Ablation experiment Pareto fronts",
        case_names=SELECTED_CASE_NAMES,
    )
