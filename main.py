from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

from modules.cg_moaco import CGMOACO, DEFAULT_PARAMS as CG_DEFAULT_PARAMS
from modules.modbo import MODBO, DEFAULT_PARAMS as MODBO_DEFAULT_PARAMS
from modules.moaco import MOACO, DEFAULT_PARAMS as MOACO_DEFAULT_PARAMS
from modules.mopso import MOPSO, DEFAULT_PARAMS as MOPSO_DEFAULT_PARAMS
from modules.nsga2 import NSGA2, DEFAULT_PARAMS as NSGA2_DEFAULT_PARAMS
from modules.rl_cg_moaco import RLCGMOACO, DEFAULT_PARAMS as RL_CG_DEFAULT_PARAMS
from modules.rl_cg_moaco_no_block_penalty import (
    RLCGMOACO_NO_BLOCK_PENALTY,
    DEFAULT_PARAMS as RL_CG_NO_BLOCK_DEFAULT_PARAMS,
)
from modules.rl_cg_moaco_no_ddqn import (
    RLCGMOACO_NO_DDQN,
    DEFAULT_PARAMS as RL_CG_NO_DDQN_DEFAULT_PARAMS,
)
from modules.rl_cg_moaco_no_final_reward import (
    RLCGMOACO_NO_FINAL_REWARD,
    DEFAULT_PARAMS as RL_CG_NO_FINAL_DEFAULT_PARAMS,
)
from modules.rl_cg_moaco_no_graph_pheromone import (
    RLCGMOACO_NO_GRAPH_PHEROMONE,
    DEFAULT_PARAMS as RL_CG_NO_GRAPH_PHEROMONE_DEFAULT_PARAMS,
)
from modules.rl_cg_moaco_no_search import (
    RLCGMOACO_NO_SEARCH,
    DEFAULT_PARAMS as RL_CG_NO_SEARCH_DEFAULT_PARAMS,
)
from modules.spea2 import SPEA2, DEFAULT_PARAMS as SPEA2_DEFAULT_PARAMS
from modules.conflict_graph import build_conflict_graph, compute_graph_features
from modules.domain import CandidateNode, Task
from modules.problem_model import DEFAULT_MODEL_PARAMS, task_completion_rate
from modules.utils import Solution


# =========================================================
# 数据集选择：按“卫星数、任务数”选择问题
# 只跑一个数据集就写成 [(5, 100)]。
# =========================================================
DEFAULT_DATASET_PREFIX = "area"

CASE_LIST = [
    (5, 100),
    # (5, 300),
    # (5, 500),
    # (5, 1000),
]

# ---------- 对比实验 ----------
RUN_COMPARISON_EXPERIMENTS = True

RUN_RL_CG_MOACO = True
RUN_CG_MOACO = False
RUN_MODBO = False
RUN_MOPSO = False
RUN_SPEA2 = False
RUN_MOACO = False
RUN_NSGA2 = False

# ---------- 消融实验 ----------
# Original CG-MOACO mechanism ablations.
RUN_RL_ABLATION_EXPERIMENTS = False
RUN_RL_CG_MOACO_NO_DDQN = False
RUN_RL_CG_MOACO_NO_BLOCK_PENALTY = False
RUN_RL_CG_MOACO_NO_FINAL_REWARD = False
RUN_RL_CG_MOACO_NO_SEARCH = False
RUN_RL_CG_MOACO_NO_GRAPH_PHEROMONE = False

# ---------- 参数分析 ----------
RUN_PARAMETER_ANALYSIS = False

# 通用实验参数
#   RUN_COUNT：独立运行轮数
#   ITERATION_COUNT：每轮迭代代数
# =========================================================
RUN_COUNT = 10
ITERATION_COUNT = 100
POP_SIZE = 50
ARCHIVE_SIZE = 100
RANDOM_SEED_BASE = 2026
VERBOSE = True

# 所有算法共用的问题模型参数。
# 修改这些值后，冲突图构建和各算法目标评价会同步更新。
MODEL_PARAMS = {
    **DEFAULT_MODEL_PARAMS,
}

# =========================================================
# 结果保存设置
# =========================================================
RESULTS_ROOT = "results"
CSV_DATA_ROOT = Path(__file__).resolve().parent / "CSV_DATA"

# True：同名 summary/archive CSV 会被覆盖。
# False：如果结果 CSV 已存在，则跳过该实验。
OVERWRITE_RESULT_CSV = True


# =========================================================
# CG-MOACO 专属参数
#
# 这些参数是你算法创新机制相关参数：
#   1. 冲突图启发式构造机制
#   2. 冲突图感知 Pareto 信息素更新
#   3. 冲突图感知快速插入-替换局部搜索

CG_EXTRA_PARAMS = {
    # ---------- 冲突图启发式函数权重 ----------
    # lambda_scarcity：窗口稀缺度权重
    # lambda_conflict：冲突度惩罚权重
    # lambda_maneuver：姿态机动代价惩罚权重
    # lambda_load：卫星负载惩罚权重
    "lambda_scarcity": 1.0,
    "lambda_conflict": 1.0,
    "lambda_maneuver": 1.0,
    "lambda_load": 1.0,

    # ---------- 局部搜索开关 ----------
    # enable_local_search：是否启用冲突图感知局部搜索
    # enable_fast_insert：是否启用快速插入操作
    # enable_replacement：是否启用替换操作
    "enable_local_search": True,
    "enable_fast_insert": True,
    "enable_replacement": True,

    # ---------- 信息素更新开关 ----------
    # use_graph_pheromone：是否使用冲突图贡献度引导信息素更新
    "use_graph_pheromone": True,

    # ---------- 替换局部搜索参数 ----------
    # replacement_attempts：每个解最多尝试替换的次数
    # max_replace_conflicts：一次替换允许涉及的最大冲突节点数
    # min_replacement_profit_gain：执行替换所需的最小收益增益
    "replacement_attempts": 30,
    "max_replace_conflicts": 3,
    "min_replacement_profit_gain": 2.0,

    # ---------- 多目标替换增益权重 ----------
    # w_profit：收益改善权重
    # w_maneuver：姿态机动代价变化权重
    # w_load：负载均衡变化权重
    "w_profit": 1.0,
    "w_maneuver": 0.01,
    "w_load": 1.0,
}


# =========================================================
# 参数分析配置
# =========================================================

PARAMETER_ANALYSIS_CONFIGS = [
    (
        "param_lambda_conflict_0_5",
        "CG-MOACO lambda_conflict=0.5",
        {"lambda_conflict": 0.5},
    ),
    (
        "param_lambda_conflict_2_0",
        "CG-MOACO lambda_conflict=2.0",
        {"lambda_conflict": 2.0},
    ),
    (
        "param_rho_0_05",
        "CG-MOACO rho=0.05",
        {"rho": 0.05},
    ),
    (
        "param_replacement_attempts_60",
        "CG-MOACO replacement_attempts=60",
        {"replacement_attempts": 60},
    ),
]


@dataclass(frozen=True)
class ExperimentSpec:
    tag: str
    name: str
    algorithm_key: str
    param_overrides: dict
    group: str


def dataset_name(
    satellite_count: int,
    task_count: int,
    prefix: str = DEFAULT_DATASET_PREFIX,
) -> str:
    """生成 area_s5_t100 形式的数据集目录名。"""

    if satellite_count <= 0:
        raise ValueError("satellite_count must be positive")
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    return f"{prefix}_s{satellite_count}_t{task_count}"


def read_tasks_csv(file_path: str | Path) -> dict[int, Task]:
    """读取前处理生成的 tasks.csv。"""

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing preprocessed task file: {path}")

    tasks: dict[int, Task] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task_id = int(row["task_id"])
            tasks[task_id] = Task(
                task_id=task_id,
                coord_1=float(row["coord_1"]),
                coord_2=float(row["coord_2"]),
                task_type=int(row["task_type"]),
                duration=float(row["duration"]),
                profit=float(row["profit"]),
            )
    return tasks


def read_candidate_nodes_csv(file_path: str | Path) -> list[CandidateNode]:
    """读取前处理生成的 candidate_nodes.csv。"""

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing preprocessed candidate file: {path}")

    nodes: list[CandidateNode] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            node = CandidateNode(
                node_id=int(row["node_id"]),
                task_id=int(row["task_id"]),
                sat_id=int(row["sat_id"]),
                window_id=int(row["window_id"]),
                start=float(row["start"]),
                end=float(row["end"]),
                duration=float(row["duration"]),
                profit=float(row["profit"]),
                task_duration=float(row["task_duration"]),
                coord_1=float(row["coord_1"]),
                coord_2=float(row["coord_2"]),
            )
            if node.duration + 1e-9 < node.task_duration:
                raise ValueError(
                    f"Invalid candidate node {node.node_id} in {path}: "
                    f"window duration {node.duration} < task duration {node.task_duration}. "
                    "Regenerate CSV files with modules/data_process.py."
                )
            nodes.append(node)
    return nodes


def budget_label() -> str:
    """返回实验预算标签，例如 10轮100代。"""

    return f"{RUN_COUNT}轮{ITERATION_COUNT}代"


def algo_label(algo_tag: str) -> str:
    """统一算法显示名称。"""

    labels = {
        "rl_cg_moaco": "RL-CG-MOACO",
        "cg_moaco": "CG-MOACO",
        "modbo": "MODBO",
        "mopso": "MOPSO",
        "spea2": "SPEA2",
        "moaco": "MOACO",
        "nsga2": "NSGA-II",
    }
    return labels.get(algo_tag.lower(), algo_tag.upper())


def result_file_stem(algo_tag: str, task_count: int) -> str:
    """生成结果文件名前缀。"""

    return f"{algo_tag}_t{task_count}"


def result_files(results_dir: Path, algo_tag: str, task_count: int) -> tuple[Path, Path]:
    """返回某算法某数据集的 summary 文件和 Pareto 档案文件。"""

    stem = result_file_stem(algo_tag, task_count)
    summary_file = results_dir / f"{stem}_summary.csv"
    archive_file = results_dir / f"{stem}_archive.csv"
    return summary_file, archive_file


def safe_mean(values: list[float]) -> float | str:
    """安全计算均值。"""

    clean_values = [float(v) for v in values if v != "" and v is not None]
    if not clean_values:
        return ""
    return mean(clean_values)


def safe_std(values: list[float]) -> float | str:
    """安全计算标准差。"""

    clean_values = [float(v) for v in values if v != "" and v is not None]
    if not clean_values:
        return ""
    return pstdev(clean_values)


def build_problem_interface(
    satellite_count: int,
    task_count: int,
    *,
    prefix: str = DEFAULT_DATASET_PREFIX,
    csv_data_root: str | Path = CSV_DATA_ROOT,
) -> dict:
    csv_dataset_dir = Path(csv_data_root) / dataset_name(
        satellite_count,
        task_count,
        prefix,
    )
    if not csv_dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Preprocessed CSV directory not found: {csv_dataset_dir}. "
            "Run modules/data_process.py first."
        )

    tasks = read_tasks_csv(csv_dataset_dir / "tasks.csv")
    nodes = read_candidate_nodes_csv(csv_dataset_dir / "candidate_nodes.csv")

    # 构建三元冲突图
    #
    # 注意：
    #   冲突图属于问题结构，不属于某个算法私有模块。
    #   CG-MOACO 会深度使用冲突图；
    #   MOACO 只用冲突图做普通可行性判断。
    #
    # 冲突图使用统一的问题模型参数；图特征额外使用 CG-MOACO 的
    # 冲突图启发式权重。
    problem_params = {
        **MODEL_PARAMS,
        **CG_EXTRA_PARAMS,
    }

    conflict_adj = build_conflict_graph(nodes, problem_params)
    graph_features = compute_graph_features(nodes, conflict_adj, problem_params)

    return {
        "case_name": dataset_name(satellite_count, task_count, prefix),
        "satellite_count": satellite_count,
        "task_count": task_count,
        "csv_dataset_dir": csv_dataset_dir,
        "tasks": tasks,
        "nodes": nodes,
        "conflict_adj": conflict_adj,
        "graph_features": graph_features,
        "satellite_ids": list(range(1, satellite_count + 1)),
        "model_params": dict(MODEL_PARAMS),
    }


def summarize_archive(
    archive: list[Solution],
    nodes_by_id: dict,
    total_task_count: int,
) -> dict:
    """对单次运行得到的 Pareto 档案生成统计结果。"""

    if not archive:
        return {
            "archive_size": 0,
            "best_f1": "",
            "best_f2": "",
            "best_f3": "",
            "best_completion_rate": "",
        }

    best_f1 = min(sol.objectives[0] for sol in archive)
    best_f2 = min(sol.objectives[1] for sol in archive)
    best_f3 = min(sol.objectives[2] for sol in archive)

    best_completion_rate = max(
        task_completion_rate(sol.node_ids, total_task_count, nodes_by_id)
        for sol in archive
    )

    return {
        "archive_size": len(archive),
        "best_f1": best_f1,
        "best_f2": best_f2,
        "best_f3": best_f3,
        "best_completion_rate": best_completion_rate,
    }


def enabled_experiments() -> list[ExperimentSpec]:
    """根据顶部开关生成本次需要运行的实验列表。"""

    experiments: list[ExperimentSpec] = []

    if RUN_COMPARISON_EXPERIMENTS:
        if RUN_RL_CG_MOACO:
            experiments.append(
                ExperimentSpec("rl_cg_moaco", "RL-CG-MOACO", "rl_cg_moaco", {}, "comparison")
            )
        if RUN_CG_MOACO:
            experiments.append(
                ExperimentSpec("cg_moaco", "CG-MOACO", "cg_moaco", {}, "comparison")
            )
        if RUN_MODBO:
            experiments.append(
                ExperimentSpec("modbo", "MODBO", "modbo", {}, "comparison")
            )
        if RUN_MOPSO:
            experiments.append(
                ExperimentSpec("mopso", "MOPSO", "mopso", {}, "comparison")
            )
        if RUN_SPEA2:
            experiments.append(
                ExperimentSpec("spea2", "SPEA2", "spea2", {}, "comparison")
            )
        if RUN_MOACO:
            experiments.append(
                ExperimentSpec("moaco", "MOACO", "moaco", {}, "comparison")
            )
        if RUN_NSGA2:
            experiments.append(
                ExperimentSpec("nsga2", "NSGA-II", "nsga2", {}, "comparison")
            )

    if RUN_RL_ABLATION_EXPERIMENTS:
        if RUN_RL_CG_MOACO_NO_DDQN:
            experiments.append(
                ExperimentSpec(
                    "rl_cg_moaco_no_ddqn",
                    "RL-CG-MOACO without DDQN",
                    "rl_cg_moaco_no_ddqn",
                    {},
                    "rl_ablation",
                )
            )
        if RUN_RL_CG_MOACO_NO_BLOCK_PENALTY:
            experiments.append(
                ExperimentSpec(
                    "rl_cg_moaco_no_block_penalty",
                    "RL-CG-MOACO without BlockPenalty",
                    "rl_cg_moaco_no_block_penalty",
                    {},
                    "rl_ablation",
                )
            )
        if RUN_RL_CG_MOACO_NO_FINAL_REWARD:
            experiments.append(
                ExperimentSpec(
                    "rl_cg_moaco_no_final_reward",
                    "RL-CG-MOACO without final reward",
                    "rl_cg_moaco_no_final_reward",
                    {},
                    "rl_ablation",
                )
            )
        if RUN_RL_CG_MOACO_NO_SEARCH:
            experiments.append(
                ExperimentSpec(
                    "rl_cg_moaco_no_search",
                    "RL-CG-MOACO without local search",
                    "rl_cg_moaco_no_search",
                    {},
                    "rl_ablation",
                )
            )
        if RUN_RL_CG_MOACO_NO_GRAPH_PHEROMONE:
            experiments.append(
                ExperimentSpec(
                    "rl_cg_moaco_no_graph_pheromone",
                    "RL-CG-MOACO without graph pheromone",
                    "rl_cg_moaco_no_graph_pheromone",
                    {},
                    "rl_ablation",
                )
            )

    if RUN_PARAMETER_ANALYSIS:
        for tag, name, overrides in PARAMETER_ANALYSIS_CONFIGS:
            experiments.append(
                ExperimentSpec(tag, name, "cg_moaco", dict(overrides), "parameter")
            )

    return experiments


def build_algorithm(spec: ExperimentSpec, problem: dict, run_idx: int):
    """根据实验配置创建算法对象。"""

    tasks = problem["tasks"]
    nodes = problem["nodes"]
    conflict_adj = problem["conflict_adj"]
    graph_features = problem["graph_features"]
    satellite_ids = problem["satellite_ids"]
    model_params = problem["model_params"]

    seed = RANDOM_SEED_BASE + run_idx
    common_params = {
        "max_iter": ITERATION_COUNT,
        "archive_size": ARCHIVE_SIZE,
        "seed": seed,
        "verbose": VERBOSE,
    }

    if spec.algorithm_key == "rl_cg_moaco":
        params = {
            **RL_CG_DEFAULT_PARAMS,
            **CG_EXTRA_PARAMS,
            **model_params,
            **common_params,
            "num_ants": POP_SIZE,
            **spec.param_overrides,
        }
        return RLCGMOACO(
            tasks=tasks,
            nodes=nodes,
            conflict_adj=conflict_adj,
            graph_features=graph_features,
            satellite_ids=satellite_ids,
            params=params,
        )

    if spec.algorithm_key == "rl_cg_moaco_no_ddqn":
        params = {
            **RL_CG_NO_DDQN_DEFAULT_PARAMS,
            **CG_EXTRA_PARAMS,
            **model_params,
            **common_params,
            "num_ants": POP_SIZE,
            **spec.param_overrides,
        }
        return RLCGMOACO_NO_DDQN(
            tasks=tasks,
            nodes=nodes,
            conflict_adj=conflict_adj,
            graph_features=graph_features,
            satellite_ids=satellite_ids,
            params=params,
        )

    if spec.algorithm_key == "rl_cg_moaco_no_block_penalty":
        params = {
            **RL_CG_NO_BLOCK_DEFAULT_PARAMS,
            **CG_EXTRA_PARAMS,
            **model_params,
            **common_params,
            "num_ants": POP_SIZE,
            **spec.param_overrides,
        }
        return RLCGMOACO_NO_BLOCK_PENALTY(
            tasks=tasks,
            nodes=nodes,
            conflict_adj=conflict_adj,
            graph_features=graph_features,
            satellite_ids=satellite_ids,
            params=params,
        )

    if spec.algorithm_key == "rl_cg_moaco_no_final_reward":
        params = {
            **RL_CG_NO_FINAL_DEFAULT_PARAMS,
            **CG_EXTRA_PARAMS,
            **model_params,
            **common_params,
            "num_ants": POP_SIZE,
            **spec.param_overrides,
        }
        return RLCGMOACO_NO_FINAL_REWARD(
            tasks=tasks,
            nodes=nodes,
            conflict_adj=conflict_adj,
            graph_features=graph_features,
            satellite_ids=satellite_ids,
            params=params,
        )

    if spec.algorithm_key == "rl_cg_moaco_no_search":
        params = {
            **RL_CG_NO_SEARCH_DEFAULT_PARAMS,
            **CG_EXTRA_PARAMS,
            **model_params,
            **common_params,
            "num_ants": POP_SIZE,
            **spec.param_overrides,
        }
        return RLCGMOACO_NO_SEARCH(
            tasks=tasks,
            nodes=nodes,
            conflict_adj=conflict_adj,
            graph_features=graph_features,
            satellite_ids=satellite_ids,
            params=params,
        )

    if spec.algorithm_key == "rl_cg_moaco_no_graph_pheromone":
        params = {
            **RL_CG_NO_GRAPH_PHEROMONE_DEFAULT_PARAMS,
            **CG_EXTRA_PARAMS,
            **model_params,
            **common_params,
            "num_ants": POP_SIZE,
            **spec.param_overrides,
        }
        return RLCGMOACO_NO_GRAPH_PHEROMONE(
            tasks=tasks,
            nodes=nodes,
            conflict_adj=conflict_adj,
            graph_features=graph_features,
            satellite_ids=satellite_ids,
            params=params,
        )

    if spec.algorithm_key == "cg_moaco":
        params = {
            **CG_DEFAULT_PARAMS,
            **CG_EXTRA_PARAMS,
            **model_params,
            **common_params,
            "num_ants": POP_SIZE,
            **spec.param_overrides,
        }
        return CGMOACO(
            tasks=tasks,
            nodes=nodes,
            conflict_adj=conflict_adj,
            graph_features=graph_features,
            satellite_ids=satellite_ids,
            params=params,
        )

    if spec.algorithm_key == "moaco":
        params = {
            **MOACO_DEFAULT_PARAMS,
            **model_params,
            **common_params,
            "num_ants": POP_SIZE,
            **spec.param_overrides,
        }
        return MOACO(
            tasks=tasks,
            nodes=nodes,
            conflict_adj=conflict_adj,
            satellite_ids=satellite_ids,
            params=params,
        )

    if spec.algorithm_key == "modbo":
        params = {
            **MODBO_DEFAULT_PARAMS,
            **common_params,
            "pop_size": POP_SIZE,
            **spec.param_overrides,
        }
        return MODBO(
            tasks=tasks,
            nodes=nodes,
            conflict_adj=conflict_adj,
            satellite_ids=satellite_ids,
            params=params,
            model_params=model_params,
        )

    if spec.algorithm_key == "mopso":
        params = {
            **MOPSO_DEFAULT_PARAMS,
            **common_params,
            "pop_size": POP_SIZE,
            **spec.param_overrides,
        }
        return MOPSO(
            tasks=tasks,
            nodes=nodes,
            conflict_adj=conflict_adj,
            satellite_ids=satellite_ids,
            params=params,
            model_params=model_params,
        )

    if spec.algorithm_key == "spea2":
        params = {
            **SPEA2_DEFAULT_PARAMS,
            **common_params,
            "pop_size": POP_SIZE,
            **spec.param_overrides,
        }
        return SPEA2(
            tasks=tasks,
            nodes=nodes,
            conflict_adj=conflict_adj,
            satellite_ids=satellite_ids,
            params=params,
            model_params=model_params,
        )

    if spec.algorithm_key == "nsga2":
        params = {
            **NSGA2_DEFAULT_PARAMS,
            **common_params,
            "pop_size": POP_SIZE,
            **spec.param_overrides,
        }
        return NSGA2(
            tasks=tasks,
            nodes=nodes,
            conflict_adj=conflict_adj,
            satellite_ids=satellite_ids,
            params=params,
            model_params=model_params,
        )

    raise ValueError(f"Unsupported algorithm key: {spec.algorithm_key}")


def run_one_experiment(
    spec: ExperimentSpec,
    problem: dict,
    results_dir: Path,
) -> dict:
    """在一个数据集上运行某个实验 RUN_COUNT 次。"""

    summary_file, archive_file = result_files(
        results_dir,
        spec.tag,
        problem["task_count"],
    )

    if (
        not OVERWRITE_RESULT_CSV
        and summary_file.exists()
        and archive_file.exists()
    ):
        print(f"\nSkip existing result: {summary_file}")
        return {
            "Group": spec.group,
            "Algorithm": spec.name,
            "Tag": spec.tag,
            "SummaryFile": str(summary_file),
            "ArchiveFile": str(archive_file),
        }

    print(f"\nRunning experiment: {spec.name}")
    print(f"Group: {spec.group}")
    print(f"Budget: {budget_label()}")

    nodes = problem["nodes"]
    tasks = problem["tasks"]
    conflict_adj = problem["conflict_adj"]
    nodes_by_id = {node.node_id: node for node in nodes}

    run_rows = []
    archive_rows = []

    for run_idx in range(1, RUN_COUNT + 1):
        print(f"\n========== {spec.name} Run {run_idx}/{RUN_COUNT} ==========")

        solver = build_algorithm(spec, problem, run_idx)
        archive = solver.run()

        summary = summarize_archive(
            archive,
            nodes_by_id,
            total_task_count=len(tasks),
        )

        runtime = getattr(solver, "runtime_seconds", "")
        rl_stats = getattr(solver, "rl_stats", {})

        run_row = {
            "Run": run_idx,
            "ExperimentGroup": spec.group,
            "Algorithm": spec.name,
            "Tag": spec.tag,
            "Case": problem["case_name"],
            "Archive_size": summary["archive_size"],
            "Best_f1": summary["best_f1"],
            "Best_f2": summary["best_f2"],
            "Best_f3": summary["best_f3"],
            "Best_completion_rate": summary["best_completion_rate"],
            "Runtime_seconds": runtime,
            "RL_T_ref": rl_stats.get("t_ref", ""),
            "RL_avg_episode_length": rl_stats.get("avg_episode_length", ""),
            "RL_last_train_loss": rl_stats.get("last_train_loss", ""),
            "RL_last_epsilon": rl_stats.get("last_epsilon", ""),
            "RL_last_kappa": rl_stats.get("last_kappa", ""),
            "RL_last_q_baseline": rl_stats.get("last_q_baseline", ""),
            "RL_last_advantage_span": rl_stats.get("last_advantage_span", ""),
            "RL_replay_size": rl_stats.get("replay_size", ""),
            "Candidate_nodes": len(nodes),
            "Conflict_edges": sum(len(v) for v in conflict_adj.values()) // 2,
            "Param_overrides": repr(spec.param_overrides),
        }
        run_rows.append(run_row)

        print(
            f"Run {run_idx} best: "
            f"archive={summary['archive_size']}, "
            f"best_f1={summary['best_f1']}, "
            f"best_f2={summary['best_f2']}, "
            f"best_f3={summary['best_f3']}, "
            f"completion={summary['best_completion_rate']}"
        )

        for sol_idx, sol in enumerate(archive):
            archive_rows.append(
                {
                    "Run": run_idx,
                    "ExperimentGroup": spec.group,
                    "Algorithm": spec.name,
                    "Tag": spec.tag,
                    "Case": problem["case_name"],
                    "Index": sol_idx,
                    "f1": sol.objectives[0],
                    "f2": sol.objectives[1],
                    "f3": sol.objectives[2],
                    "Node_ids": " ".join(str(node_id) for node_id in sorted(sol.node_ids)),
                }
            )

    # 添加 Average 和 Std 行
    numeric_keys = [
        "Archive_size",
        "Best_f1",
        "Best_f2",
        "Best_f3",
        "Best_completion_rate",
        "Runtime_seconds",
        "RL_T_ref",
        "RL_avg_episode_length",
        "RL_last_train_loss",
        "RL_last_epsilon",
        "RL_last_kappa",
        "RL_last_q_baseline",
        "RL_last_advantage_span",
        "RL_replay_size",
        "Candidate_nodes",
        "Conflict_edges",
    ]

    average_row = {
        "Run": "Average",
        "ExperimentGroup": spec.group,
        "Algorithm": spec.name,
        "Tag": spec.tag,
        "Case": problem["case_name"],
        "Param_overrides": repr(spec.param_overrides),
    }

    std_row = {
        "Run": "Std",
        "ExperimentGroup": spec.group,
        "Algorithm": spec.name,
        "Tag": spec.tag,
        "Case": problem["case_name"],
        "Param_overrides": repr(spec.param_overrides),
    }

    for key in numeric_keys:
        values = [row[key] for row in run_rows]
        average_row[key] = safe_mean(values)
        std_row[key] = safe_std(values)

    run_rows.append(average_row)
    run_rows.append(std_row)

    # 保存每轮 summary
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with summary_file.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "Run",
            "ExperimentGroup",
            "Algorithm",
            "Tag",
            "Case",
            "Archive_size",
            "Best_f1",
            "Best_f2",
            "Best_f3",
            "Best_completion_rate",
            "Runtime_seconds",
            "RL_T_ref",
            "RL_avg_episode_length",
            "RL_last_train_loss",
            "RL_last_epsilon",
            "RL_last_kappa",
            "RL_last_q_baseline",
            "RL_last_advantage_span",
            "RL_replay_size",
            "Candidate_nodes",
            "Conflict_edges",
            "Param_overrides",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(run_rows)

    # 保存每轮 Pareto 档案
    with archive_file.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "Run",
            "ExperimentGroup",
            "Algorithm",
            "Tag",
            "Case",
            "Index",
            "f1",
            "f2",
            "f3",
            "Node_ids",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(archive_rows)

    print(f"\nSummary saved to {summary_file}")
    print(f"Archive saved to {archive_file}")

    return {
        "Group": spec.group,
        "Algorithm": spec.name,
        "Tag": spec.tag,
        "SummaryFile": str(summary_file),
        "ArchiveFile": str(archive_file),
    }


def run_case(
    base_dir: Path,
    satellite_count: int,
    task_count: int,
    experiments: list[ExperimentSpec],
) -> list[dict]:
    """运行一个数据集上的全部开启算法。"""

    case_name = dataset_name(
        satellite_count,
        task_count,
        DEFAULT_DATASET_PREFIX,
    )

    output_dir = base_dir / RESULTS_ROOT / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========== CASE {case_name} ==========")
    print(f"Outputs: {output_dir}")
    print(f"Budget: {budget_label()}")

    problem = build_problem_interface(
        satellite_count,
        task_count,
        prefix=DEFAULT_DATASET_PREFIX,
    )

    print(
        f"Prepared {case_name}: "
        f"tasks={len(problem['tasks'])}, "
        f"nodes={len(problem['nodes'])}, "
        f"conflict_edges={sum(len(v) for v in problem['conflict_adj'].values()) // 2}"
    )

    outputs = []
    for spec in experiments:
        outputs.append(run_one_experiment(spec, problem, output_dir))
    return outputs


def save_experiment_index(base_dir: Path, rows: list[dict]) -> None:
    """保存本次运行产生的结果索引。"""

    if not rows:
        return

    index_file = base_dir / RESULTS_ROOT / "experiment_index.csv"
    index_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Case",
        "Group",
        "Algorithm",
        "Tag",
        "SummaryFile",
        "ArchiveFile",
    ]

    with index_file.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nExperiment index saved to {index_file}")


def main() -> None:
    """主函数：依次运行 CASE_LIST 中的所有数据集。"""

    base_dir = Path(__file__).parent
    experiments = enabled_experiments()

    print("\nEnabled experiment groups:")
    print(f"  - Comparison:         {RUN_COMPARISON_EXPERIMENTS}")
    print(f"  - RL ablation:        {RUN_RL_ABLATION_EXPERIMENTS}")
    print(f"  - Parameter analysis: {RUN_PARAMETER_ANALYSIS}")

    print("\nEnabled RL-CG-MOACO final-paper ablation algorithms:")
    print(f"  - RL-CG-MOACO-NO-DDQN:        {RUN_RL_CG_MOACO_NO_DDQN}")
    print(f"  - RL-CG-MOACO-NO-BLOCK:       {RUN_RL_CG_MOACO_NO_BLOCK_PENALTY}")
    print(f"  - RL-CG-MOACO-NO-FINAL:       {RUN_RL_CG_MOACO_NO_FINAL_REWARD}")
    print(f"  - RL-CG-MOACO-NO-SEARCH:      {RUN_RL_CG_MOACO_NO_SEARCH}")
    print(f"  - RL-CG-MOACO-NO-GRAPH-PHER:  {RUN_RL_CG_MOACO_NO_GRAPH_PHEROMONE}")

    print("\nEnabled comparison algorithms:")
    print(f"  - RL-CG-MOACO: {RUN_RL_CG_MOACO}")
    print(f"  - CG-MOACO: {RUN_CG_MOACO}")
    print(f"  - MODBO:    {RUN_MODBO}")
    print(f"  - MOPSO:    {RUN_MOPSO}")
    print(f"  - SPEA2:    {RUN_SPEA2}")
    print(f"  - MOACO:    {RUN_MOACO}")
    print(f"  - NSGA-II:  {RUN_NSGA2}")

    print("\nCommon experiment parameters:")
    print(f"  - RUN_COUNT       = {RUN_COUNT}")
    print(f"  - ITERATION_COUNT = {ITERATION_COUNT}")
    print(f"  - POP_SIZE        = {POP_SIZE}")
    print(f"  - ARCHIVE_SIZE    = {ARCHIVE_SIZE}")
    print(f"  - OVERWRITE_RESULT_CSV = {OVERWRITE_RESULT_CSV}")

    print("\nSelected cases:")
    for satellite_count, task_count in CASE_LIST:
        print(f"  - satellites={satellite_count}, tasks={task_count}")

    print("\nSelected experiments:")
    for spec in experiments:
        print(f"  - [{spec.group}] {spec.tag}: {spec.name}")

    if not CASE_LIST:
        raise ValueError("CASE_LIST is empty. Please enable at least one dataset.")
    if not experiments:
        raise ValueError("No experiment is enabled. Please turn on at least one algorithm or experiment group.")

    index_rows = []
    for satellite_count, task_count in CASE_LIST:
        case_outputs = run_case(base_dir, satellite_count, task_count, experiments)
        for row in case_outputs:
            index_rows.append(
                {
                    "Case": dataset_name(satellite_count, task_count),
                    **row,
                }
            )

    save_experiment_index(base_dir, index_rows)

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
