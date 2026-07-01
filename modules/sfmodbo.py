from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .domain import CandidateNode, Task
from .problem_model import (
    DEFAULT_MODEL_PARAMS,
    evaluate_solution,
    task_completion_rate,
)
from .utils import Solution, dominates, set_random_seed, update_archive


# =========================================================
# SFMODBO 默认参数
#
# 注意：
#   1. max_iter 不在这里设置。
#      所有算法的迭代次数统一由 main.py 中的 GENERATIONS 控制，
#      并在 build_algorithm() 中通过 params["max_iter"] 传入。
#
#   2. 问题模型参数不在这里设置。
#      min_transition_time、maneuver_time_per_degree、load_mode 等
#      由 problem_model.py 中的 DEFAULT_MODEL_PARAMS 统一管理。
#
#   3. 本文件只保留 SFMODBO 自身搜索参数。
# =========================================================
DEFAULT_PARAMS = {
    # ---------- 种群与档案 ----------
    # pop_size：蜣螂种群规模
    # archive_size：Pareto 外部档案容量
    "pop_size": 50,
    "archive_size": 100,

    # ---------- 随机键编码边界 ----------
    # 每个候选节点对应一个连续位置值。
    # 解码时按位置值从大到小尝试选择节点。
    "lower_bound": 0.0,
    "upper_bound": 1.0,

    # ---------- DBO 群体角色比例 ----------
    # roller_ratio：滚动蜣螂比例
    # brood_ratio：繁殖/产卵蜣螂比例
    # forager_ratio：觅食蜣螂比例
    # stealer_ratio：偷窃蜣螂比例
    #
    # 四者最好近似相加为 1.0。
    "roller_ratio": 0.25,
    "brood_ratio": 0.25,
    "forager_ratio": 0.25,
    "stealer_ratio": 0.25,

    # ---------- 不同 DBO 策略的搜索强度 ----------
    "rolling_step": 0.15,
    "dance_probability": 0.10,
    "dance_angle_min_deg": 1.0,
    "dance_angle_max_deg": 89.0,
    "brood_radius": 0.20,
    "forage_step": 0.30,
    "steal_step": 0.50,

    # ---------- 变异参数 ----------
    "mutation_probability": 0.05,
    "mutation_strength": 0.10,

    # ---------- 多目标接受准则参数 ----------
    # 当新旧解互不支配时，按一定概率使用随机加权标量值判断是否接受新解。
    "scalar_accept_probability": 0.35,

    # ---------- 随机种子与日志 ----------
    # seed 由 main.py 每轮运行时覆盖。
    "seed": 42,
    "verbose": True,
}


class SFMODBO:

    LOG_NAME = "SFMODBO"

    def __init__(
        self,
        tasks: Dict[int, Task],
        nodes: List[CandidateNode],
        conflict_adj: dict[int, set[int]],
        params: dict | None = None,
        satellite_ids: Sequence[int] | None = None,
        model_params: dict | None = None,
    ) -> None:
        # 合并默认参数和外部传入参数
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model_params = {
            **DEFAULT_MODEL_PARAMS,
            **(model_params or {}),
        }

        # max_iter 必须由 main.py 统一传入
        if "max_iter" not in self.params:
            raise ValueError(
                "SFMODBO requires params['max_iter']. "
                "Please pass it from main.py using GENERATIONS."
            )

        # 设置随机种子
        set_random_seed(self.params.get("seed"))

        self.tasks = tasks
        self.nodes = nodes
        self.nodes_by_id = {node.node_id: node for node in nodes}

        # 候选节点编号列表
        self.node_ids = [node.node_id for node in nodes]

        # node_id 到连续向量下标的映射
        self.node_index = {
            node.node_id: idx
            for idx, node in enumerate(nodes)
        }

        self.conflict_adj = conflict_adj
        self.satellite_ids = sorted(
            set(satellite_ids) if satellite_ids is not None else {node.sat_id for node in nodes}
        )

        # 随机键向量维度 = 候选节点数量
        self.dimension = len(self.nodes)

        # 种群连续位置向量
        self.population: list[list[float]] = []

        # 种群对应的调度解
        self.population_solutions: list[Solution] = []

        # Pareto 外部档案
        self.archive: List[Solution] = []

        # 记录档案解对应的位置向量
        self.archive_positions: dict[frozenset[int], list[float]] = {}

        self.runtime_seconds = 0.0

    # =====================================================
    # 基础工具函数
    # =====================================================
    def _clip_value(self, value: float) -> float:
        """将单个位置值限制在上下界内。"""

        lb = float(self.params["lower_bound"])
        ub = float(self.params["upper_bound"])
        return max(lb, min(ub, value))

    def _clip_position(self, position: Sequence[float]) -> list[float]:
        """将整个连续位置向量限制在上下界内。"""

        return [self._clip_value(v) for v in position]

    def _random_position(self) -> list[float]:
        """随机生成一个连续位置向量。"""

        lb = float(self.params["lower_bound"])
        ub = float(self.params["upper_bound"])

        return [
            random.uniform(lb, ub)
            for _ in range(self.dimension)
        ]

    def _mutate(self, position: list[float]) -> list[float]:
        """对连续位置向量执行随机扰动变异。"""

        probability = float(self.params.get("mutation_probability", 0.05))
        strength = float(self.params.get("mutation_strength", 0.10))

        if probability <= 0 or strength <= 0:
            return position

        lb = float(self.params["lower_bound"])
        ub = float(self.params["upper_bound"])
        span = ub - lb

        mutated = position[:]

        for i in range(self.dimension):
            if random.random() < probability:
                mutated[i] += random.uniform(-strength, strength) * span

        return self._clip_position(mutated)

    # =====================================================
    # 编码与解码
    # =====================================================
    def _decode_position(self, position: Sequence[float]) -> set[int]:
        """将连续随机键位置向量解码为可行调度方案。

        解码步骤：
            1. 按 position 值从大到小排序候选节点；
            2. 依次尝试选择节点；
            3. 若该节点与当前已选节点不冲突，则加入调度方案；
            4. 最终得到一个满足冲突约束的节点集合。
        """

        order = sorted(
            range(self.dimension),
            key=lambda idx: position[idx],
            reverse=True,
        )

        solution: set[int] = set()

        for idx in order:
            node_id = self.node_ids[idx]

            # 若该节点与当前已选节点无冲突，则加入方案
            if not (self.conflict_adj.get(node_id, set()) & solution):
                solution.add(node_id)

        return solution

    def _make_solution(self, node_ids: Iterable[int]) -> Solution:
        """根据 node_id 集合构造 Solution 对象。

        目标函数直接调用 problem_model.evaluate_solution()，
        并使用 main.py 传入的统一问题模型参数。
        """

        frozen = frozenset(node_ids)

        objectives = evaluate_solution(
            frozen,
            self.nodes_by_id,
            self.tasks,
            self.satellite_ids,
            self.model_params,
        )

        return Solution(
            node_ids=frozen,
            objectives=objectives,
        )

    def _evaluate_position(self, position: Sequence[float]) -> Solution:
        """评价一个连续位置向量。"""

        node_ids = self._decode_position(position)
        return self._make_solution(node_ids)

    # =====================================================
    # 多目标比较与接受准则
    # =====================================================
    def _weighted_score(
        self,
        objectives: Sequence[float],
        weights: Sequence[float] | None = None,
    ) -> float:
        """将三目标向量转成加权标量值。

        注意：
            所有目标均为最小化。
            该函数仅用于互不支配情况下的辅助接受判断，
            不改变算法的 Pareto 多目标本质。
        """

        if weights is None:
            # f2 姿态机动代价可能数值较大，因此默认给较小权重。
            weights = (1.0, 0.01, 1.0)

        return sum(
            w * float(v)
            for w, v in zip(weights, objectives)
        )

    def _better_or_accept(
        self,
        old: Solution,
        new: Solution,
    ) -> bool:
        """判断是否接受新解。

        接受规则：
            1. 如果 new 支配 old，则接受；
            2. 如果 old 支配 new，则拒绝；
            3. 如果二者互不支配，则使用随机权重标量化辅助判断；
            4. 同时保留一定随机接受概率以维持多样性。
        """

        if dominates(new.objectives, old.objectives):
            return True

        if dominates(old.objectives, new.objectives):
            return False

        if random.random() < float(self.params.get("scalar_accept_probability", 0.35)):
            weights = [random.random() + 1e-9 for _ in range(3)]
            total = sum(weights)
            weights = [w / total for w in weights]

            return self._weighted_score(new.objectives, weights) < self._weighted_score(
                old.objectives,
                weights,
            )

        return random.random() < 0.5

    def _select_reference_position(self) -> list[float]:
        """从 Pareto 档案或当前种群中选择参考位置。"""

        # 优先从 Pareto 档案中随机选择参考解
        if self.archive:
            sol = random.choice(self.archive)
            pos = self.archive_positions.get(sol.node_ids)
            if pos is not None:
                return pos[:]

        # 如果档案为空，则选择当前种群中加权分数最好的个体
        if self.population_solutions:
            best_idx = min(
                range(len(self.population_solutions)),
                key=lambda i: self._weighted_score(
                    self.population_solutions[i].objectives
                ),
            )
            return self.population[best_idx][:]

        return self._random_position()

    def _worst_position(self) -> list[float]:
        """返回当前种群中加权分数最差个体的位置。"""

        if not self.population_solutions:
            return self._random_position()

        worst_idx = max(
            range(len(self.population_solutions)),
            key=lambda i: self._weighted_score(
                self.population_solutions[i].objectives
            ),
        )

        return self.population[worst_idx][:]

    # =====================================================
    # DBO 策略融合更新算子
    # =====================================================
    def _roller_update(
        self,
        pos: Sequence[float],
        worst: Sequence[float],
    ) -> list[float]:
        """滚动/跳舞行为更新。

        滚动蜣螂用于全局探索：
            1. 大多数情况下沿远离较差区域的方向移动；
            2. 少数情况下执行跳舞转向扰动。
        """

        step = float(self.params.get("rolling_step", 0.15))
        dance_probability = float(self.params.get("dance_probability", 0.10))

        new_pos = []

        if random.random() < dance_probability:
            amin = float(self.params.get("dance_angle_min_deg", 1.0))
            amax = float(self.params.get("dance_angle_max_deg", 89.0))
            theta = math.radians(random.uniform(amin, amax))

            factor = math.tan(theta)
            factor = max(-3.0, min(3.0, factor))

            for x in pos:
                new_pos.append(
                    x + step * factor * random.uniform(-1.0, 1.0)
                )
        else:
            direction = random.choice([-1.0, 1.0])

            for x, w in zip(pos, worst):
                new_pos.append(
                    x + direction * step * abs(x - w)
                )

        return self._mutate(self._clip_position(new_pos))

    def _brood_update(
        self,
        pos: Sequence[float],
        best: Sequence[float],
        iteration: int,
        max_iter: int,
    ) -> list[float]:
        """繁殖/产卵行为更新。

        该策略主要在优秀区域附近进行局部搜索。
        搜索半径随迭代次数逐渐减小。
        """

        radius0 = float(self.params.get("brood_radius", 0.20))
        radius = radius0 * (1.0 - iteration / max(1, max_iter))

        new_pos = []

        for x, b in zip(pos, best):
            low = max(float(self.params["lower_bound"]), b - radius)
            high = min(float(self.params["upper_bound"]), b + radius)

            local = random.uniform(low, high)
            new_pos.append(0.5 * x + 0.5 * local)

        return self._mutate(self._clip_position(new_pos))

    def _forager_update(
        self,
        pos: Sequence[float],
        best: Sequence[float],
    ) -> list[float]:
        """觅食行为更新。

        个体同时受到当前优秀位置和档案参考位置的引导。
        """

        step = float(self.params.get("forage_step", 0.30))
        ref = self._select_reference_position()

        new_pos = []

        for x, b, r in zip(pos, best, ref):
            new_pos.append(
                x
                + random.random() * step * (b - x)
                + random.random() * step * (r - x)
            )

        return self._mutate(self._clip_position(new_pos))

    def _stealer_update(
        self,
        pos: Sequence[float],
        best: Sequence[float],
    ) -> list[float]:
        """偷窃行为更新。

        个体围绕优秀位置进行较强 exploitation。
        """

        step = float(self.params.get("steal_step", 0.50))
        ref = self._select_reference_position()

        new_pos = []

        for x, b, r in zip(pos, best, ref):
            new_pos.append(
                b + random.uniform(-step, step) * (abs(x - b) + abs(r - b))
            )

        return self._mutate(self._clip_position(new_pos))

    def _strategy_update(
        self,
        index: int,
        position: Sequence[float],
        best: Sequence[float],
        worst: Sequence[float],
        iteration: int,
        max_iter: int,
    ) -> list[float]:
        """根据个体所在分组选择对应的 DBO 更新策略。"""

        pop_size = len(self.population)

        roller_end = int(
            pop_size * float(self.params.get("roller_ratio", 0.25))
        )
        brood_end = roller_end + int(
            pop_size * float(self.params.get("brood_ratio", 0.25))
        )
        forager_end = brood_end + int(
            pop_size * float(self.params.get("forager_ratio", 0.25))
        )

        if index < roller_end:
            return self._roller_update(position, worst)

        if index < brood_end:
            return self._brood_update(position, best, iteration, max_iter)

        if index < forager_end:
            return self._forager_update(position, best)

        return self._stealer_update(position, best)

    # =====================================================
    # 种群初始化与 Pareto 档案更新
    # =====================================================
    def _initialize_population(self) -> None:
        """初始化 SFMODBO 种群。"""

        pop_size = int(self.params.get("pop_size", 50))

        self.population = [
            self._random_position()
            for _ in range(pop_size)
        ]

        self.population_solutions = [
            self._evaluate_position(pos)
            for pos in self.population
        ]

        self._update_archive_with_positions(
            self.population_solutions,
            self.population,
        )

    def _update_archive_with_positions(
        self,
        solutions: Sequence[Solution],
        positions: Sequence[Sequence[float]],
    ) -> None:
        """更新 Pareto 外部档案，并保存档案解对应的位置向量。"""

        archive_size = int(self.params["archive_size"])

        old_archive = list(self.archive)

        self.archive = update_archive(
            self.archive,
            solutions,
            archive_size,
        )

        archive_keys = {sol.node_ids for sol in self.archive}

        # 保存当前新进入档案的解对应的位置向量
        for sol, pos in zip(solutions, positions):
            if sol.node_ids in archive_keys:
                self.archive_positions[sol.node_ids] = list(pos)

        # 删除已经不在档案中的位置记录
        self.archive_positions = {
            key: value
            for key, value in self.archive_positions.items()
            if key in archive_keys
        }

        # 如果旧档案中的解保留了下来但缺少位置记录，则补一个随机位置
        for sol in old_archive:
            if (
                sol.node_ids in archive_keys
                and sol.node_ids not in self.archive_positions
            ):
                self.archive_positions[sol.node_ids] = self._random_position()

    # =====================================================
    # 主循环
    # =====================================================
    def run(self) -> List[Solution]:
        """运行 SFMODBO 算法并返回 Pareto 档案。"""

        start_time = time.perf_counter()

        max_iter = int(self.params["max_iter"])
        verbose = bool(self.params.get("verbose", True))

        self._initialize_population()

        for iteration in range(1, max_iter + 1):
            best = self._select_reference_position()
            worst = self._worst_position()

            new_population: list[list[float]] = []
            new_solutions: list[Solution] = []

            for idx, pos in enumerate(self.population):
                old_sol = self.population_solutions[idx]

                candidate_pos = self._strategy_update(
                    idx,
                    pos,
                    best,
                    worst,
                    iteration,
                    max_iter,
                )

                candidate_sol = self._evaluate_position(candidate_pos)

                if self._better_or_accept(old_sol, candidate_sol):
                    new_population.append(candidate_pos)
                    new_solutions.append(candidate_sol)
                else:
                    new_population.append(pos)
                    new_solutions.append(old_sol)

            self.population = new_population
            self.population_solutions = new_solutions

            self._update_archive_with_positions(
                new_solutions,
                new_population,
            )

            if verbose and (
                iteration == 1
                or iteration % max(1, max_iter // 10) == 0
                or iteration == max_iter
            ):
                best_f1 = max(
                    (-s.objectives[0] for s in self.archive),
                    default=float("nan"),
                )

                print(
                    f"[{self.LOG_NAME}] Iter {iteration:>4}/{max_iter}: "
                    f"archive={len(self.archive):>3}, "
                    f"best_f1={best_f1:.4f}"
                )

        self.runtime_seconds = time.perf_counter() - start_time

        return self.archive

    # =====================================================
    # 结果保存
    # =====================================================
    def save_archive_csv(self, file_path: str | Path) -> None:
        """保存 Pareto 档案到 CSV 文件。"""

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    "solution_index",
                    "f1_total_profit",
                    "f2_maneuver_cost",
                    "f3_load_imbalance",
                    "scheduled_nodes",
                    "scheduled_tasks",
                    "task_completion_rate",
                    "node_ids",
                ]
            )

            total_task_count = len(self.tasks)

            for idx, sol in enumerate(self.archive):
                scheduled_tasks = {
                    self.nodes_by_id[nid].task_id
                    for nid in sol.node_ids
                }

                writer.writerow(
                    [
                        idx,
                        -sol.objectives[0],
                        sol.objectives[1],
                        sol.objectives[2],
                        len(sol.node_ids),
                        len(scheduled_tasks),
                        task_completion_rate(
                            sol.node_ids,
                            total_task_count,
                            self.nodes_by_id,
                        ),
                        " ".join(map(str, sorted(sol.node_ids))),
                    ]
                )
