from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .domain import CandidateNode, Task
from .problem_model import DEFAULT_MODEL_PARAMS, evaluate_solution, is_feasible, task_completion_rate
from .utils import Solution, dominates, set_random_seed, update_archive


# =========================================================
# NSGA-II 默认参数
#
# 注意：
#   max_iter 不在这里设置。
#   main.py 中通过：
#
#       params = {
#           **NSGA2_DEFAULT_PARAMS,
#           "max_iter": GENERATIONS,
#           "seed": seed,
#       }
#
#   统一传入迭代次数。
# =========================================================
DEFAULT_PARAMS = {
    # ---------- 种群与档案 ----------
    # pop_size：种群规模
    # archive_size：Pareto 外部档案容量
    "pop_size": 50,
    "archive_size": 100,

    # ---------- 随机键编码边界 ----------
    "lower_bound": 0.0,
    "upper_bound": 1.0,

    # ---------- 交叉与变异 ----------
    # crossover_probability：模拟二进制交叉 SBX 的执行概率
    # mutation_probability：多项式变异概率；若为 None，则默认 1 / dimension
    # eta_c：SBX 分布指数，越大子代越接近父代
    # eta_m：多项式变异分布指数，越大变异幅度越小
    "crossover_probability": 0.8,
    "mutation_probability": None,
    "eta_c": 15.0,
    "eta_m": 20.0,
    "ref_front_file": None,
    "print_every": 10,

    # ---------- 锦标赛选择 ----------
    "tournament_size": 2,

    # ---------- 随机种子与日志 ----------
    # seed 由 main.py 每轮运行时覆盖
    "seed": 42,
    "verbose": True,
}


class _Individual:
    """NSGA-II 内部个体结构。"""

    def __init__(self, chromosome: list[float], solution: Solution | None = None) -> None:
        self.chromosome = chromosome
        self.solution = solution
        self.rank: int = 10**9
        self.crowding_distance: float = 0.0


class NSGA2:
    """适配当前卫星调度问题的 NSGA-II 对比算法。"""

    def __init__(
        self,
        tasks: Dict[int, Task],
        nodes: List[CandidateNode],
        conflict_adj: dict[int, set[int]],
        params: dict | None = None,
        satellite_ids: Sequence[int] | None = None,
        model_params: dict | None = None,
        pop_size: int | None = None,
        generations: int | None = None,
        ref_front_file: str | Path | None = None,
        seed: int | None = None,
        crossover_prob: float | None = None,
        crossover_eta: float | None = None,
        mutation_eta: float | None = None,
        mutation_probability: float | None = None,
        print_every: int | None = None,
    ) -> None:
        # 合并默认参数和外部参数
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self._apply_pymoo_style_aliases()

        explicit_overrides = {
            "pop_size": pop_size,
            "max_iter": generations,
            "ref_front_file": ref_front_file,
            "seed": seed,
            "crossover_probability": crossover_prob,
            "eta_c": crossover_eta,
            "eta_m": mutation_eta,
            "mutation_probability": mutation_probability,
            "print_every": print_every,
        }
        for key, value in explicit_overrides.items():
            if value is not None:
                self.params[key] = value
        self.model_params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}

        # max_iter 必须由 main.py 统一传入
        if "max_iter" not in self.params:
            raise ValueError(
                "NSGA2 requires params['max_iter']. "
                "Please pass it from main.py using ITERATION_COUNT or generations."
            )

        # 设置随机种子
        set_random_seed(self.params.get("seed"))

        self.tasks = tasks
        self.nodes = nodes
        self.nodes_by_id = {node.node_id: node for node in nodes}
        self.node_ids = [node.node_id for node in nodes]
        self.conflict_adj = conflict_adj
        self.satellite_ids = sorted(
            set(satellite_ids) if satellite_ids is not None else {node.sat_id for node in nodes}
        )

        # 染色体维度 = 候选节点数量
        self.dimension = len(self.nodes)

        # 当前种群
        self.population: list[_Individual] = []

        # Pareto 外部档案
        self.archive: List[Solution] = []

        self.runtime_seconds = 0.0

    def _apply_pymoo_style_aliases(self) -> None:
        """Map familiar NSGA2_PYMOO-style names to this implementation."""

        alias_map = {
            "generations": "max_iter",
            "crossover_prob": "crossover_probability",
            "crossover_eta": "eta_c",
            "mutation_eta": "eta_m",
        }
        for old_key, new_key in alias_map.items():
            if old_key in self.params and self.params[old_key] is not None:
                self.params[new_key] = self.params[old_key]

    # =====================================================
    # 随机键编码与解码
    # =====================================================
    def _clip_value(self, value: float) -> float:
        """将基因值限制在上下界内。"""

        lb = float(self.params["lower_bound"])
        ub = float(self.params["upper_bound"])
        return max(lb, min(ub, value))

    def _clip_chromosome(self, chromosome: Sequence[float]) -> list[float]:
        """将整条染色体限制在上下界内。"""

        return [self._clip_value(x) for x in chromosome]

    def _random_chromosome(self) -> list[float]:
        """随机生成一条染色体。"""

        lb = float(self.params["lower_bound"])
        ub = float(self.params["upper_bound"])

        return [
            random.uniform(lb, ub)
            for _ in range(self.dimension)
        ]

    def _decode_chromosome(self, chromosome: Sequence[float]) -> set[int]:
        """将随机键染色体解码为可行调度方案。

        解码策略：
            按基因值从大到小排序候选节点，
            依次尝试加入当前解。
            若节点与当前已选节点无冲突，则加入。
        """

        order = sorted(
            range(self.dimension),
            key=lambda idx: chromosome[idx],
            reverse=True,
        )

        solution: set[int] = set()

        for idx in order:
            node_id = self.node_ids[idx]
            if not (self.conflict_adj.get(node_id, set()) & solution):
                solution.add(node_id)

        return solution

    def _repair_by_conflict_removal(self, node_ids: Iterable[int]) -> set[int]:
        """防御性修复：若解中存在冲突，则按收益优先保留节点。"""

        repaired: set[int] = set()

        ordered = sorted(
            node_ids,
            key=lambda nid: self.nodes_by_id[nid].profit,
            reverse=True,
        )

        for nid in ordered:
            if not (self.conflict_adj.get(nid, set()) & repaired):
                repaired.add(nid)

        return repaired

    def _make_solution(self, node_ids: Iterable[int]) -> Solution:
        """根据 node_id 集合构造 Solution 对象。"""

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

    def _evaluate_chromosome(self, chromosome: Sequence[float]) -> Solution:
        """评价一条染色体。"""

        node_ids = self._decode_chromosome(chromosome)

        if not is_feasible(node_ids, self.conflict_adj):
            node_ids = self._repair_by_conflict_removal(node_ids)

        return self._make_solution(node_ids)

    def _make_individual(self, chromosome: Sequence[float]) -> _Individual:
        """生成并评价个体。"""

        clipped = self._clip_chromosome(chromosome)
        sol = self._evaluate_chromosome(clipped)
        return _Individual(chromosome=clipped, solution=sol)

    # =====================================================
    # 非支配排序与拥挤距离
    # =====================================================
    def _fast_non_dominated_sort(self, individuals: list[_Individual]) -> list[list[_Individual]]:
        """NSGA-II 快速非支配排序。"""

        domination_sets: dict[int, list[int]] = {}
        dominated_counts: dict[int, int] = {}
        fronts: list[list[int]] = [[]]

        for i, p in enumerate(individuals):
            domination_sets[i] = []
            dominated_counts[i] = 0

            for j, q in enumerate(individuals):
                if i == j:
                    continue

                if dominates(p.solution.objectives, q.solution.objectives):
                    domination_sets[i].append(j)
                elif dominates(q.solution.objectives, p.solution.objectives):
                    dominated_counts[i] += 1

            if dominated_counts[i] == 0:
                p.rank = 0
                fronts[0].append(i)

        current_front = 0

        while fronts[current_front]:
            next_front: list[int] = []

            for i in fronts[current_front]:
                for j in domination_sets[i]:
                    dominated_counts[j] -= 1

                    if dominated_counts[j] == 0:
                        individuals[j].rank = current_front + 1
                        next_front.append(j)

            current_front += 1
            fronts.append(next_front)

        # 删除最后一个空 front，并转成个体列表
        if fronts and not fronts[-1]:
            fronts.pop()

        return [
            [individuals[i] for i in front]
            for front in fronts
        ]

    def _assign_crowding_distance(self, front: list[_Individual]) -> None:
        """计算某一非支配层内个体的拥挤距离。"""

        if not front:
            return

        for ind in front:
            ind.crowding_distance = 0.0

        num_objectives = len(front[0].solution.objectives)

        if len(front) <= 2:
            for ind in front:
                ind.crowding_distance = float("inf")
            return

        for m in range(num_objectives):
            front.sort(key=lambda ind: ind.solution.objectives[m])

            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            min_obj = front[0].solution.objectives[m]
            max_obj = front[-1].solution.objectives[m]
            denom = max(max_obj - min_obj, 1e-12)

            for i in range(1, len(front) - 1):
                prev_obj = front[i - 1].solution.objectives[m]
                next_obj = front[i + 1].solution.objectives[m]
                front[i].crowding_distance += (next_obj - prev_obj) / denom

    def _rank_and_crowding(self, individuals: list[_Individual]) -> list[list[_Individual]]:
        """对种群执行非支配排序并计算拥挤距离。"""

        fronts = self._fast_non_dominated_sort(individuals)

        for front in fronts:
            self._assign_crowding_distance(front)

        return fronts

    # =====================================================
    # 选择、交叉、变异
    # =====================================================
    def _tournament_select(self) -> _Individual:
        """拥挤比较锦标赛选择。"""

        k = int(self.params.get("tournament_size", 2))
        candidates = random.sample(self.population, k)

        # rank 越小越好，crowding_distance 越大越好
        candidates.sort(
            key=lambda ind: (
                ind.rank,
                -ind.crowding_distance,
            )
        )

        return candidates[0]

    def _sbx_crossover(
        self,
        parent1: Sequence[float],
        parent2: Sequence[float],
    ) -> tuple[list[float], list[float]]:
        """模拟二进制交叉 SBX。"""

        crossover_probability = float(self.params.get("crossover_probability", 0.9))
        eta_c = float(self.params.get("eta_c", 20.0))
        lb = float(self.params["lower_bound"])
        ub = float(self.params["upper_bound"])

        if random.random() > crossover_probability:
            return list(parent1), list(parent2)

        child1 = list(parent1)
        child2 = list(parent2)

        for i in range(self.dimension):
            x1 = parent1[i]
            x2 = parent2[i]

            if random.random() > 0.5 or abs(x1 - x2) < 1e-14:
                continue

            if x1 > x2:
                x1, x2 = x2, x1

            rand = random.random()

            beta = 1.0 + (2.0 * (x1 - lb) / max(x2 - x1, 1e-14))
            alpha = 2.0 - beta ** (-(eta_c + 1.0))

            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))

            c1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

            beta = 1.0 + (2.0 * (ub - x2) / max(x2 - x1, 1e-14))
            alpha = 2.0 - beta ** (-(eta_c + 1.0))

            if rand <= 1.0 / alpha:
                betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))

            c2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))

            c1 = self._clip_value(c1)
            c2 = self._clip_value(c2)

            if random.random() <= 0.5:
                child1[i] = c2
                child2[i] = c1
            else:
                child1[i] = c1
                child2[i] = c2

        return child1, child2

    def _polynomial_mutation(self, chromosome: Sequence[float]) -> list[float]:
        """多项式变异。"""

        eta_m = float(self.params.get("eta_m", 20.0))
        mutation_probability = self.params.get("mutation_probability")

        if mutation_probability is None:
            mutation_probability = 1.0 / max(1, self.dimension)

        mutation_probability = float(mutation_probability)

        lb = float(self.params["lower_bound"])
        ub = float(self.params["upper_bound"])

        mutated = list(chromosome)

        for i in range(self.dimension):
            if random.random() > mutation_probability:
                continue

            x = mutated[i]
            delta1 = (x - lb) / max(ub - lb, 1e-14)
            delta2 = (ub - x) / max(ub - lb, 1e-14)
            rand = random.random()
            mut_pow = 1.0 / (eta_m + 1.0)

            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                deltaq = (val ** mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                deltaq = 1.0 - (val ** mut_pow)

            x = x + deltaq * (ub - lb)
            mutated[i] = self._clip_value(x)

        return mutated

    def _make_offspring(self) -> list[_Individual]:
        """生成子代种群。"""

        pop_size = int(self.params.get("pop_size", 50))
        offspring: list[_Individual] = []

        while len(offspring) < pop_size:
            p1 = self._tournament_select()
            p2 = self._tournament_select()

            c1, c2 = self._sbx_crossover(
                p1.chromosome,
                p2.chromosome,
            )

            c1 = self._polynomial_mutation(c1)
            c2 = self._polynomial_mutation(c2)

            offspring.append(self._make_individual(c1))

            if len(offspring) < pop_size:
                offspring.append(self._make_individual(c2))

        return offspring

    # =====================================================
    # 环境选择与档案更新
    # =====================================================
    def _environmental_selection(
        self,
        combined: list[_Individual],
    ) -> list[_Individual]:
        """NSGA-II 环境选择。"""

        pop_size = int(self.params.get("pop_size", 50))

        fronts = self._rank_and_crowding(combined)

        next_population: list[_Individual] = []

        for front in fronts:
            if len(next_population) + len(front) <= pop_size:
                next_population.extend(front)
            else:
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                remaining = pop_size - len(next_population)
                next_population.extend(front[:remaining])
                break

        return next_population

    def _update_archive_from_population(self) -> None:
        """根据当前种群更新 Pareto 外部档案。"""

        archive_size = int(self.params["archive_size"])
        solutions = [ind.solution for ind in self.population]

        self.archive = update_archive(
            self.archive,
            solutions,
            archive_size,
        )

    # =====================================================
    # 初始化与主循环
    # =====================================================
    def _initialize_population(self) -> None:
        """初始化种群。"""

        pop_size = int(self.params.get("pop_size", 50))

        self.population = [
            self._make_individual(self._random_chromosome())
            for _ in range(pop_size)
        ]

        self._rank_and_crowding(self.population)
        self._update_archive_from_population()

    def run(self) -> List[Solution]:
        """运行 NSGA-II 并返回 Pareto 档案。"""

        start_time = time.perf_counter()

        max_iter = int(self.params["max_iter"])
        verbose = bool(self.params.get("verbose", True))
        print_every = self.params.get("print_every")
        print_every = int(print_every) if print_every is not None else None

        self._initialize_population()

        for iteration in range(1, max_iter + 1):
            offspring = self._make_offspring()

            combined = self.population + offspring

            self.population = self._environmental_selection(combined)

            # 重新计算当前种群 rank 和 crowding distance，供下一轮锦标赛使用
            self._rank_and_crowding(self.population)

            self._update_archive_from_population()

            should_print = (
                iteration == 1
                or iteration == max_iter
                or (
                    print_every is not None
                    and iteration % max(1, print_every) == 0
                )
                or (
                    print_every is None
                    and iteration % max(1, max_iter // 10) == 0
                )
            )

            if verbose and should_print:
                best_f1 = max(
                    (-s.objectives[0] for s in self.archive),
                    default=float("nan"),
                )

                print(
                    f"[NSGA-II] Iter {iteration:>4}/{max_iter}: "
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
