"""SPEA2 baseline for the satellite scheduling problem."""

from __future__ import annotations

import csv
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .domain import CandidateNode, Task
from .problem_model import DEFAULT_MODEL_PARAMS, evaluate_solution, is_feasible, task_completion_rate
from .utils import Solution, dominates, set_random_seed, update_archive


DEFAULT_PARAMS = {
    "pop_size": 50,
    "max_iter": 100,
    "archive_size": 100,
    "lower_bound": 0.0,
    "upper_bound": 1.0,
    "crossover_probability": 0.90,
    "mutation_probability": None,
    "mutation_strength": 0.10,
    "tournament_size": 2,
    "seed": 42,
    "verbose": True,
}


class SPEA2:
    """Strength Pareto evolutionary algorithm using random-key decoding."""

    def __init__(
        self,
        tasks: Dict[int, Task],
        nodes: List[CandidateNode],
        conflict_adj: dict[int, set[int]],
        params: dict | None = None,
        satellite_ids: Sequence[int] | None = None,
        model_params: dict | None = None,
    ) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model_params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
        set_random_seed(self.params.get("seed"))

        self.tasks = tasks
        self.nodes = nodes
        self.nodes_by_id = {node.node_id: node for node in nodes}
        self.node_ids = [node.node_id for node in nodes]
        self.conflict_adj = conflict_adj
        self.satellite_ids = sorted(
            set(satellite_ids) if satellite_ids is not None else {node.sat_id for node in nodes}
        )
        self.dimension = len(nodes)
        self.population: list[list[float]] = []
        self.population_solutions: list[Solution] = []
        self.archive: List[Solution] = []
        self.archive_positions: list[list[float]] = []
        self.runtime_seconds = 0.0

    def _clip_value(self, value: float) -> float:
        return max(float(self.params["lower_bound"]), min(float(self.params["upper_bound"]), value))

    def _clip_position(self, position: Sequence[float]) -> list[float]:
        return [self._clip_value(v) for v in position]

    def _random_position(self) -> list[float]:
        lb = float(self.params["lower_bound"])
        ub = float(self.params["upper_bound"])
        return [random.uniform(lb, ub) for _ in range(self.dimension)]

    def _decode_position(self, position: Sequence[float]) -> set[int]:
        order = sorted(range(self.dimension), key=lambda idx: position[idx], reverse=True)
        solution: set[int] = set()
        for idx in order:
            node_id = self.node_ids[idx]
            if not (self.conflict_adj.get(node_id, set()) & solution):
                solution.add(node_id)
        return solution

    def _make_solution(self, node_ids: Iterable[int]) -> Solution:
        frozen = frozenset(node_ids)
        return Solution(
            node_ids=frozen,
            objectives=evaluate_solution(
                frozen,
                self.nodes_by_id,
                self.tasks,
                self.satellite_ids,
                self.model_params,
            ),
        )

    def _evaluate_position(self, position: Sequence[float]) -> Solution:
        node_ids = self._decode_position(position)
        if not is_feasible(node_ids, self.conflict_adj):
            repaired: set[int] = set()
            for nid in sorted(node_ids, key=lambda x: self.nodes_by_id[x].profit, reverse=True):
                if not (self.conflict_adj.get(nid, set()) & repaired):
                    repaired.add(nid)
            node_ids = repaired
        return self._make_solution(node_ids)

    def _distance(self, a: Solution, b: Solution) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a.objectives, b.objectives)))

    def _fitness_values(self, solutions: Sequence[Solution]) -> list[float]:
        n = len(solutions)
        if n == 0:
            return []

        strength = [0.0] * n
        raw = [0.0] * n
        for i in range(n):
            for j in range(n):
                if i != j and dominates(solutions[i].objectives, solutions[j].objectives):
                    strength[i] += 1.0
        for i in range(n):
            raw[i] = sum(
                strength[j]
                for j in range(n)
                if j != i and dominates(solutions[j].objectives, solutions[i].objectives)
            )

        k = max(1, int(math.sqrt(n)))
        fitness = []
        for i in range(n):
            distances = sorted(
                self._distance(solutions[i], solutions[j])
                for j in range(n)
                if j != i
            )
            sigma = distances[min(k - 1, len(distances) - 1)] if distances else 1.0
            density = 1.0 / (sigma + 2.0)
            fitness.append(raw[i] + density)
        return fitness

    def _environmental_selection(
        self,
        positions: Sequence[Sequence[float]],
        solutions: Sequence[Solution],
    ) -> tuple[list[list[float]], list[Solution]]:
        archive_size = int(self.params["archive_size"])
        fitness = self._fitness_values(solutions)

        selected_idx = [i for i, fit in enumerate(fitness) if fit < 1.0]
        if len(selected_idx) < archive_size:
            remaining = [i for i in range(len(solutions)) if i not in selected_idx]
            remaining.sort(key=lambda i: fitness[i])
            selected_idx.extend(remaining[: archive_size - len(selected_idx)])

        selected_idx = selected_idx[:]
        while len(selected_idx) > archive_size:
            distances_by_idx = []
            for idx in selected_idx:
                distances = sorted(
                    self._distance(solutions[idx], solutions[j])
                    for j in selected_idx
                    if j != idx
                )
                distances_by_idx.append((distances[0] if distances else 0.0, idx))
            remove_idx = min(distances_by_idx)[1]
            selected_idx.remove(remove_idx)

        archive_positions = [list(positions[i]) for i in selected_idx[:archive_size]]
        archive_solutions = [solutions[i] for i in selected_idx[:archive_size]]

        archive_solutions = update_archive([], archive_solutions, archive_size)
        keep_keys = {sol.node_ids for sol in archive_solutions}
        archive_positions = [
            pos
            for pos, sol in zip([list(positions[i]) for i in selected_idx[:archive_size]], [solutions[i] for i in selected_idx[:archive_size]])
            if sol.node_ids in keep_keys
        ][: len(archive_solutions)]
        while len(archive_positions) < len(archive_solutions):
            archive_positions.append(self._random_position())

        return archive_positions, archive_solutions

    def _tournament_select(self, positions: Sequence[Sequence[float]], solutions: Sequence[Solution]) -> list[float]:
        k = min(int(self.params.get("tournament_size", 2)), len(solutions))
        candidates = random.sample(range(len(solutions)), k)
        fitness = self._fitness_values([solutions[i] for i in candidates])
        best_local = min(range(len(candidates)), key=lambda i: fitness[i])
        return list(positions[candidates[best_local]])

    def _crossover(self, a: Sequence[float], b: Sequence[float]) -> tuple[list[float], list[float]]:
        if random.random() > float(self.params["crossover_probability"]):
            return list(a), list(b)
        child_a = []
        child_b = []
        for x, y in zip(a, b):
            alpha = random.random()
            child_a.append(alpha * x + (1.0 - alpha) * y)
            child_b.append(alpha * y + (1.0 - alpha) * x)
        return self._clip_position(child_a), self._clip_position(child_b)

    def _mutate(self, position: Sequence[float]) -> list[float]:
        probability = self.params.get("mutation_probability")
        if probability is None:
            probability = 1.0 / max(1, self.dimension)
        strength = float(self.params.get("mutation_strength", 0.10))
        lb = float(self.params["lower_bound"])
        ub = float(self.params["upper_bound"])
        span = ub - lb
        out = list(position)
        for i in range(self.dimension):
            if random.random() < float(probability):
                out[i] = self._clip_value(out[i] + random.uniform(-strength, strength) * span)
        return out

    def _make_offspring(self) -> list[list[float]]:
        source_positions = self.archive_positions or self.population
        source_solutions = self.archive or self.population_solutions
        pop_size = int(self.params["pop_size"])
        offspring: list[list[float]] = []
        while len(offspring) < pop_size:
            p1 = self._tournament_select(source_positions, source_solutions)
            p2 = self._tournament_select(source_positions, source_solutions)
            c1, c2 = self._crossover(p1, p2)
            offspring.append(self._mutate(c1))
            if len(offspring) < pop_size:
                offspring.append(self._mutate(c2))
        return offspring

    def _initialize(self) -> None:
        pop_size = int(self.params["pop_size"])
        self.population = [self._random_position() for _ in range(pop_size)]
        self.population_solutions = [self._evaluate_position(pos) for pos in self.population]
        self.archive_positions, self.archive = self._environmental_selection(
            self.population,
            self.population_solutions,
        )

    def run(self) -> List[Solution]:
        start_time = time.perf_counter()
        self._initialize()

        max_iter = int(self.params["max_iter"])
        verbose = bool(self.params.get("verbose", True))

        for iteration in range(1, max_iter + 1):
            offspring = self._make_offspring()
            offspring_solutions = [self._evaluate_position(pos) for pos in offspring]

            combined_positions = self.population + self.archive_positions + offspring
            combined_solutions = self.population_solutions + self.archive + offspring_solutions

            self.archive_positions, self.archive = self._environmental_selection(
                combined_positions,
                combined_solutions,
            )

            self.population = offspring
            self.population_solutions = offspring_solutions

            if verbose and (
                iteration == 1
                or iteration % max(1, max_iter // 10) == 0
                or iteration == max_iter
            ):
                best_f1 = min((s.objectives[0] for s in self.archive), default=float("nan"))
                print(
                    f"[SPEA2] Iter {iteration:>4}/{max_iter}: "
                    f"archive={len(self.archive):>3}, best_f1={best_f1:.4f}"
                )

        self.runtime_seconds = time.perf_counter() - start_time
        return self.archive

    def save_archive_csv(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "solution_index",
                "f1_uncompleted_profit_ratio",
                "f2_maneuver_cost",
                "f3_load_imbalance",
                "scheduled_nodes",
                "scheduled_tasks",
                "task_completion_rate",
                "node_ids",
            ])
            total_task_count = len(self.tasks)
            for idx, sol in enumerate(self.archive):
                scheduled_tasks = {self.nodes_by_id[nid].task_id for nid in sol.node_ids}
                writer.writerow([
                    idx,
                    sol.objectives[0],
                    sol.objectives[1],
                    sol.objectives[2],
                    len(sol.node_ids),
                    len(scheduled_tasks),
                    task_completion_rate(sol.node_ids, total_task_count, self.nodes_by_id),
                    " ".join(map(str, sorted(sol.node_ids))),
                ])

