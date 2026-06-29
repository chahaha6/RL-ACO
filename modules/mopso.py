"""MOPSO baseline for the satellite scheduling problem."""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .domain import CandidateNode, Task
from .problem_model import DEFAULT_MODEL_PARAMS, evaluate_solution, is_feasible, task_completion_rate
from .utils import Solution, dominates, set_random_seed, update_archive


DEFAULT_PARAMS = {
    "pop_size": 50,
    "max_iter": 300,
    "archive_size": 100,
    "lower_bound": 0.0,
    "upper_bound": 0.8,
    "inertia": 0.45,
    "c1": 1.5,
    "c2": 1.5,
    "vmax": 0.20,
    "mutation_probability": 0.03,
    "mutation_strength": 0.10,
    "seed": 42,
    "verbose": True,
}


class MOPSO:
    """Random-key MOPSO with greedy conflict-graph decoding."""

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

        self.positions: list[list[float]] = []
        self.velocities: list[list[float]] = []
        self.solutions: list[Solution] = []
        self.best_positions: list[list[float]] = []
        self.best_solutions: list[Solution] = []
        self.archive: List[Solution] = []
        self.archive_positions: dict[frozenset[int], list[float]] = {}
        self.runtime_seconds = 0.0

    def _clip_value(self, value: float) -> float:
        return max(float(self.params["lower_bound"]), min(float(self.params["upper_bound"]), value))

    def _random_position(self) -> list[float]:
        lb = float(self.params["lower_bound"])
        ub = float(self.params["upper_bound"])
        return [random.uniform(lb, ub) for _ in range(self.dimension)]

    def _random_velocity(self) -> list[float]:
        vmax = float(self.params["vmax"])
        return [random.uniform(-vmax, vmax) for _ in range(self.dimension)]

    def _clip_position(self, position: Sequence[float]) -> list[float]:
        return [self._clip_value(v) for v in position]

    def _mutate_position(self, position: list[float]) -> list[float]:
        probability = float(self.params.get("mutation_probability", 0.03))
        strength = float(self.params.get("mutation_strength", 0.10))
        if probability <= 0.0 or strength <= 0.0:
            return position

        lb = float(self.params["lower_bound"])
        ub = float(self.params["upper_bound"])
        span = ub - lb
        out = position[:]
        for i in range(self.dimension):
            if random.random() < probability:
                out[i] = self._clip_value(out[i] + random.uniform(-strength, strength) * span)
        return out

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

    def _select_global_best_position(self) -> list[float]:
        if self.archive:
            sol = random.choice(self.archive)
            pos = self.archive_positions.get(sol.node_ids)
            if pos is not None:
                return pos[:]
        return self._random_position()

    def _update_archive_with_positions(
        self,
        solutions: Sequence[Solution],
        positions: Sequence[Sequence[float]],
    ) -> None:
        self.archive = update_archive(
            self.archive,
            solutions,
            int(self.params["archive_size"]),
        )
        archive_keys = {sol.node_ids for sol in self.archive}
        for sol, pos in zip(solutions, positions):
            if sol.node_ids in archive_keys:
                self.archive_positions[sol.node_ids] = list(pos)
        for sol in self.archive:
            if sol.node_ids not in self.archive_positions:
                self.archive_positions[sol.node_ids] = self._random_position()
        self.archive_positions = {
            key: value
            for key, value in self.archive_positions.items()
            if key in archive_keys
        }

    def _initialize(self) -> None:
        pop_size = int(self.params["pop_size"])
        self.positions = [self._random_position() for _ in range(pop_size)]
        self.velocities = [self._random_velocity() for _ in range(pop_size)]
        self.solutions = [self._evaluate_position(pos) for pos in self.positions]
        self.best_positions = [pos[:] for pos in self.positions]
        self.best_solutions = list(self.solutions)
        self._update_archive_with_positions(self.solutions, self.positions)

    def _accept_personal_best(self, old: Solution, new: Solution) -> bool:
        if dominates(new.objectives, old.objectives):
            return True
        if dominates(old.objectives, new.objectives):
            return False
        return random.random() < 0.5

    def run(self) -> List[Solution]:
        start_time = time.perf_counter()
        self._initialize()

        max_iter = int(self.params["max_iter"])
        verbose = bool(self.params.get("verbose", True))
        inertia = float(self.params["inertia"])
        c1 = float(self.params["c1"])
        c2 = float(self.params["c2"])
        vmax = float(self.params["vmax"])

        for iteration in range(1, max_iter + 1):
            new_positions: list[list[float]] = []
            new_velocities: list[list[float]] = []
            new_solutions: list[Solution] = []

            for i, position in enumerate(self.positions):
                gbest = self._select_global_best_position()
                velocity = []
                next_position = []
                for d, x in enumerate(position):
                    v = (
                        inertia * self.velocities[i][d]
                        + c1 * random.random() * (self.best_positions[i][d] - x)
                        + c2 * random.random() * (gbest[d] - x)
                    )
                    v = max(-vmax, min(vmax, v))
                    velocity.append(v)
                    next_position.append(x + v)

                next_position = self._mutate_position(self._clip_position(next_position))
                sol = self._evaluate_position(next_position)

                if self._accept_personal_best(self.best_solutions[i], sol):
                    self.best_positions[i] = next_position[:]
                    self.best_solutions[i] = sol

                new_positions.append(next_position)
                new_velocities.append(velocity)
                new_solutions.append(sol)

            self.positions = new_positions
            self.velocities = new_velocities
            self.solutions = new_solutions
            self._update_archive_with_positions(self.solutions, self.positions)

            if verbose and (
                iteration == 1
                or iteration % max(1, max_iter // 10) == 0
                or iteration == max_iter
            ):
                best_f1 = max((-s.objectives[0] for s in self.archive), default=float("nan"))
                print(
                    f"[MOPSO] Iter {iteration:>4}/{max_iter}: "
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
                "f1_total_profit",
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
                    -sol.objectives[0],
                    sol.objectives[1],
                    sol.objectives[2],
                    len(sol.node_ids),
                    len(scheduled_tasks),
                    task_completion_rate(sol.node_ids, total_task_count, self.nodes_by_id),
                    " ".join(map(str, sorted(sol.node_ids))),
                ])
