"""Main CG-MOACO algorithm.

CG-MOACO = Conflict-Graph-Guided Multiobjective Ant Colony Optimization.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import time
from typing import Dict, Iterable, List, Sequence

from .domain import CandidateNode, Task
from .local_search import local_search
from .problem_model import evaluate_solution, is_feasible, task_completion_rate
from .utils import Solution, crowding_distance, roulette_select, set_random_seed, update_archive


DEFAULT_PARAMS = {
    "num_ants": 50,
    "max_iter": 100,
    "archive_size": 100,
    "alpha": 1.0,
    "beta": 2.0,
    "rho": 0.1,
    "q": 0.2,
    "tau0": 1.0,
    "tau_min": 1e-6,
    "tau_max": 50.0,
    "lambda_scarcity": 1.0,
    "lambda_conflict": 1.0,
    "lambda_maneuver": 1.0,
    "lambda_load": 1.0,
    "min_transition_time": 5.0,
    "maneuver_time_per_degree": 0.2,
    "load_mode": "task_count",
    "enable_local_search": True,
    "enable_fast_insert": True,
    "enable_replacement": True,
    "use_graph_pheromone": True,
    "replacement_attempts": 100,
    "max_replace_conflicts": 3,
    "min_replacement_profit_gain": 0.0,
    "w_profit": 1.0,
    "w_maneuver": 0.01,
    "w_load": 1.0,
    "seed": 42,
    "verbose": True,
}


class CGMOACO:
    def __init__(
        self,
        tasks: Dict[int, Task],
        nodes: List[CandidateNode],
        conflict_adj: dict[int, set[int]],
        graph_features: dict[str, dict[int, float]],
        params: dict | None = None,
        satellite_ids: Sequence[int] | None = None,
    ) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        set_random_seed(self.params.get("seed"))
        self.tasks = tasks
        self.nodes = nodes
        self.nodes_by_id = {node.node_id: node for node in nodes}
        self.conflict_adj = conflict_adj
        self.graph_features = graph_features
        self.satellite_ids = sorted(
            set(satellite_ids) if satellite_ids is not None else {node.sat_id for node in nodes}
        )
        self.pheromone = {node.node_id: float(self.params["tau0"]) for node in nodes}
        self.archive: List[Solution] = []
        self.runtime_seconds = 0.0

    def _dynamic_heuristic(self, node_id: int, current_solution: set[int], sat_loads: dict[int, int]) -> float:
        node = self.nodes_by_id[node_id]
        max_load = max(1, max(sat_loads.values()) if sat_loads else 1)
        norm_load = sat_loads.get(node.sat_id, 0) / max_load
        numerator = (
            self.graph_features["norm_profit"][node_id]
            + self.params["lambda_scarcity"] * self.graph_features["norm_scarcity"][node_id]
        )
        denominator = (
            1.0
            + self.params["lambda_conflict"] * self.graph_features["norm_conflict"][node_id]
            + self.params["lambda_maneuver"] * self.graph_features["norm_maneuver"][node_id]
            + self.params["lambda_load"] * norm_load
        )
        return max(1e-9, numerator / denominator)

    def construct_solution(self) -> set[int]:
        """Construct one schedule by choosing non-conflicting graph nodes."""

        solution: set[int] = set()
        available: set[int] = {node.node_id for node in self.nodes}
        sat_loads = {sat_id: 0 for sat_id in self.satellite_ids}

        while available:
            items = list(available)
            weights = []
            for node_id in items:
                tau = max(self.params["tau_min"], self.pheromone[node_id])
                eta = self._dynamic_heuristic(node_id, solution, sat_loads)
                weights.append((tau ** self.params["alpha"]) * (eta ** self.params["beta"]))

            chosen = roulette_select(items, weights)
            if not (self.conflict_adj.get(chosen, set()) & solution):
                solution.add(chosen)
                sat_loads[self.nodes_by_id[chosen].sat_id] += 1
                # Remove chosen node and everything incompatible with it.
                available.discard(chosen)
                available.difference_update(self.conflict_adj.get(chosen, set()))
            else:
                available.discard(chosen)
        return solution

    def _make_solution(self, node_ids: Iterable[int]) -> Solution:
        frozen = frozenset(node_ids)
        objectives = evaluate_solution(frozen, self.nodes_by_id, self.tasks, self.satellite_ids, self.params)
        return Solution(node_ids=frozen, objectives=objectives)

    def _evaporate_pheromone(self) -> None:
        rho = self.params["rho"]
        tau_min = self.params["tau_min"]
        for node_id in self.pheromone:
            self.pheromone[node_id] = max(tau_min, (1.0 - rho) * self.pheromone[node_id])

    def _update_pheromone_by_archive(self) -> None:
        if not self.archive:
            return
        q = self.params["q"]
        tau_max = self.params["tau_max"]
        cd = crowding_distance(self.archive)
        finite = [v for v in cd.values() if math.isfinite(v)]
        max_cd = max(finite) if finite else 1.0

        for idx, sol in enumerate(self.archive):
            diversity_weight = 1.0
            if math.isfinite(cd.get(idx, 0.0)) and max_cd > 0:
                diversity_weight += cd[idx] / max_cd
            elif math.isinf(cd.get(idx, 0.0)):
                diversity_weight += 1.0
            for node_id in sol.node_ids:
                if self.params.get("use_graph_pheromone", True):
                    graph_contribution = self.graph_features["contribution"].get(node_id, 1.0)
                else:
                    graph_contribution = 1.0
                self.pheromone[node_id] = min(
                    tau_max,
                    self.pheromone[node_id] + q * diversity_weight * graph_contribution,
                )

    def run(self) -> List[Solution]:
        start_time = time.perf_counter()
        max_iter = int(self.params["max_iter"])
        num_ants = int(self.params["num_ants"])
        archive_size = int(self.params["archive_size"])
        verbose = bool(self.params.get("verbose", True))

        for iteration in range(1, max_iter + 1):
            population: List[Solution] = []
            for _ in range(num_ants):
                constructed = self.construct_solution()
                improved = local_search(
                    constructed,
                    self.nodes,
                    self.nodes_by_id,
                    self.conflict_adj,
                    self.tasks,
                    self.satellite_ids,
                    self.graph_features,
                    self.params,
                )
                # Constructed and local-search solutions should be feasible by design.
                # If not, skip malformed solution rather than poisoning the archive.
                if is_feasible(improved, self.conflict_adj):
                    population.append(self._make_solution(improved))

            self.archive = update_archive(self.archive, population, archive_size)
            self._evaporate_pheromone()
            self._update_pheromone_by_archive()

            if verbose and (iteration == 1 or iteration % max(1, max_iter // 10) == 0 or iteration == max_iter):
                best_profit_ratio = min((s.objectives[0] for s in self.archive), default=float("nan"))
                print(
                    f"Iter {iteration:>4}/{max_iter}: archive={len(self.archive):>3}, "
                    f"best_f1={best_profit_ratio:.4f}"
                )

        self.runtime_seconds = time.perf_counter() - start_time
        return self.archive

    def save_archive_csv(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "solution_index", "f1_uncompleted_profit_ratio", "f2_maneuver_cost", "f3_load_imbalance",
                "scheduled_nodes", "scheduled_tasks", "task_completion_rate", "node_ids",
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
