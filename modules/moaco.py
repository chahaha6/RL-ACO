
from __future__ import annotations

import csv
import random
from pathlib import Path
import time
from typing import Dict, Iterable, List, Sequence

from .domain import CandidateNode, Task
from .problem_model import (
    evaluate_solution,
    estimate_transition_time,
    node_load_amount,
    task_completion_rate,
)
from .utils import (
    Solution,
    normalize_dict,
    observation_fits_window,
    roulette_select,
    set_random_seed,
    time_not_after,
    update_archive,
)


DEFAULT_PARAMS = {
    "num_ants": 50,
    "max_iter": 300,
    "archive_size": 100,
    "alpha": 0.7,
    "beta": 1.4,
    "rho": 0.15,
    "q": 0.1,
    "tau0": 1.0,
    "tau_min": 1e-6,
    "tau_max": 30.0,
    "candidate_pool_size": 200,
    "candidate_random_ratio": 0.1,
    "lambda_load": 1.0,
    "load_mode": "task_count",
    "validate_each_solution": False,
    "validate_interval": 20,
    "validate_final_archive": True,
    "seed": 42,
    "verbose": True,
}


class MOACO:
    """Plain MOACO baseline using direct scheduling-constraint checks."""

    def __init__(
        self,
        tasks: Dict[int, Task],
        nodes: List[CandidateNode],
        params: dict | None = None,
        satellite_ids: Sequence[int] | None = None,
    ) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        set_random_seed(self.params.get("seed"))
        self.tasks = tasks
        self.nodes = nodes
        self.nodes_by_id = {node.node_id: node for node in nodes}
        self.node_ids_by_task: dict[int, set[int]] = {}
        for node in nodes:
            self.node_ids_by_task.setdefault(node.task_id, set()).add(node.node_id)
        self.satellite_ids = sorted(
            set(satellite_ids) if satellite_ids is not None else {node.sat_id for node in nodes}
        )
        self.pheromone = {node.node_id: float(self.params["tau0"]) for node in nodes}
        self.archive: List[Solution] = []
        self.runtime_seconds = 0.0
        self._lambda_load = float(self.params.get("lambda_load", 1.0))
        self._load_mode = str(self.params.get("load_mode", "task_count"))
        if self._load_mode not in {"task_count", "duration"}:
            raise ValueError(f"Unknown load_mode: {self._load_mode}")
        self.simple_heuristic = self._build_simple_heuristic()
        self._candidate_rank = sorted(
            (node.node_id for node in nodes),
            key=lambda node_id: self.simple_heuristic.get(node_id, 0.0),
            reverse=True,
        )

    def _build_simple_heuristic(self) -> dict[int, float]:
        """Build a simple non-graph heuristic for baseline MOACO.

        The heuristic uses only task profit and a rough maneuver pressure, but
        intentionally excludes conflict-degree and window-scarcity information.
        """

        profit = {node.node_id: node.profit for node in self.nodes}
        norm_profit = normalize_dict(profit, higher_is_better=True)

        # Rough static maneuver pressure: average transition cost to a few
        # temporally-near same-satellite nodes. This is not a conflict-graph
        # feature; it only reflects the second objective's cost tendency.
        by_sat: dict[int, list[CandidateNode]] = {}
        for node in self.nodes:
            by_sat.setdefault(node.sat_id, []).append(node)

        maneuver_pressure: dict[int, float] = {}
        for node in self.nodes:
            related = by_sat.get(node.sat_id, [])
            if len(related) <= 1:
                maneuver_pressure[node.node_id] = 0.0
                continue
            candidates = sorted(related, key=lambda n: abs(n.start - node.start))[:30]
            vals = [estimate_transition_time(node, other, self.params) for other in candidates if other.node_id != node.node_id]
            maneuver_pressure[node.node_id] = sum(vals) / len(vals) if vals else 0.0

        # For maneuver, lower is better. Convert to a [0, 1] benefit.
        maneuver_benefit = normalize_dict(maneuver_pressure, higher_is_better=False)

        heuristic: dict[int, float] = {}
        for node in self.nodes:
            nid = node.node_id
            # Simple baseline: reward high profit and low maneuver pressure.
            heuristic[nid] = max(1e-9, 0.7 * norm_profit.get(nid, 1.0) + 0.3 * maneuver_benefit.get(nid, 1.0))
        return heuristic

    def _candidate_pool(self, available: set[int]) -> list[int]:
        """Return a simple-heuristic candidate pool for baseline MOACO."""

        pool_size = int(self.params.get("candidate_pool_size", 0))
        if pool_size <= 0 or len(available) <= pool_size:
            return list(available)

        random_ratio = float(self.params.get("candidate_random_ratio", 0.0))
        random_ratio = max(0.0, min(1.0, random_ratio))
        random_count = int(round(pool_size * random_ratio))
        random_count = min(max(0, random_count), pool_size - 1)
        top_count = pool_size - random_count

        pool: list[int] = []
        for node_id in self._candidate_rank:
            if node_id in available:
                pool.append(node_id)
                if len(pool) >= top_count:
                    break

        if random_count > 0 and len(pool) < pool_size:
            pool_set = set(pool)
            # Uniform sampling does not depend on list order. Avoid repeatedly
            # sorting the full remaining node set during solution construction.
            remaining = [node_id for node_id in available if node_id not in pool_set]
            sample_count = min(random_count, pool_size - len(pool), len(remaining))
            if sample_count > 0:
                pool.extend(random.sample(remaining, sample_count))

        return pool or list(available)

    def _dynamic_heuristic(
        self,
        node_id: int,
        sat_loads: dict[int, float],
        max_load: float | None = None,
    ) -> float:
        """Apply a dynamic satellite-load penalty to the base heuristic."""

        node = self.nodes_by_id[node_id]
        if max_load is None:
            max_load = max(1.0, max(sat_loads.values()) if sat_loads else 1.0)
        norm_load = sat_loads.get(node.sat_id, 0.0) / max_load
        base_eta = self.simple_heuristic.get(node_id, 1.0)
        return max(1e-9, base_eta / (1.0 + self._lambda_load * norm_load))

    @staticmethod
    def _insertion_index(
        sequence: list[CandidateNode],
        node: CandidateNode,
    ) -> int:
        """Return the chronological insertion position without graph lookups."""

        node_key = (node.start, node.node_id)
        low = 0
        high = len(sequence)
        while low < high:
            middle = (low + high) // 2
            middle_node = sequence[middle]
            if (middle_node.start, middle_node.node_id) < node_key:
                low = middle + 1
            else:
                high = middle
        return low

    def _can_insert_node(
        self,
        node: CandidateNode,
        selected_tasks: set[int],
        sat_sequences: dict[int, list[CandidateNode]],
    ) -> bool:
        """Check task, window, overlap, and transition constraints directly."""

        if node.task_id in selected_tasks:
            return False

        node_end = node.start + node.task_duration
        if not observation_fits_window(node.start, node.end, node.task_duration):
            return False

        sequence = sat_sequences.get(node.sat_id, [])
        position = self._insertion_index(sequence, node)

        if position > 0:
            previous = sequence[position - 1]
            previous_end = previous.start + previous.task_duration
            transition = estimate_transition_time(previous, node, self.params)
            if not time_not_after(previous_end + transition, node.start):
                return False

        if position < len(sequence):
            following = sequence[position]
            transition = estimate_transition_time(node, following, self.params)
            if not time_not_after(node_end + transition, following.start):
                return False

        return True

    def _insert_node(
        self,
        node: CandidateNode,
        sat_sequences: dict[int, list[CandidateNode]],
    ) -> None:
        sequence = sat_sequences.setdefault(node.sat_id, [])
        sequence.insert(self._insertion_index(sequence, node), node)

    def _is_solution_feasible_direct(self, node_ids: Iterable[int]) -> bool:
        """Validate a complete solution without a precomputed conflict graph."""

        selected_tasks: set[int] = set()
        sat_sequences: dict[int, list[CandidateNode]] = {
            sat_id: [] for sat_id in self.satellite_ids
        }
        ordered_nodes = sorted(
            (self.nodes_by_id[node_id] for node_id in node_ids),
            key=lambda node: (node.start, node.node_id),
        )
        for node in ordered_nodes:
            if not self._can_insert_node(node, selected_tasks, sat_sequences):
                return False
            selected_tasks.add(node.task_id)
            self._insert_node(node, sat_sequences)
        return True

    def construct_solution(self) -> set[int]:
        """Construct one feasible schedule using direct constraint checks."""

        solution: set[int] = set()
        available: set[int] = {node.node_id for node in self.nodes}
        selected_tasks: set[int] = set()
        sat_sequences: dict[int, list[CandidateNode]] = {
            sat_id: [] for sat_id in self.satellite_ids
        }
        sat_loads = {sat_id: 0.0 for sat_id in self.satellite_ids}

        alpha = float(self.params["alpha"])
        beta = float(self.params["beta"])
        tau_min = float(self.params["tau_min"])

        while available:
            items = self._candidate_pool(available)
            weights = []
            max_load = max(1.0, max(sat_loads.values()) if sat_loads else 1.0)
            for node_id in items:
                tau = max(tau_min, self.pheromone[node_id])
                eta = self._dynamic_heuristic(node_id, sat_loads, max_load)
                weights.append((tau ** alpha) * (eta ** beta))

            chosen = roulette_select(items, weights)
            node = self.nodes_by_id[chosen]
            if self._can_insert_node(node, selected_tasks, sat_sequences):
                solution.add(chosen)
                selected_tasks.add(node.task_id)
                self._insert_node(node, sat_sequences)
                sat_loads[node.sat_id] += node_load_amount(node, self._load_mode)
                available.difference_update(self.node_ids_by_task[node.task_id])
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
        """Uniform Pareto archive pheromone update for baseline MOACO."""

        if not self.archive:
            return
        q = self.params["q"]
        tau_max = self.params["tau_max"]
        for sol in self.archive:
            for node_id in sol.node_ids:
                self.pheromone[node_id] = min(tau_max, self.pheromone[node_id] + q)

    def run(self) -> List[Solution]:
        start_time = time.perf_counter()
        max_iter = int(self.params["max_iter"])
        num_ants = int(self.params["num_ants"])
        archive_size = int(self.params["archive_size"])
        verbose = bool(self.params.get("verbose", True))
        validate_each_solution = bool(self.params.get("validate_each_solution", False))
        validate_interval = int(self.params.get("validate_interval", 20))
        validate_final_archive = bool(self.params.get("validate_final_archive", True))

        for iteration in range(1, max_iter + 1):
            population: List[Solution] = []
            for _ in range(num_ants):
                constructed = self.construct_solution()
                if validate_each_solution and not self._is_solution_feasible_direct(constructed):
                    continue
                population.append(self._make_solution(constructed))

            if validate_interval > 0 and iteration % validate_interval == 0:
                for idx, sol in enumerate(population):
                    if not self._is_solution_feasible_direct(sol.node_ids):
                        raise RuntimeError(
                            "MOACO produced infeasible solution at "
                            f"iteration {iteration}, population index {idx}"
                        )

            self.archive = update_archive(self.archive, population, archive_size)
            self._evaporate_pheromone()
            self._update_pheromone_by_archive()

            if verbose and (iteration == 1 or iteration % max(1, max_iter // 10) == 0 or iteration == max_iter):
                best_total_profit = max((-s.objectives[0] for s in self.archive), default=float("nan"))
                print(
                    f"[MOACO] Iter {iteration:>4}/{max_iter}: archive={len(self.archive):>3}, "
                    f"best_f1={best_total_profit:.4f}"
                )

        if validate_final_archive:
            for idx, sol in enumerate(self.archive):
                if not self._is_solution_feasible_direct(sol.node_ids):
                    raise RuntimeError(
                        f"MOACO final archive contains infeasible solution at index {idx}"
                    )

        self.runtime_seconds = time.perf_counter() - start_time
        return self.archive

    def save_archive_csv(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "solution_index", "f1_total_profit", "f2_maneuver_cost", "f3_load_imbalance",
                "scheduled_nodes", "scheduled_tasks", "task_completion_rate", "node_ids",
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
