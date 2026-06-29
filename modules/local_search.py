"""Conflict-graph-aware fast insertion and replacement local search."""

from __future__ import annotations

from heapq import nlargest
from typing import Dict, Iterable, List, Sequence

from .domain import CandidateNode, Task
from .problem_model import evaluate_solution
from .utils import dominates


def _selected_tasks(solution: Iterable[int], nodes_by_id: Dict[int, CandidateNode]) -> set[int]:
    return {nodes_by_id[node_id].task_id for node_id in solution}


def _is_node_feasible_for_solution(node_id: int, solution: set[int], conflict_adj: dict[int, set[int]]) -> bool:
    return not (conflict_adj.get(node_id, set()) & solution)


def _node_load_amount(node: CandidateNode, params: dict) -> float:
    """Return the load contribution of one scheduled node.

    The definition is kept consistent with problem_model.calculate_load_balance:
    - task_count: each scheduled task contributes 1 load unit;
    - duration: each scheduled task contributes its actual observation duration.
    """

    load_mode = params.get("load_mode", "task_count")

    if load_mode == "task_count":
        return 1.0
    if load_mode == "duration":
        return float(node.task_duration)

    raise ValueError(f"Unknown load_mode: {load_mode}")


def _sat_loads(
    solution: Iterable[int],
    nodes_by_id: Dict[int, CandidateNode],
    params: dict,
) -> dict[int, float]:
    loads: dict[int, float] = {}
    for node_id in solution:
        node = nodes_by_id[node_id]
        loads[node.sat_id] = loads.get(node.sat_id, 0.0) + _node_load_amount(node, params)
    return loads


def _dynamic_insert_score(
    node: CandidateNode,
    sat_loads: dict[int, float],
    total_solution_load: float,
    graph_features: dict[str, dict[int, float]],
    params: dict,
) -> float:
    sat_load = sat_loads.get(node.sat_id, 0.0)
    max_load = max(1.0, total_solution_load)
    norm_load = sat_load / max_load
    nid = node.node_id
    numerator = graph_features["norm_profit"][nid] + params.get("lambda_scarcity", 1.0) * graph_features["norm_scarcity"][nid]
    denominator = (
        1.0
        + params.get("lambda_conflict", 1.0) * graph_features["norm_conflict"][nid]
        + params.get("lambda_maneuver", 1.0) * graph_features["norm_maneuver"][nid]
        + params.get("lambda_load", 1.0) * norm_load
    )
    return max(1e-9, numerator / denominator)


def _rank_candidate_nodes(
    candidate_nodes: list[CandidateNode],
    limit: int,
    sat_loads: dict[int, float],
    total_solution_load: float,
    graph_features: dict[str, dict[int, float]],
    params: dict,
) -> list[CandidateNode]:
    def score(node: CandidateNode) -> float:
        return _dynamic_insert_score(
            node,
            sat_loads,
            total_solution_load,
            graph_features,
            params,
        )

    if limit > 0 and len(candidate_nodes) > limit:
        return nlargest(limit, candidate_nodes, key=score)
    return sorted(candidate_nodes, key=score, reverse=True)


def conflict_aware_fast_insert(
    solution: set[int],
    nodes: List[CandidateNode],
    nodes_by_id: Dict[int, CandidateNode],
    conflict_adj: dict[int, set[int]],
    graph_features: dict[str, dict[int, float]],
    params: dict,
) -> set[int]:
    """Try to directly insert unscheduled tasks using graph-aware scores."""

    improved = set(solution)
    scheduled_tasks = _selected_tasks(improved, nodes_by_id)
    sat_loads = _sat_loads(improved, nodes_by_id, params)
    total_solution_load = sum(sat_loads.values())
    candidate_nodes = [node for node in nodes if node.task_id not in scheduled_tasks]
    candidate_nodes = _rank_candidate_nodes(
        candidate_nodes,
        int(params.get("local_search_candidate_limit", 0)),
        sat_loads,
        total_solution_load,
        graph_features,
        params,
    )

    for node in candidate_nodes:
        if node.task_id in scheduled_tasks:
            continue
        if _is_node_feasible_for_solution(node.node_id, improved, conflict_adj):
            node_load = _node_load_amount(node, params)
            improved.add(node.node_id)
            scheduled_tasks.add(node.task_id)
            sat_loads[node.sat_id] = sat_loads.get(node.sat_id, 0.0) + node_load
            total_solution_load += node_load
    return improved


def conflict_aware_replacement(
    solution: set[int],
    nodes: List[CandidateNode],
    nodes_by_id: Dict[int, CandidateNode],
    conflict_adj: dict[int, set[int]],
    tasks: Dict[int, Task],
    satellite_ids: Sequence[int],
    graph_features: dict[str, dict[int, float]],
    params: dict,
) -> set[int]:
    """Try replacing low-value conflict neighbors with high-value unscheduled nodes."""

    improved = set(solution)
    scheduled_tasks = _selected_tasks(improved, nodes_by_id)
    old_obj = evaluate_solution(improved, nodes_by_id, tasks, satellite_ids, params)
    sat_loads = _sat_loads(improved, nodes_by_id, params)
    total_solution_load = sum(sat_loads.values())

    unscheduled_nodes = [node for node in nodes if node.task_id not in scheduled_tasks]
    candidate_limit = int(
        params.get(
            "replacement_candidate_limit",
            params.get("local_search_candidate_limit", 0),
        )
    )
    unscheduled_nodes = _rank_candidate_nodes(
        unscheduled_nodes,
        candidate_limit,
        sat_loads,
        total_solution_load,
        graph_features,
        params,
    )

    max_attempts = params.get("replacement_attempts", 100)
    attempts = 0
    for node in unscheduled_nodes:
        if attempts >= max_attempts:
            break
        attempts += 1
        if node.task_id in scheduled_tasks:
            continue
        conflict_set = conflict_adj.get(node.node_id, set()) & improved
        if not conflict_set:
            continue
        # Avoid deleting too many tasks for one insertion.
        if len(conflict_set) > params.get("max_replace_conflicts", 3):
            continue

        removed_profit = sum(nodes_by_id[nid].profit for nid in conflict_set)
        profit_gain = node.profit - removed_profit
        if profit_gain < params.get("min_replacement_profit_gain", 0.0):
            continue

        trial = set(improved)
        trial.difference_update(conflict_set)
        if _is_node_feasible_for_solution(node.node_id, trial, conflict_adj):
            trial.add(node.node_id)
        else:
            continue

        new_obj = evaluate_solution(trial, nodes_by_id, tasks, satellite_ids, params)
        # Accept if Pareto-better, or if weighted scalar gain improves.
        old_scalar = params.get("w_profit", 1.0) * old_obj[0] + params.get("w_maneuver", 0.01) * old_obj[1] + params.get("w_load", 1.0) * old_obj[2]
        new_scalar = params.get("w_profit", 1.0) * new_obj[0] + params.get("w_maneuver", 0.01) * new_obj[1] + params.get("w_load", 1.0) * new_obj[2]
        if dominates(new_obj, old_obj) or new_scalar < old_scalar:
            improved = trial
            scheduled_tasks = _selected_tasks(improved, nodes_by_id)
            sat_loads = _sat_loads(improved, nodes_by_id, params)
            total_solution_load = sum(sat_loads.values())
            old_obj = new_obj
    return improved


def local_search(
    solution: set[int],
    nodes: List[CandidateNode],
    nodes_by_id: Dict[int, CandidateNode],
    conflict_adj: dict[int, set[int]],
    tasks: Dict[int, Task],
    satellite_ids: Sequence[int],
    graph_features: dict[str, dict[int, float]],
    params: dict,
) -> set[int]:
    """Run fast insertion followed by replacement."""

    if not params.get("enable_local_search", True):
        return set(solution)
    improved = set(solution)
    if params.get("enable_fast_insert", True):
        improved = conflict_aware_fast_insert(
            improved,
            nodes,
            nodes_by_id,
            conflict_adj,
            graph_features,
            params,
        )
    if params.get("enable_replacement", True):
        improved = conflict_aware_replacement(
            improved,
            nodes,
            nodes_by_id,
            conflict_adj,
            tasks,
            satellite_ids,
            graph_features,
            params,
        )
    return improved
