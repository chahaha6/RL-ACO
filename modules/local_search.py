"""Conflict-graph-aware fast insertion and replacement local search."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Set

from .domain import CandidateNode, Task
from .problem_model import calculate_maneuver_cost, evaluate_solution
from .utils import dominates


def _selected_tasks(solution: Iterable[int], nodes_by_id: Dict[int, CandidateNode]) -> set[int]:
    return {nodes_by_id[node_id].task_id for node_id in solution}


def _is_node_feasible_for_solution(node_id: int, solution: set[int], conflict_adj: dict[int, set[int]]) -> bool:
    return not (conflict_adj.get(node_id, set()) & solution)


def _dynamic_insert_score(
    node: CandidateNode,
    solution: set[int],
    nodes_by_id: Dict[int, CandidateNode],
    graph_features: dict[str, dict[int, float]],
    params: dict,
) -> float:
    sat_load = sum(1 for nid in solution if nodes_by_id[nid].sat_id == node.sat_id)
    max_load = max(1, len(solution))
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
    candidate_nodes = [node for node in nodes if node.task_id not in scheduled_tasks]
    candidate_nodes.sort(
        key=lambda n: _dynamic_insert_score(n, improved, nodes_by_id, graph_features, params),
        reverse=True,
    )

    for node in candidate_nodes:
        if node.task_id in scheduled_tasks:
            continue
        if _is_node_feasible_for_solution(node.node_id, improved, conflict_adj):
            improved.add(node.node_id)
            scheduled_tasks.add(node.task_id)
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

    unscheduled_nodes = [node for node in nodes if node.task_id not in scheduled_tasks]
    unscheduled_nodes.sort(
        key=lambda n: _dynamic_insert_score(n, improved, nodes_by_id, graph_features, params),
        reverse=True,
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
