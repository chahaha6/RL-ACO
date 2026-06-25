"""Feature and reward helpers for RL-CG-MOACO."""

from __future__ import annotations

from bisect import bisect_left
import math
from typing import Sequence

from .domain import CandidateNode
from .problem_model import estimate_transition_time


GLOBAL_FEATURE_DIM = 8
NODE_FEATURE_DIM = 8


def load_std_from_loads(loads: dict[int, float], satellite_ids: Sequence[int]) -> float:
    values = [float(loads.get(sat_id, 0.0)) for sat_id in satellite_ids]
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    return math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def max_load_std(total_tasks: int, satellite_count: int) -> float:
    if total_tasks <= 0 or satellite_count <= 1:
        return 1.0
    mean_value = total_tasks / satellite_count
    values = [float(total_tasks), *([0.0] * (satellite_count - 1))]
    return max(1.0, math.sqrt(sum((value - mean_value) ** 2 for value in values) / satellite_count))


def transition_reference(nodes: Sequence[CandidateNode], params: dict) -> float:
    if not nodes:
        return 1.0
    coord_1_values = [node.coord_1 for node in nodes]
    coord_2_values = [node.coord_2 for node in nodes]
    diagonal = math.hypot(
        max(coord_1_values) - min(coord_1_values),
        max(coord_2_values) - min(coord_2_values),
    )
    return max(
        1.0,
        float(params.get("min_transition_time", 5.0))
        + float(params.get("maneuver_time_per_degree", 0.2)) * diagonal,
    )


def incremental_maneuver_cost(
    node: CandidateNode,
    sat_sequence: Sequence[CandidateNode],
    params: dict,
) -> float:
    """Cost change when inserting node into one satellite's time-ordered sequence."""

    if not sat_sequence:
        return 0.0
    starts = [item.start for item in sat_sequence]
    pos = bisect_left(starts, node.start)
    prev_node = sat_sequence[pos - 1] if pos > 0 else None
    next_node = sat_sequence[pos] if pos < len(sat_sequence) else None

    delta = 0.0
    if prev_node is not None:
        delta += estimate_transition_time(prev_node, node, params)
    if next_node is not None:
        delta += estimate_transition_time(node, next_node, params)
    if prev_node is not None and next_node is not None:
        delta -= estimate_transition_time(prev_node, next_node, params)
    return max(0.0, delta)


def insert_sorted_by_start(sequence: list[CandidateNode], node: CandidateNode) -> None:
    starts = [item.start for item in sequence]
    sequence.insert(bisect_left(starts, node.start), node)


def block_penalty(
    node_id: int,
    available: set[int],
    profit_by_id: dict[int, float],
    conflict_adj: dict[int, set[int]],
    available_profit_sum: float,
    *,
    enabled: bool = True,
) -> float:
    if not enabled or available_profit_sum <= 0:
        return 0.0
    blocked_profit = sum(profit_by_id[nid] for nid in conflict_adj.get(node_id, set()) & available)
    return min(1.0, max(0.0, blocked_profit / available_profit_sum))


def build_global_features(
    *,
    solution_size: int,
    task_count: int,
    available_size: int,
    node_count: int,
    selected_profit: float,
    total_profit: float,
    current_maneuver: float,
    transition_ref: float,
    sat_loads: dict[int, float],
    satellite_ids: Sequence[int],
    load_std_ref: float,
    graph_features: dict[str, dict[int, float]],
    available: set[int],
    archive_size: int,
    archive_limit: int,
) -> list[float]:
    avg_conflict = 0.0
    if available:
        norm_conflict = graph_features.get("norm_conflict", {})
        avg_conflict = sum(norm_conflict.get(nid, 0.0) for nid in available) / len(available)

    max_load = max((sat_loads.get(sat_id, 0.0) for sat_id in satellite_ids), default=0.0)
    return [
        solution_size / max(1, task_count),
        selected_profit / max(1.0, total_profit),
        available_size / max(1, node_count),
        current_maneuver / max(1.0, transition_ref),
        load_std_from_loads(sat_loads, satellite_ids) / max(1.0, load_std_ref),
        avg_conflict,
        max_load / max(1, solution_size),
        archive_size / max(1, archive_limit),
    ]


def build_node_features(
    *,
    node: CandidateNode,
    graph_features: dict[str, dict[int, float]],
    sat_loads: dict[int, float],
    solution_size: int,
    tau: float,
    tau_min: float,
    tau_max: float,
    eta: float,
    maneuver_delta: float,
    transition_ref: float,
    block_value: float,
) -> list[float]:
    nid = node.node_id
    tau_range = max(1e-12, tau_max - tau_min)
    tau_norm = min(1.0, max(0.0, (tau - tau_min) / tau_range))
    return [
        graph_features.get("norm_profit", {}).get(nid, 0.0),
        graph_features.get("norm_scarcity", {}).get(nid, 0.0),
        graph_features.get("norm_conflict", {}).get(nid, 0.0),
        maneuver_delta / max(1.0, transition_ref),
        sat_loads.get(node.sat_id, 0.0) / max(1, solution_size),
        tau_norm,
        min(1.0, max(0.0, eta)),
        min(1.0, max(0.0, block_value)),
    ]


def construction_score(
    *,
    profit: float,
    max_profit: float,
    maneuver_delta: float,
    transition_ref: float,
    load_before: float,
    load_after: float,
    load_std_ref: float,
    block_value: float,
    block_coef: float,
) -> float:
    profit_gain = profit / max(1.0, max_profit)
    maneuver_gain = maneuver_delta / max(1.0, transition_ref)
    load_delta = (load_after - load_before) / max(1.0, load_std_ref)
    score = (profit_gain - maneuver_gain - load_delta) / 3.0
    score -= float(block_coef) * block_value
    return max(-1.0, min(1.0, score))

