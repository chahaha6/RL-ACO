"""Task-satellite-window conflict graph construction.

Each candidate node represents v=(task, satellite, window). An undirected
conflict edge means two candidate nodes cannot appear in the same schedule.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, List

from .domain import CandidateNode
from .problem_model import estimate_transition_time
from .utils import normalize_dict, observation_fits_window, time_not_after


def add_undirected_edge(adj: dict[int, set[int]], a: int, b: int) -> None:
    if a == b:
        return
    adj.setdefault(a, set()).add(b)
    adj.setdefault(b, set()).add(a)


def has_maneuver_conflict(
    node_a: CandidateNode,
    node_b: CandidateNode,
    params: dict | None = None,
) -> bool:
    """Return True when two fixed observation executions cannot coexist.

    A candidate observation starts at its visibility-window start and occupies
    ``task_duration`` seconds. Using fixed execution intervals preserves the
    conflict-graph property that every independent set is a feasible schedule.
    """

    if node_a.sat_id != node_b.sat_id:
        return False

    execution_end_a = node_a.start + node_a.task_duration
    execution_end_b = node_b.start + node_b.task_duration
    if not observation_fits_window(node_a.start, node_a.end, node_a.task_duration):
        return True
    if not observation_fits_window(node_b.start, node_b.end, node_b.task_duration):
        return True

    if time_not_after(execution_end_a, node_b.start):
        return not time_not_after(
            execution_end_a + estimate_transition_time(node_a, node_b, params),
            node_b.start,
        )
    if time_not_after(execution_end_b, node_a.start):
        return not time_not_after(
            execution_end_b + estimate_transition_time(node_b, node_a, params),
            node_a.start,
        )
    return True


def build_conflict_graph(nodes: List[CandidateNode], params: dict | None = None) -> dict[int, set[int]]:
    """Build conflict graph adjacency list.

    Conflict types:
    1. Same task mutual exclusion.
    2. Same satellite cannot execute overlapping fixed observation intervals.
    3. Each actual observation interval must fit inside its visibility window.
    4. Adjacent same-satellite observations must leave enough attitude
       transition time.
    """

    for node in nodes:
        if not observation_fits_window(node.start, node.end, node.task_duration):
            actual_duration = node.end - node.start
            raise ValueError(
                f"Candidate node {node.node_id} cannot fit inside its visibility "
                f"window: end-start={actual_duration:.9f}, "
                f"task_duration={node.task_duration:.9f}"
            )

    conflict_adj: dict[int, set[int]] = {node.node_id: set() for node in nodes}

    # Type 1: same task mutual exclusion.
    by_task: dict[int, List[CandidateNode]] = defaultdict(list)
    for node in nodes:
        by_task[node.task_id].append(node)
    for task_nodes in by_task.values():
        for i in range(len(task_nodes)):
            for j in range(i + 1, len(task_nodes)):
                add_undirected_edge(conflict_adj, task_nodes[i].node_id, task_nodes[j].node_id)

    # Type 2: no feasible same-satellite execution order.
    by_sat: dict[int, List[CandidateNode]] = defaultdict(list)
    for node in nodes:
        by_sat[node.sat_id].append(node)

    for sat_nodes in by_sat.values():
        sorted_nodes = sorted(sat_nodes, key=lambda n: (n.start, n.end))
        for i in range(len(sorted_nodes)):
            a = sorted_nodes[i]
            for j in range(i + 1, len(sorted_nodes)):
                b = sorted_nodes[j]
                if has_maneuver_conflict(a, b, params):
                    add_undirected_edge(conflict_adj, a.node_id, b.node_id)
    return conflict_adj


def compute_window_scarcity(nodes: List[CandidateNode]) -> dict[int, float]:
    counts = Counter(node.task_id for node in nodes)
    return {node.node_id: 1.0 / counts[node.task_id] for node in nodes}


def compute_conflict_degree(conflict_adj: dict[int, set[int]]) -> dict[int, float]:
    return {node_id: float(len(neighbors)) for node_id, neighbors in conflict_adj.items()}


def compute_graph_features(nodes: List[CandidateNode], conflict_adj: dict[int, set[int]], params: dict | None = None) -> dict[str, dict[int, float]]:
    """Compute graph features and static heuristic components."""

    profit = {node.node_id: node.profit for node in nodes}
    scarcity = compute_window_scarcity(nodes)
    conflict_degree = compute_conflict_degree(conflict_adj)

    # Static maneuver pressure: average estimated transition time from this node
    # to a small neighborhood of same-satellite nodes. This is only a heuristic.
    same_sat = defaultdict(list)
    for node in nodes:
        same_sat[node.sat_id].append(node)
    maneuver_pressure: dict[int, float] = {}
    for node in nodes:
        related = same_sat[node.sat_id]
        if len(related) <= 1:
            maneuver_pressure[node.node_id] = 0.0
            continue
        # Use up to 30 nearest-in-time nodes to keep this cheap.
        candidates = sorted(related, key=lambda n: abs(n.start - node.start))[:30]
        vals = [estimate_transition_time(node, other, params) for other in candidates if other.node_id != node.node_id]
        maneuver_pressure[node.node_id] = sum(vals) / len(vals) if vals else 0.0

    norm_profit = normalize_dict(profit, higher_is_better=True)
    norm_scarcity = normalize_dict(scarcity, higher_is_better=True)
    norm_conflict = normalize_dict(conflict_degree, higher_is_better=True)
    norm_maneuver = normalize_dict(maneuver_pressure, higher_is_better=True)

    # Graph contribution for pheromone updates and static heuristic baseline.
    contribution: dict[int, float] = {}
    for node in nodes:
        nid = node.node_id
        numerator = norm_profit[nid] + (params or {}).get("lambda_scarcity", 1.0) * norm_scarcity[nid]
        denominator = 1.0 + (params or {}).get("lambda_conflict", 1.0) * norm_conflict[nid] + (params or {}).get("lambda_maneuver", 1.0) * norm_maneuver[nid]
        contribution[nid] = max(1e-9, numerator / denominator)

    return {
        "profit": profit,
        "scarcity": scarcity,
        "conflict_degree": conflict_degree,
        "maneuver_pressure": maneuver_pressure,
        "norm_profit": norm_profit,
        "norm_scarcity": norm_scarcity,
        "norm_conflict": norm_conflict,
        "norm_maneuver": norm_maneuver,
        "contribution": contribution,
    }


def solution_conflict_edges(solution: Iterable[int], conflict_adj: dict[int, set[int]]) -> int:
    selected = set(solution)
    count = 0
    for node_id in selected:
        count += len(conflict_adj.get(node_id, set()) & selected)
    return count // 2
