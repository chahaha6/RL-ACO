"""Utility functions for CG-MOACO.

This module intentionally contains only general-purpose helpers:
time parsing, interval checks, normalization, random seed control,
and Pareto archive utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import random
from typing import Iterable, List, Sequence, Tuple


# Scheduling timestamps are stored as floating-point seconds. This tolerance
# only absorbs arithmetic noise; a genuine millisecond-level overrun remains
# infeasible.
TIME_TOLERANCE_SECONDS = 1e-6


def time_not_after(
    value: float,
    limit: float,
    tolerance: float = TIME_TOLERANCE_SECONDS,
) -> bool:
    """Return whether ``value`` is no later than ``limit`` within tolerance."""

    return float(value) <= float(limit) + float(tolerance)


def observation_fits_window(
    start: float,
    end: float,
    task_duration: float,
    tolerance: float = TIME_TOLERANCE_SECONDS,
) -> bool:
    """Return whether a fixed-start observation finishes inside its window."""

    return time_not_after(
        float(start) + float(task_duration),
        float(end),
        tolerance,
    )


@dataclass(frozen=True)
class Solution:
    """A scheduled solution and its objective vector.

    All objective values are minimized.
    """

    node_ids: frozenset[int]
    objectives: Tuple[float, float, float]

    @property
    def size(self) -> int:
        return len(self.node_ids)


def set_random_seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)


def parse_time_to_seconds(tokens: Sequence[str]) -> float:
    """Parse [yyyy, mm, dd, HH, MM, SS.sss] into seconds from midnight.

    The input data uses values such as '.000' for seconds. Python float can
    handle that format, so we parse seconds separately and compute seconds of
    day. The date part is preserved only for validation.
    """

    if len(tokens) != 6:
        raise ValueError(f"Expected 6 time tokens, got {len(tokens)}: {tokens}")
    year, month, day, hour, minute = map(int, tokens[:5])
    second = float(tokens[5])
    # Validate date fields, then return seconds from midnight.
    datetime(year, month, day, hour, minute, int(second))
    return hour * 3600.0 + minute * 60.0 + second


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def intervals_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    """Return True if [start_a, end_a] and [start_b, end_b] overlap."""

    return max(start_a, start_b) < min(end_a, end_b)


def normalize_dict(values: dict[int, float], higher_is_better: bool = True) -> dict[int, float]:
    """Normalize a dictionary of numeric values into [0, 1].

    If all values are the same, every normalized value is set to 1.0 because
    the feature does not distinguish nodes.
    """

    if not values:
        return {}
    vals = list(values.values())
    min_v, max_v = min(vals), max(vals)
    if math.isclose(max_v, min_v):
        return {k: 1.0 for k in values}
    if higher_is_better:
        return {k: (v - min_v) / (max_v - min_v) for k, v in values.items()}
    return {k: (max_v - v) / (max_v - min_v) for k, v in values.items()}


def normalize_list(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    min_v, max_v = min(values), max(values)
    if math.isclose(max_v, min_v):
        return [1.0 for _ in values]
    return [(v - min_v) / (max_v - min_v) for v in values]


def dominates(obj_a: Sequence[float], obj_b: Sequence[float], eps: float = 1e-12) -> bool:
    """Return True if obj_a Pareto-dominates obj_b.

    All objectives are assumed to be minimized.
    """

    not_worse = all(a <= b + eps for a, b in zip(obj_a, obj_b))
    strictly_better = any(a < b - eps for a, b in zip(obj_a, obj_b))
    return not_worse and strictly_better


def get_non_dominated(solutions: Sequence[Solution]) -> List[Solution]:
    """Return non-dominated solutions, removing exact duplicate schedules."""

    unique: dict[frozenset[int], Solution] = {}
    for sol in solutions:
        old = unique.get(sol.node_ids)
        if old is None or sum(sol.objectives) < sum(old.objectives):
            unique[sol.node_ids] = sol
    sols = list(unique.values())

    nd: List[Solution] = []
    for i, sol in enumerate(sols):
        dominated = False
        for j, other in enumerate(sols):
            if i != j and dominates(other.objectives, sol.objectives):
                dominated = True
                break
        if not dominated:
            nd.append(sol)
    return nd


def crowding_distance(solutions: Sequence[Solution]) -> dict[int, float]:
    """Compute crowding distance indexed by list position."""

    n = len(solutions)
    if n == 0:
        return {}
    if n <= 2:
        return {i: float("inf") for i in range(n)}

    distances = [0.0] * n
    m = len(solutions[0].objectives)
    for obj_idx in range(m):
        order = sorted(range(n), key=lambda idx: solutions[idx].objectives[obj_idx])
        min_v = solutions[order[0]].objectives[obj_idx]
        max_v = solutions[order[-1]].objectives[obj_idx]
        denom = max_v - min_v
        if math.isclose(denom, 0.0, abs_tol=1e-12):
            continue
        distances[order[0]] = float("inf")
        distances[order[-1]] = float("inf")
        for pos in range(1, n - 1):
            prev_v = solutions[order[pos - 1]].objectives[obj_idx]
            next_v = solutions[order[pos + 1]].objectives[obj_idx]
            distances[order[pos]] += (next_v - prev_v) / denom
    return {i: distances[i] for i in range(n)}


def update_archive(
    archive: Sequence[Solution],
    population: Sequence[Solution],
    archive_size: int,
) -> List[Solution]:
    """Merge archive and population, keep non-dominated diverse solutions."""

    nd = get_non_dominated([*archive, *population])
    if len(nd) <= archive_size:
        return nd

    kept = list(nd)
    while len(kept) > archive_size:
        cd = crowding_distance(kept)
        finite_items = [(idx, dist) for idx, dist in cd.items() if math.isfinite(dist)]
        if not finite_items:
            kept.pop(-1)
        else:
            remove_idx = min(finite_items, key=lambda item: item[1])[0]
            kept.pop(remove_idx)
    return kept


def update_archive_with_acceptance(
    archive: Sequence[Solution],
    candidate: Solution,
    archive_size: int,
) -> tuple[List[Solution], bool]:
    """Incrementally update a non-dominated archive with one candidate.

    A duplicate schedule is not considered a new archive entry. A
    non-dominated candidate also counts as rejected when archive truncation
    removes it. Existing archive members are already mutually non-dominated,
    so only candidate-to-archive comparisons are required.
    """

    existing_node_sets = {solution.node_ids for solution in archive}
    if candidate.node_ids in existing_node_sets:
        return list(archive), False

    dominated_indices: set[int] = set()
    for idx, solution in enumerate(archive):
        if dominates(solution.objectives, candidate.objectives):
            return list(archive), False
        if dominates(candidate.objectives, solution.objectives):
            dominated_indices.add(idx)

    updated = [
        solution
        for idx, solution in enumerate(archive)
        if idx not in dominated_indices
    ]
    updated.append(candidate)

    while len(updated) > archive_size:
        distances = crowding_distance(updated)
        finite_items = [
            (idx, distance)
            for idx, distance in distances.items()
            if math.isfinite(distance)
        ]
        if not finite_items:
            updated.pop(-1)
        else:
            remove_idx = min(finite_items, key=lambda item: item[1])[0]
            updated.pop(remove_idx)

    accepted = any(
        solution.node_ids == candidate.node_ids
        for solution in updated
    )
    return updated, accepted


def roulette_select(items: Sequence[int], weights: Sequence[float]) -> int:
    """Roulette-wheel selection with robust fallback."""

    if not items:
        raise ValueError("Cannot select from an empty item list.")
    total = sum(w for w in weights if w > 0)
    if total <= 0:
        return random.choice(list(items))
    threshold = random.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        if weight <= 0:
            continue
        cumulative += weight
        if cumulative >= threshold:
            return item
    return items[-1]
