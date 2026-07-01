"""Lightweight Pareto local search for CG-MOACO variants."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

from .domain import CandidateNode, Task
from .problem_model import evaluate_solution, is_feasible
from .utils import Solution, dominates, update_archive_with_acceptance


def _sample_candidates(
    candidates: list[CandidateNode],
    limit: int,
) -> list[CandidateNode]:
    """Shuffle candidates and return at most ``limit`` of them."""

    random.shuffle(candidates)
    if limit > 0:
        return candidates[:limit]
    return candidates


def _released_refill_candidates(
    conflict_set: set[int],
    replacement_base: set[int],
    nodes_by_id: Dict[int, CandidateNode],
    conflict_adj: dict[int, set[int]],
) -> list[CandidateNode]:
    """Return nodes made feasible by removing ``conflict_set``.

    A newly feasible node must neighbor at least one removed node. Building
    this union avoids scanning every candidate node for each replacement.
    """

    released_candidate_ids: set[int] = set()
    for removed_id in conflict_set:
        released_candidate_ids.update(conflict_adj.get(removed_id, set()))

    return [
        nodes_by_id[node_id]
        for node_id in sorted(released_candidate_ids)
        if node_id in nodes_by_id
        and node_id not in replacement_base
        and not (conflict_adj.get(node_id, set()) & replacement_base)
    ]


def _make_solution(
    node_ids: set[int],
    nodes_by_id: Dict[int, CandidateNode],
    tasks: Dict[int, Task],
    satellite_ids: Sequence[int],
    params: dict,
) -> Solution:
    frozen = frozenset(node_ids)
    objectives = evaluate_solution(
        frozen,
        nodes_by_id,
        tasks,
        satellite_ids,
        params,
    )
    return Solution(node_ids=frozen, objectives=objectives)


def _try_pareto_candidate(
    current: Solution,
    trial_node_ids: set[int],
    archive: List[Solution],
    archive_size: int,
    nodes_by_id: Dict[int, CandidateNode],
    conflict_adj: dict[int, set[int]],
    tasks: Dict[int, Task],
    satellite_ids: Sequence[int],
    params: dict,
) -> tuple[Solution, List[Solution], bool]:
    """Evaluate one feasible neighbor and apply the Pareto acceptance rule."""

    if not is_feasible(trial_node_ids, conflict_adj):
        return current, archive, False

    candidate = _make_solution(
        trial_node_ids,
        nodes_by_id,
        tasks,
        satellite_ids,
        params,
    )

    if dominates(current.objectives, candidate.objectives):
        return current, archive, False

    updated_archive, accepted = update_archive_with_acceptance(
        archive,
        candidate,
        archive_size,
    )
    if not accepted:
        return current, archive, False

    if dominates(candidate.objectives, current.objectives):
        return candidate, updated_archive, True

    # The candidate is non-dominated with the current solution. Keep the
    # current ant solution unchanged, but retain the new archive member.
    return current, updated_archive, True


def pareto_local_search(
    current: Solution,
    archive: List[Solution],
    archive_size: int,
    nodes: List[CandidateNode],
    nodes_by_id: Dict[int, CandidateNode],
    conflict_adj: dict[int, set[int]],
    tasks: Dict[int, Task],
    satellite_ids: Sequence[int],
    params: dict,
) -> tuple[Solution, List[Solution], bool]:
    """Run replacement-refill composite moves with first improvement.

    Each sampled replacement first creates a feasible base solution. Random
    insertions into the space released by that replacement are evaluated
    before the bare replacement. A Pareto-dominating neighbor replaces
    ``current``; a mutually non-dominated neighbor is added to the archive
    while ``current`` remains unchanged.
    """

    if not params.get("enable_local_search", True):
        return current, archive, False

    selected = set(current.node_ids)

    if not params.get("enable_replacement", True):
        return current, archive, False

    max_conflicts = int(params.get("max_replace_conflicts", 3))
    replacement_candidates = []
    if max_conflicts >= 1:
        replacement_candidates = [
            node
            for node in nodes
            if node.node_id not in selected
            and 1
            <= len(conflict_adj.get(node.node_id, set()) & selected)
            <= max_conflicts
        ]
    replacement_candidates = _sample_candidates(
        replacement_candidates,
        int(params.get("replacement_attempts", 100)),
    )

    for replacement in replacement_candidates:
        conflict_set = conflict_adj.get(replacement.node_id, set()) & selected
        replacement_base = selected - conflict_set
        replacement_base.add(replacement.node_id)

        if not is_feasible(replacement_base, conflict_adj):
            continue

        if params.get("enable_fast_insert", True):
            refill_candidates = _released_refill_candidates(
                conflict_set,
                replacement_base,
                nodes_by_id,
                conflict_adj,
            )
            refill_candidates = _sample_candidates(
                refill_candidates,
                int(params.get("local_search_candidate_limit", 0)),
            )

            for insertion in refill_candidates:
                composite_trial = set(replacement_base)
                composite_trial.add(insertion.node_id)
                accepted_solution, updated_archive, accepted = _try_pareto_candidate(
                    current,
                    composite_trial,
                    archive,
                    archive_size,
                    nodes_by_id,
                    conflict_adj,
                    tasks,
                    satellite_ids,
                    params,
                )
                if accepted:
                    return accepted_solution, updated_archive, True

        # If no replacement-refill neighbor is accepted, retain the original
        # replacement operator as a fallback neighborhood.
        accepted_solution, updated_archive, accepted = _try_pareto_candidate(
            current,
            replacement_base,
            archive,
            archive_size,
            nodes_by_id,
            conflict_adj,
            tasks,
            satellite_ids,
            params,
        )
        if accepted:
            return accepted_solution, updated_archive, True

    return current, archive, False
