"""CG-MOACO ablation without conflict-graph-aware pheromone update."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .cg_moaco import CGMOACO, DEFAULT_PARAMS as CG_DEFAULT_PARAMS
from .domain import CandidateNode, Task


ABLATION_PARAMS = {
    "use_graph_pheromone": False,
}

DEFAULT_PARAMS = {
    **CG_DEFAULT_PARAMS,
    **ABLATION_PARAMS,
}


class CGMOACO_NO_MESSAGE(CGMOACO):
    """CG-MOACO-NO-MESSAGE."""

    def __init__(
        self,
        tasks: Dict[int, Task],
        nodes: List[CandidateNode],
        conflict_adj: dict[int, set[int]],
        graph_features: dict[str, dict[int, float]],
        params: dict | None = None,
        satellite_ids: Sequence[int] | None = None,
    ) -> None:
        merged_params = {
            **(params or {}),
            **ABLATION_PARAMS,
        }
        super().__init__(
            tasks=tasks,
            nodes=nodes,
            conflict_adj=conflict_adj,
            graph_features=graph_features,
            params=merged_params,
            satellite_ids=satellite_ids,
        )

