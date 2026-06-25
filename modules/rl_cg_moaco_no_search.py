"""RL-CG-MOACO ablation without local search."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .domain import CandidateNode, Task
from .rl_cg_moaco import RLCGMOACO, DEFAULT_PARAMS as RL_CG_DEFAULT_PARAMS


ABLATION_PARAMS = {
    "enable_local_search": False,
}

DEFAULT_PARAMS = {
    **RL_CG_DEFAULT_PARAMS,
    **ABLATION_PARAMS,
}


class RLCGMOACO_NO_SEARCH(RLCGMOACO):
    """RL-CG-MOACO without fast insertion and replacement local search."""

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

