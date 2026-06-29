"""RL-CG-MOACO with advantage-corrected MLP-DDQN node selection."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import math
import random
from pathlib import Path
import time
from typing import Dict, Iterable, List, Sequence

from .cg_moaco import CGMOACO, DEFAULT_PARAMS as CG_DEFAULT_PARAMS
from .ddqn_agent import DDQNAgent, ReplayTransition
from .domain import CandidateNode, Task
from .local_search import local_search
from .problem_model import evaluate_solution, is_feasible, task_completion_rate
from .rl_features import (
    GLOBAL_FEATURE_DIM,
    NODE_FEATURE_DIM,
    block_penalty,
    build_global_features,
    build_node_features,
    construction_score,
    incremental_maneuver_cost,
    insert_sorted_by_start,
    load_std_from_loads,
    max_load_std,
    transition_reference,
)
from .utils import (
    Solution,
    crowding_distance,
    roulette_select,
    set_random_seed,
    update_archive,
)


DEFAULT_PARAMS = {
    **CG_DEFAULT_PARAMS,
    "max_iter": 300,
    "warmup_iter": 30,
    "use_ddqn": True,
    "use_block_penalty": True,
    "use_final_reward": True,
    "gamma": 0.9,
    "rl_lr": 1e-3,
    "rl_lr_late": 5e-4,
    "rl_batch_size": 64,
    "rl_replay_capacity": 10000,
    "rl_target_sync_iter": 10,
    "rl_train_steps_per_iter": 1,
    "rl_hidden_dims": (128, 128, 64),
    "rl_epsilon_start": 0.30,
    "rl_epsilon_end": 0.05,
    "rl_kappa_max": 1.2,
    "rl_q_clip": 5.0,
    "use_advantage_correction": True,
    "block_penalty_coef": 0.2,
    "rl_freeze_start_iter": None,
}


@dataclass
class _RawTransition:
    state_action: list[float]
    raw_score: float
    next_actions: list[list[float]]
    done: bool


class RLCGMOACO(CGMOACO):
    """Conflict-graph-guided MOACO with online advantage-corrected DDQN assistance."""

    def __init__(
        self,
        tasks: Dict[int, Task],
        nodes: List[CandidateNode],
        conflict_adj: dict[int, set[int]],
        graph_features: dict[str, dict[int, float]],
        params: dict | None = None,
        satellite_ids: Sequence[int] | None = None,
    ) -> None:
        merged_params = {**DEFAULT_PARAMS, **(params or {})}
        super().__init__(
            tasks=tasks,
            nodes=nodes,
            conflict_adj=conflict_adj,
            graph_features=graph_features,
            params=merged_params,
            satellite_ids=satellite_ids,
        )
        set_random_seed(self.params.get("seed"))
        self.total_profit = sum(task.profit for task in self.tasks.values())
        self.max_node_profit = max((node.profit for node in self.nodes), default=1.0)
        self.profit_by_id = {node.node_id: node.profit for node in self.nodes}
        self.transition_ref = transition_reference(self.nodes, self.params)
        if self.params.get("load_mode", "task_count") == "duration":
            total_load = sum(task.duration for task in self.tasks.values())
        else:
            total_load = len(self.tasks)
        self.load_std_ref = max_load_std(total_load, len(self.satellite_ids))
        self.input_dim = GLOBAL_FEATURE_DIM + NODE_FEATURE_DIM
        self.t_ref: int | None = None
        self.warmup_lengths: list[int] = []
        self._pending_episodes: list[tuple[list[_RawTransition], bool]] = []
        self.last_epsilon = 0.0
        self.last_kappa = 0.0
        self.last_q_baseline = 0.0
        self.last_advantage_span = 0.0
        self.avg_episode_length = 0.0

        self.agent: DDQNAgent | None = None
        if self.params.get("use_ddqn", True):
            self.agent = DDQNAgent(
                self.input_dim,
                hidden_dims=tuple(self.params.get("rl_hidden_dims", (128, 128, 64))),
                gamma=float(self.params.get("gamma", 0.9)),
                lr=float(self.params.get("rl_lr", 1e-3)),
                replay_capacity=int(self.params.get("rl_replay_capacity", 10000)),
                batch_size=int(self.params.get("rl_batch_size", 64)),
                seed=self.params.get("seed"),
            )

        if int(self.params.get("warmup_iter", 30)) <= 0:
            self.t_ref = 1

    @property
    def rl_stats(self) -> dict[str, float | int | str]:
        replay_size = len(self.agent.replay) if self.agent is not None else 0
        last_loss = self.agent.last_loss if self.agent is not None else None
        return {
            "t_ref": self.t_ref if self.t_ref is not None else "",
            "avg_episode_length": self.avg_episode_length,
            "last_train_loss": last_loss if last_loss is not None else "",
            "last_epsilon": self.last_epsilon,
            "last_kappa": self.last_kappa,
            "last_q_baseline": self.last_q_baseline,
            "last_advantage_span": self.last_advantage_span,
            "replay_size": replay_size,
        }

    def _node_load_amount(self, node: CandidateNode) -> float:
        load_mode = self.params.get("load_mode", "task_count")

        if load_mode == "task_count":
            return 1.0
        if load_mode == "duration":
            return float(node.task_duration)

        raise ValueError(f"Unknown load_mode: {load_mode}")

    def _current_epsilon(self, iteration: int, max_iter: int) -> float:
        if not self.params.get("use_ddqn", True):
            return 0.0
        warmup_iter = int(self.params.get("warmup_iter", 30))
        if iteration <= warmup_iter:
            return 0.0
        start = float(self.params.get("rl_epsilon_start", 0.30))
        end = float(self.params.get("rl_epsilon_end", 0.05))
        denom = max(1, max_iter - warmup_iter)
        progress = min(1.0, max(0.0, (iteration - warmup_iter) / denom))
        return start + (end - start) * progress

    def _current_kappa(self, iteration: int, max_iter: int) -> float:
        if not self.params.get("use_ddqn", True):
            return 0.0
        warmup_iter = int(self.params.get("warmup_iter", 30))
        if iteration <= warmup_iter:
            return 0.0
        progress = min(1.0, max(0.0, (iteration - warmup_iter) / max(1, max_iter - warmup_iter)))
        kappa_max = float(self.params.get("rl_kappa_max", 1.2))
        if progress < 0.25:
            local = progress / 0.25
            return 0.1 + (0.4 - 0.1) * local
        if progress < 0.85:
            local = (progress - 0.25) / 0.60
            return 0.4 + (1.0 - 0.4) * local
        local = (progress - 0.85) / 0.15
        return 1.0 + (kappa_max - 1.0) * min(1.0, local)

    def _freeze_start_iter(self, max_iter: int) -> int:
        explicit = self.params.get("rl_freeze_start_iter")
        if explicit is not None:
            return int(explicit)
        return max(1, int(0.93 * max_iter) + 1)

    def _set_learning_rate_for_iter(self, iteration: int, max_iter: int) -> None:
        if self.agent is None:
            return
        late_start = max(1, int(0.80 * max_iter))
        lr = self.params.get("rl_lr_late", 5e-4) if iteration >= late_start else self.params.get("rl_lr", 1e-3)
        self.agent.set_lr(float(lr))

    def _action_space_features(
        self,
        *,
        solution: set[int],
        available: set[int],
        sat_loads: dict[int, float],
        sat_sequences: dict[int, list[CandidateNode]],
        selected_profit: float,
        current_maneuver: float,
        available_profit_sum: float,
    ) -> tuple[list[int], list[list[float]], dict[int, dict[str, float]]]:
        global_features = build_global_features(
            solution_size=len(solution),
            task_count=len(self.tasks),
            available_size=len(available),
            node_count=len(self.nodes),
            selected_profit=selected_profit,
            total_profit=self.total_profit,
            current_maneuver=current_maneuver,
            transition_ref=self.transition_ref,
            sat_loads=sat_loads,
            satellite_ids=self.satellite_ids,
            load_std_ref=self.load_std_ref,
            graph_features=self.graph_features,
            available=available,
            archive_size=len(self.archive),
            archive_limit=int(self.params.get("archive_size", 100)),
        )

        load_before = load_std_from_loads(sat_loads, self.satellite_ids)
        items = self._candidate_pool(available)
        max_load = max(1.0, max(sat_loads.values()) if sat_loads else 1.0)
        feature_rows: list[list[float]] = []
        infos: dict[int, dict[str, float]] = {}
        block_enabled = bool(self.params.get("use_block_penalty", True))
        for node_id in items:
            node = self.nodes_by_id[node_id]
            eta = self._dynamic_heuristic(node_id, solution, sat_loads, max_load)
            maneuver_delta = incremental_maneuver_cost(
                node,
                sat_sequences.get(node.sat_id, []),
                self.params,
            )
            block_value = block_penalty(
                node_id,
                available,
                self.profit_by_id,
                self.conflict_adj,
                available_profit_sum,
                enabled=block_enabled,
            )
            load_amount = self._node_load_amount(node)
            next_loads = dict(sat_loads)
            next_loads[node.sat_id] = next_loads.get(node.sat_id, 0.0) + load_amount
            load_after = load_std_from_loads(next_loads, self.satellite_ids)
            raw_score = construction_score(
                profit=node.profit,
                max_profit=self.max_node_profit,
                maneuver_delta=maneuver_delta,
                transition_ref=self.transition_ref,
                load_before=load_before,
                load_after=load_after,
                load_std_ref=self.load_std_ref,
                block_value=block_value,
                block_coef=float(self.params.get("block_penalty_coef", 0.2)),
            )
            node_features = build_node_features(
                node=node,
                graph_features=self.graph_features,
                sat_loads=sat_loads,
                solution_size=len(solution),
                tau=max(float(self.params["tau_min"]), self.pheromone[node_id]),
                tau_min=float(self.params["tau_min"]),
                tau_max=float(self.params["tau_max"]),
                eta=eta,
                maneuver_delta=maneuver_delta,
                transition_ref=self.transition_ref,
                block_value=block_value,
            )
            feature_rows.append([*global_features, *node_features])
            infos[node_id] = {
                "eta": eta,
                "raw_score": raw_score,
                "maneuver_delta": maneuver_delta,
                "load_amount": load_amount,
                "block": block_value,
            }
        return items, feature_rows, infos

    def _select_node(
        self,
        items: Sequence[int],
        feature_rows: Sequence[Sequence[float]],
        infos: dict[int, dict[str, float]],
        *,
        iteration: int,
        max_iter: int,
    ) -> int:
        epsilon = self._current_epsilon(iteration, max_iter)
        kappa = self._current_kappa(iteration, max_iter)
        self.last_epsilon = epsilon
        self.last_kappa = kappa

        use_ddqn = self.agent is not None and iteration > int(self.params.get("warmup_iter", 30))
        if use_ddqn and random.random() < epsilon:
            return random.choice(list(items))

        q_values = self.agent.predict(feature_rows) if use_ddqn and self.agent is not None else [0.0] * len(items)
        q_clip = float(self.params.get("rl_q_clip", 5.0))
        clipped_q_values = [max(-q_clip, min(q_clip, float(q_value))) for q_value in q_values]

        if (
            use_ddqn
            and clipped_q_values
            and bool(self.params.get("use_advantage_correction", True))
        ):
            q_baseline = sum(clipped_q_values) / len(clipped_q_values)
            correction_values = [q_value - q_baseline for q_value in clipped_q_values]
        else:
            q_baseline = 0.0
            correction_values = clipped_q_values

        self.last_q_baseline = q_baseline
        self.last_advantage_span = (
            max(correction_values) - min(correction_values)
            if correction_values
            else 0.0
        )

        log_weights: list[float] = []
        for node_id, correction_value in zip(items, correction_values):
            tau = max(float(self.params["tau_min"]), self.pheromone[node_id])
            eta = max(1e-9, infos[node_id]["eta"])
            log_weight = (
                float(self.params["alpha"]) * math.log(tau)
                + float(self.params["beta"]) * math.log(eta)
                + kappa * correction_value
            )
            log_weights.append(log_weight)

        max_log = max(log_weights) if log_weights else 0.0
        weights = [math.exp(value - max_log) for value in log_weights]
        return roulette_select(items, weights)

    def _construct_episode(self, iteration: int, max_iter: int) -> tuple[set[int], list[_RawTransition]]:
        solution: set[int] = set()
        available: set[int] = {node.node_id for node in self.nodes}
        sat_loads = {sat_id: 0.0 for sat_id in self.satellite_ids}
        sat_sequences: dict[int, list[CandidateNode]] = {sat_id: [] for sat_id in self.satellite_ids}
        selected_profit = 0.0
        current_maneuver = 0.0
        available_profit_sum = sum(self.profit_by_id.values())
        episode: list[_RawTransition] = []
        pending_state_action: list[float] | None = None
        pending_raw_score = 0.0

        while available:
            items, feature_rows, infos = self._action_space_features(
                solution=solution,
                available=available,
                sat_loads=sat_loads,
                sat_sequences=sat_sequences,
                selected_profit=selected_profit,
                current_maneuver=current_maneuver,
                available_profit_sum=available_profit_sum,
            )
            if pending_state_action is not None:
                episode.append(
                    _RawTransition(
                        state_action=pending_state_action,
                        raw_score=pending_raw_score,
                        next_actions=[list(row) for row in feature_rows],
                        done=False,
                    )
                )

            chosen = self._select_node(
                items,
                feature_rows,
                infos,
                iteration=iteration,
                max_iter=max_iter,
            )
            chosen_idx = items.index(chosen)
            chosen_node = self.nodes_by_id[chosen]
            chosen_info = infos[chosen]

            solution.add(chosen)
            selected_profit += chosen_node.profit
            current_maneuver += chosen_info["maneuver_delta"]
            sat_loads[chosen_node.sat_id] = sat_loads.get(chosen_node.sat_id, 0.0) + chosen_info["load_amount"]
            insert_sorted_by_start(sat_sequences.setdefault(chosen_node.sat_id, []), chosen_node)

            removed = {chosen} | (self.conflict_adj.get(chosen, set()) & available)
            available_profit_sum -= sum(self.profit_by_id[nid] for nid in removed)
            available.difference_update(removed)
            available_profit_sum = max(0.0, available_profit_sum)

            pending_state_action = list(feature_rows[chosen_idx])
            pending_raw_score = float(chosen_info["raw_score"])

        if pending_state_action is not None:
            episode.append(
                _RawTransition(
                    state_action=pending_state_action,
                    raw_score=pending_raw_score,
                    next_actions=[],
                    done=True,
                )
            )
        return solution, episode

    def _candidate_enters_archive(self, candidate: Solution, archive_size: int) -> bool:
        before = {sol.node_ids for sol in self.archive}
        updated = update_archive(self.archive, [candidate], archive_size)
        after = {sol.node_ids for sol in updated}
        self.archive = updated
        return candidate.node_ids in after and candidate.node_ids not in before

    def _store_episode(self, episode: list[_RawTransition], entered_archive: bool) -> None:
        if self.agent is None or not episode:
            return
        t_ref = max(1, int(self.t_ref or len(episode) or 1))
        transitions: list[ReplayTransition] = []
        for idx, raw in enumerate(episode):
            reward = raw.raw_score / t_ref
            if idx == len(episode) - 1 and self.params.get("use_final_reward", True) and entered_archive:
                reward += 1.0
            transitions.append(
                ReplayTransition(
                    state_action=raw.state_action,
                    reward=reward,
                    next_actions=raw.next_actions,
                    done=raw.done,
                )
            )
        self.agent.add_transitions(transitions)

    def _finish_warmup(self) -> None:
        if self.t_ref is None:
            if self.warmup_lengths:
                self.t_ref = max(1, round(sum(self.warmup_lengths) / len(self.warmup_lengths)))
            else:
                self.t_ref = 1
        for episode, entered_archive in self._pending_episodes:
            self._store_episode(episode, entered_archive)
        self._pending_episodes.clear()

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
        finite = [value for value in cd.values() if math.isfinite(value)]
        max_cd = max(finite) if finite else 1.0

        for idx, sol in enumerate(self.archive):
            diversity_weight = 1.0
            if math.isfinite(cd.get(idx, 0.0)) and max_cd > 0:
                diversity_weight += cd[idx] / max_cd
            elif math.isinf(cd.get(idx, 0.0)):
                diversity_weight += 1.0
            for node_id in sol.node_ids:
                graph_contribution = (
                    self.graph_features["contribution"].get(node_id, 1.0)
                    if self.params.get("use_graph_pheromone", True)
                    else 1.0
                )
                self.pheromone[node_id] = min(
                    tau_max,
                    self.pheromone[node_id] + q * diversity_weight * graph_contribution,
                )

    def _make_solution(self, node_ids: Iterable[int]) -> Solution:
        frozen = frozenset(node_ids)
        objectives = evaluate_solution(frozen, self.nodes_by_id, self.tasks, self.satellite_ids, self.params)
        return Solution(node_ids=frozen, objectives=objectives)

    def run(self) -> List[Solution]:
        start_time = time.perf_counter()
        max_iter = int(self.params["max_iter"])
        num_ants = int(self.params["num_ants"])
        archive_size = int(self.params["archive_size"])
        warmup_iter = int(self.params.get("warmup_iter", 30))
        verbose = bool(self.params.get("verbose", True))
        validate_each_solution = bool(self.params.get("validate_each_solution", False))
        validate_interval = int(self.params.get("validate_interval", 20))
        validate_final_archive = bool(self.params.get("validate_final_archive", True))
        episode_lengths: list[int] = []

        for iteration in range(1, max_iter + 1):
            population: List[Solution] = []
            for _ in range(num_ants):
                constructed, episode = self._construct_episode(iteration, max_iter)
                episode_lengths.append(len(episode))
                if iteration <= warmup_iter and self.t_ref is None:
                    self.warmup_lengths.append(len(episode))

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
                if validate_each_solution and not is_feasible(improved, self.conflict_adj):
                    continue

                candidate = self._make_solution(improved)
                population.append(candidate)
                entered_archive = self._candidate_enters_archive(candidate, archive_size)
                if self.t_ref is None:
                    self._pending_episodes.append((episode, entered_archive))
                else:
                    self._store_episode(episode, entered_archive)

            if validate_interval > 0 and iteration % validate_interval == 0:
                for idx, sol in enumerate(population):
                    if not is_feasible(sol.node_ids, self.conflict_adj):
                        raise RuntimeError(
                            "RL-CG-MOACO produced infeasible solution at "
                            f"iteration {iteration}, population index {idx}"
                        )

            if self.t_ref is None and iteration >= warmup_iter:
                self._finish_warmup()

            self._evaporate_pheromone()
            self._update_pheromone_by_archive()

            if (
                self.agent is not None
                and self.t_ref is not None
                and iteration > warmup_iter
                and iteration < self._freeze_start_iter(max_iter)
            ):
                self._set_learning_rate_for_iter(iteration, max_iter)
                for _ in range(int(self.params.get("rl_train_steps_per_iter", 1))):
                    self.agent.train_step()
                if iteration % max(1, int(self.params.get("rl_target_sync_iter", 10))) == 0:
                    self.agent.sync_target()

            if episode_lengths:
                self.avg_episode_length = sum(episode_lengths) / len(episode_lengths)

            if verbose and (iteration == 1 or iteration % max(1, max_iter // 10) == 0 or iteration == max_iter):
                best_total_profit = max((-s.objectives[0] for s in self.archive), default=float("nan"))
                last_loss = self.agent.last_loss if self.agent is not None else None
                loss_text = f"{last_loss:.4g}" if last_loss is not None else "NA"
                print(
                    f"[RL-CG-MOACO] Iter {iteration:>4}/{max_iter}: "
                    f"archive={len(self.archive):>3}, best_f1={best_total_profit:.4f}, "
                    f"T_ref={self.t_ref}, eps={self.last_epsilon:.3f}, "
                    f"kappa={self.last_kappa:.3f}, adv_span={self.last_advantage_span:.3f}, "
                    f"loss={loss_text}"
                )

        if self.t_ref is None:
            self._finish_warmup()

        if validate_final_archive:
            for idx, sol in enumerate(self.archive):
                if not is_feasible(sol.node_ids, self.conflict_adj):
                    raise RuntimeError(
                        "RL-CG-MOACO final archive contains infeasible "
                        f"solution at index {idx}"
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
