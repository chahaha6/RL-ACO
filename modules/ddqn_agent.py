from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ReplayTransition:
    state_action: list[float]
    reward: float
    next_actions: list[list[float]]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._data: deque[ReplayTransition] = deque(maxlen=max(1, int(capacity)))

    def __len__(self) -> int:
        return len(self._data)

    def append(self, transition: ReplayTransition) -> None:
        self._data.append(transition)

    def sample(self, batch_size: int) -> list[ReplayTransition]:
        size = min(int(batch_size), len(self._data))
        return random.sample(list(self._data), size)


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 128, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DDQNAgent:
    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: Sequence[int] = (128, 128, 64),
        gamma: float = 0.9,
        lr: float = 1e-3,
        replay_capacity: int = 10000,
        batch_size: int = 64,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.input_dim = int(input_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(input_dim, hidden_dims).to(self.device)
        self.target_net = QNetwork(input_dim, hidden_dims).to(self.device)
        self.sync_target()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=float(lr))
        self.replay = ReplayBuffer(replay_capacity)
        self.last_loss: float | None = None

    def set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def add_transition(self, transition: ReplayTransition) -> None:
        self.replay.append(transition)

    def add_transitions(self, transitions: Iterable[ReplayTransition]) -> None:
        for transition in transitions:
            self.add_transition(transition)

    @torch.no_grad()
    def predict(self, features: Sequence[Sequence[float]]) -> list[float]:
        if not features:
            return []
        x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=self.device)
        return self.policy_net(x).detach().cpu().tolist()

    def train_step(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None

        batch = self.replay.sample(self.batch_size)
        states = torch.as_tensor(
            np.asarray([item.state_action for item in batch], dtype=np.float32),
            device=self.device,
        )
        rewards = torch.as_tensor(
            np.asarray([item.reward for item in batch], dtype=np.float32),
            device=self.device,
        )

        current_q = self.policy_net(states)
        targets: list[float] = []
        with torch.no_grad():
            for item in batch:
                if item.done or not item.next_actions:
                    targets.append(float(item.reward))
                    continue
                next_x = torch.as_tensor(
                    np.asarray(item.next_actions, dtype=np.float32),
                    device=self.device,
                )
                best_idx = int(torch.argmax(self.policy_net(next_x)).item())
                next_q = float(self.target_net(next_x[best_idx : best_idx + 1]).item())
                targets.append(float(item.reward) + self.gamma * next_q)

        target_q = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=self.device)
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.last_loss = float(loss.item())
        return self.last_loss
