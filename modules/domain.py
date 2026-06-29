"""数据的结构"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    task_id: int
    coord_1: float
    coord_2: float
    task_type: int
    duration: float
    profit: float


@dataclass(frozen=True)
class CandidateNode:
    node_id: int
    task_id: int
    sat_id: int
    window_id: int
    start: float
    end: float
    duration: float
    profit: float
    task_duration: float
    coord_1: float
    coord_2: float
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    end_roll: float | None = None
    end_pitch: float | None = None
    end_yaw: float | None = None
