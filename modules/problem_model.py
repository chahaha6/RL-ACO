"""CG-MOACO 的问题模型与目标函数定义。

三个目标均统一返回为“最小化”形式：

1. f1：总收益目标
   代码内部统一最小化目标，因此返回 f1 = -scheduled_profit。
   其中 scheduled_profit 表示当前调度方案完成任务的总收益。
   f1 越小，等价于总收益越高。

2. f2：姿态机动代价
   优先使用候选节点中的滚转角和俯仰角计算姿态差，
   并按分段函数计算姿态转换时间。
   如果旧 CSV 中没有姿态角字段，则退回目标坐标距离近似。
   f2 越小，说明任务序列的姿态转换代价越低。

3. f3：卫星负载不均衡度
   默认使用每颗卫星的调度任务数量作为负载，
   并计算各卫星负载的标准差。
   f3 越小，说明任务分配越均衡。
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

from .domain import CandidateNode, Task


# =========================================================
# 默认问题模型参数
# =========================================================
# use_attitude_transition_time:
#   True 时优先使用 outputattitude 里的滚转角、俯仰角计算分段姿态转换时间。
#
# min_transition_time / maneuver_time_per_degree:
#   旧 CSV 没有姿态角字段时的兼容 fallback，使用目标坐标距离近似姿态转换时间。
#
# load_mode:
#   负载计算方式。
#   "task_count" 表示用任务数量作为卫星负载；
#   "duration" 表示用总观测时长作为卫星负载。
# =========================================================
DEFAULT_MODEL_PARAMS = {
    "min_transition_time": 5.0,
    "maneuver_time_per_degree": 0.2,
    "use_attitude_transition_time": True,
    "load_mode": "task_count",  # 可选："task_count" 或 "duration"
}


def attitude_delta_degrees(
    node_a: CandidateNode,
    node_b: CandidateNode,
) -> float | None:
    """Return |roll_b - roll_a| + |pitch_b - pitch_a| in degrees.

    Yaw is intentionally ignored according to the scheduling model.
    For the previous observation, end_roll/end_pitch are used because the
    transition starts after the actual observation segment finishes.
    """

    roll_a = node_a.end_roll if node_a.end_roll is not None else node_a.roll
    pitch_a = node_a.end_pitch if node_a.end_pitch is not None else node_a.pitch
    roll_b = node_b.roll
    pitch_b = node_b.pitch

    if roll_a is None or pitch_a is None or roll_b is None or pitch_b is None:
        return None

    return abs(roll_b - roll_a) + abs(pitch_b - pitch_a)


def attitude_transition_time(delta_degrees: float) -> float:
    """Piecewise attitude transition time in seconds."""

    delta = abs(delta_degrees)
    if delta <= 10.0:
        return 35.0 / 3.0
    if delta < 30.0:
        return delta / 1.5 + 5.0
    if delta < 60.0:
        return delta / 2.0 + 10.0
    if delta < 90.0:
        return delta / 2.5 + 16.0
    return delta / 3.0 + 22.0


def estimate_transition_time(
    node_a: CandidateNode,
    node_b: CandidateNode,
    params: dict | None = None,
) -> float:
    """估计两个候选观测节点之间的姿态机动时间/代价。

    参数
    ----
    node_a:
        前一个任务-卫星-窗口候选节点。

    node_b:
        后一个任务-卫星-窗口候选节点。

    params:
        问题模型参数，主要包括：
        - min_transition_time
        - maneuver_time_per_degree


    如果后续有真实姿态角或姿态转移矩阵，可以直接替换此函数。
    """

    # 合并默认参数和外部传入参数
    p = {**DEFAULT_MODEL_PARAMS, **(params or {})}

    if p.get("use_attitude_transition_time", True):
        delta = attitude_delta_degrees(node_a, node_b)
        if delta is not None:
            return attitude_transition_time(delta)

    # 使用目标坐标之间的欧氏距离近似姿态变化量
    dist = math.hypot(
        node_a.coord_1 - node_b.coord_1,
        node_a.coord_2 - node_b.coord_2,
    )

    # 基础转换时间 + 姿态变化距离对应的额外机动时间
    return p["min_transition_time"] + p["maneuver_time_per_degree"] * dist


def group_solution_by_satellite(
    solution: Iterable[int],
    nodes_by_id: Dict[int, CandidateNode],
) -> dict[int, List[CandidateNode]]:
    """将一个调度方案按照卫星编号分组。

    每颗卫星上的任务会按照开始时间排序，
    便于后续计算相邻任务之间的姿态机动代价。
    """

    grouped: dict[int, List[CandidateNode]] = {}

    # 按照卫星编号收集节点
    for node_id in solution:
        node = nodes_by_id[node_id]
        grouped.setdefault(node.sat_id, []).append(node)

    # 每颗卫星内部按开始时间排序
    for sat_id in grouped:
        grouped[sat_id].sort(key=lambda n: (n.start, n.end, n.task_id))

    return grouped


def calculate_profit(
    solution: Iterable[int],
    nodes_by_id: Dict[int, CandidateNode],
) -> float:
    """计算调度方案的任务总收益。

    对于可行解，每个任务最多只会被调度一次。
    但为了增强鲁棒性，如果某个异常解中同一任务出现多次，
    这里对同一任务只计一次收益，并取该任务出现节点中的最大收益。
    """

    best_profit_by_task: dict[int, float] = {}

    for node_id in solution:
        node = nodes_by_id[node_id]

        # 同一任务只计一次收益，防止异常解重复计分
        best_profit_by_task[node.task_id] = max(
            best_profit_by_task.get(node.task_id, 0.0),
            node.profit,
        )

    return sum(best_profit_by_task.values())


def calculate_maneuver_cost(
    solution: Iterable[int],
    nodes_by_id: Dict[int, CandidateNode],
    params: dict | None = None,
) -> float:
    """计算调度方案的姿态机动总代价。

    1. 先将调度方案按照卫星分组；
    2. 每颗卫星内部按任务开始时间排序；
    3. 对每颗卫星上的相邻任务对，累加姿态转换代价。
    """

    total = 0.0

    # 按卫星分组，并按开始时间排序
    grouped = group_solution_by_satellite(solution, nodes_by_id)

    # 对每颗卫星，累加相邻任务之间的姿态转换代价
    for sat_nodes in grouped.values():
        for prev, curr in zip(sat_nodes, sat_nodes[1:]):
            total += estimate_transition_time(prev, curr, params)

    return total

def node_load_amount(node: CandidateNode, load_mode: str) -> float:
    if load_mode == "task_count":
        return 1.0
    if load_mode == "duration":
        return float(node.task_duration)

    raise ValueError(f"Unknown load_mode: {load_mode}")


def calculate_load_balance(
    solution: Iterable[int],
    nodes_by_id: Dict[int, CandidateNode],
    satellite_ids: Sequence[int],
    load_mode: str = "task_count",
) -> float:

    if not satellite_ids:
        raise ValueError("satellite_ids is empty")

    loads = {sat_id: 0.0 for sat_id in satellite_ids}

    for node_id in solution:
        if node_id not in nodes_by_id:
            raise ValueError(f"Unknown node_id in solution: {node_id}")

        node = nodes_by_id[node_id]

        if node.sat_id not in loads:
            raise ValueError(
                f"Node {node_id} uses satellite {node.sat_id}, "
                f"but satellite_ids={list(satellite_ids)}"
            )

        loads[node.sat_id] += node_load_amount(node, load_mode)

    values = list(loads.values())
    mean_load = sum(values) / len(values)

    return math.sqrt(
        sum((value - mean_load) ** 2 for value in values) / len(values)
    )

def is_feasible(
    solution: Iterable[int],
    conflict_adj: dict[int, set[int]],
) -> bool:

    selected = set(solution)

    # 检查每个已选节点是否与其他已选节点冲突
    for node_id in selected:
        if conflict_adj.get(node_id, set()) & selected:
            return False

    return True


def evaluate_solution(
    solution: Iterable[int],
    nodes_by_id: Dict[int, CandidateNode],
    tasks: Dict[int, Task],
    satellite_ids: Sequence[int],
    params: dict | None = None,
) -> Tuple[float, float, float]:
    """计算调度方案的三目标函数值。

        f1：总收益目标，内部为 -scheduled_profit，越小越好；
        f2：姿态机动代价，越小越好；
        f3：负载不均衡度，越小越好。
    """
    solution = list(solution)

    # 合并默认参数和外部参数
    p = {**DEFAULT_MODEL_PARAMS, **(params or {})}

    # 当前方案完成任务的收益
    scheduled_profit = calculate_profit(solution, nodes_by_id)

    # 目标 1：总收益。
    # 原始目标是最大化收益；算法框架统一最小化，因此内部使用负收益。
    f1 = -scheduled_profit

    # 目标 2：姿态机动代价
    f2 = calculate_maneuver_cost(solution, nodes_by_id, p)

    # 目标 3：负载不均衡度
    f3 = calculate_load_balance(
        solution,
        nodes_by_id,
        satellite_ids,
        p.get("load_mode", "task_count"),
    )

    return (f1, f2, f3)


def task_completion_rate(
    solution: Iterable[int],
    total_task_count: int,
    nodes_by_id: Dict[int, CandidateNode],
) -> float:
    """计算任务完成率。

    任务完成率只统计任务数量，不考虑任务收益。
    它和 f1 不完全相同：

        f1 关注总收益；
        completion_rate 关注任务数量完成情况。
    """

    # 提取当前方案中被调度的不同 task_id
    scheduled_tasks = {
        nodes_by_id[node_id].task_id
        for node_id in solution
    }

    if total_task_count <= 0:
        return 0.0

    return len(scheduled_tasks) / total_task_count
