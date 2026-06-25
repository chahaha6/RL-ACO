"""CG-MOACO 的问题模型与目标函数定义。

三个目标均统一返回为“最小化”形式：

1. f1：未完成收益率
   f1 = 1 - scheduled_profit / total_profit
   其中 scheduled_profit 表示当前调度方案完成任务的总收益。
   f1 越小，说明完成收益越高。

2. f2：姿态机动代价
   由于当前数据集中没有完整的姿态转移矩阵，
   这里用同一颗卫星上连续任务之间的目标坐标距离近似姿态机动代价。
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
# min_transition_time:
#   两个连续任务之间的最小基础姿态转换时间。
#
# maneuver_time_per_degree:
#   姿态机动代价的比例系数。
#   当前代码使用目标坐标之间的欧氏距离近似姿态变化量，
#   再乘以该系数得到额外机动代价。
#
# load_mode:
#   负载计算方式。
#   "task_count" 表示用任务数量作为卫星负载；
#   "duration" 表示用总观测时长作为卫星负载。
# =========================================================
DEFAULT_MODEL_PARAMS = {
    "min_transition_time": 5.0,
    "maneuver_time_per_degree": 0.2,
    "load_mode": "task_count",  # 可选："task_count" 或 "duration"
}


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


def calculate_load_balance(
    solution: Iterable[int],
    nodes_by_id: Dict[int, CandidateNode],
    satellite_ids: Sequence[int],
    load_mode: str = "task_count",
) -> float:
    """计算卫星负载不均衡度。

        负载计算方式：
        - "task_count"：每颗卫星调度任务数量；
        - "duration"：每颗卫星总观测时长。

    """

    # 初始化每颗卫星的负载
    loads = {sat_id: 0.0 for sat_id in satellite_ids}

    # 统计每颗卫星的负载
    for node_id in solution:
        node = nodes_by_id[node_id]

        if load_mode == "duration":
            # 使用观测窗口时长作为负载
            loads[node.sat_id] += node.duration
        else:
            # 默认使用任务数量作为负载
            loads[node.sat_id] += 1.0

    values = list(loads.values())

    if not values:
        return 0.0

    # 计算负载平均值
    mean_load = sum(values) / len(values)

    # 返回负载标准差
    return math.sqrt(
        sum((value - mean_load) ** 2 for value in values) / len(values)
    )


def is_feasible(
    solution: Iterable[int],
    conflict_adj: dict[int, set[int]],
) -> bool:
    """判断调度方案是否满足冲突图约束。

    参数
    ----
    solution:
        调度方案，表示为被选中的 node_id 集合或列表。

    conflict_adj:
        冲突图邻接表。
        conflict_adj[node_id] 表示与 node_id 冲突的所有节点集合。

    返回
    ----
    bool:
        True 表示方案可行；
        False 表示方案中存在冲突节点。

    如果一个方案中存在两个互相冲突的节点，
    即二者在冲突图中有边相连，则该方案不可行。
    """

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

        f1：未完成收益率，越小越好；
        f2：姿态机动代价，越小越好；
        f3：负载不均衡度，越小越好。
    """

    # 合并默认参数和外部参数
    p = {**DEFAULT_MODEL_PARAMS, **(params or {})}

    # 所有任务的总收益
    total_profit = sum(task.profit for task in tasks.values())

    # 当前方案完成任务的收益
    scheduled_profit = calculate_profit(solution, nodes_by_id)

    # 目标 1：未完成收益率
    # 原始目标是最大化收益，这里转成最小化形式
    f1 = 1.0 - scheduled_profit / total_profit if total_profit > 0 else 1.0

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

        f1 关注收益完成情况；
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