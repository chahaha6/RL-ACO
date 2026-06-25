
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Dict, Iterable, List

try:
    from .domain import CandidateNode, Task
    from .utils import parse_time_to_seconds
except ImportError:
    from domain import CandidateNode, Task
    from utils import parse_time_to_seconds

DEFAULT_DATASET_PREFIX = "area"
SATELLITE_COUNT = 5
TASK_COUNT = 500
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_ROOT = PROJECT_ROOT / "Local_Data"
CSV_DATA_ROOT = PROJECT_ROOT / "CSV_DATA"


@dataclass(frozen=True)
class TimeWindow:
    task_id: int
    sat_id: int
    window_id: int
    start: float
    end: float
    duration: float


def dataset_name(satellite_count: int, task_count: int, prefix: str = DEFAULT_DATASET_PREFIX) -> str:
    """Build the Local_Data/CSV_DATA subfolder name for one experiment scale."""

    if satellite_count <= 0:
        raise ValueError("satellite_count must be positive")
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    return f"{prefix}_s{satellite_count}_t{task_count}"


def resolve_dataset_dir(
    root_dir: str | Path,
    satellite_count: int,
    task_count: int,
    prefix: str = DEFAULT_DATASET_PREFIX,
) -> Path:
    """Resolve one dataset subdirectory such as area_s5_t100."""

    path = Path(root_dir) / dataset_name(satellite_count, task_count, prefix)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    return path


def resolve_csv_dataset_dir(
    csv_root: str | Path,
    satellite_count: int,
    task_count: int,
    prefix: str = DEFAULT_DATASET_PREFIX,
) -> Path:
    """Resolve and create the CSV output subdirectory for one experiment scale."""

    path = Path(csv_root) / dataset_name(satellite_count, task_count, prefix)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_candidate_nodes_csv(
    local_data_root: str | Path,
    csv_root: str | Path,
    satellite_count: int,
    task_count: int,
    prefix: str = DEFAULT_DATASET_PREFIX,
) -> tuple[Dict[int, Task], List[CandidateNode], Path]:
    """Read one raw dataset and write its candidate_nodes.csv under CSV_DATA."""

    local_dataset_dir = resolve_dataset_dir(local_data_root, satellite_count, task_count, prefix)
    csv_dataset_dir = resolve_csv_dataset_dir(csv_root, satellite_count, task_count, prefix)
    tasks = read_tasklist(local_dataset_dir / "tasklist.txt")
    time_windows = read_all_timewindows(local_dataset_dir)
    nodes = build_candidate_nodes(tasks, time_windows)
    save_tasks_csv(tasks.values(), csv_dataset_dir / "tasks.csv")
    save_candidate_nodes_csv(nodes, csv_dataset_dir / "candidate_nodes.csv")
    return tasks, nodes, csv_dataset_dir


def read_tasklist(file_path: str | Path) -> Dict[int, Task]:
    """Read tasklist.txt.

    The uploaded data has six columns:
    task_id, coord_1, coord_2, task_type, duration, profit.
    """

    path = Path(file_path)
    tasks: Dict[int, Task] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"Invalid task line {line_no}: {raw!r}")
            task_id = int(parts[0])
            tasks[task_id] = Task(
                task_id=task_id,
                coord_1=float(parts[1]),
                coord_2=float(parts[2]),
                task_type=int(float(parts[3])),
                duration=float(parts[4]),
                profit=float(parts[5]),
            )
    return tasks


def _is_int_line(line: str) -> bool:
    parts = line.split()
    return len(parts) == 1 and parts[0].lstrip("+-").isdigit()


def read_timewindow_file(file_path: str | Path) -> List[TimeWindow]:
    """Read one outputtimewindow_*.txt file."""

    path = Path(file_path)
    lines = [line.rstrip("\n") for line in path.open("r", encoding="utf-8") if line.strip()]
    if not lines:
        return []
    sat_header = lines[0].strip().lower()
    if not sat_header.startswith("s"):
        raise ValueError(f"Expected satellite header like 's1' in {path}, got {sat_header!r}")
    sat_id = int(sat_header[1:])

    windows: List[TimeWindow] = []
    current_task_id: int | None = None
    per_task_window_count: dict[int, int] = {}

    for raw in lines[1:]:
        line = raw.strip()
        if _is_int_line(line):
            current_task_id = int(line)
            per_task_window_count.setdefault(current_task_id, 0)
            continue
        if current_task_id is None:
            raise ValueError(f"Time-window row appears before task id in {path}: {line!r}")
        parts = line.split()
        if len(parts) < 13:
            raise ValueError(f"Invalid time-window row in {path}: {line!r}")
        start = parse_time_to_seconds(parts[0:6])
        end = parse_time_to_seconds(parts[6:12])
        duration = float(parts[12])
        window_id = per_task_window_count[current_task_id]
        per_task_window_count[current_task_id] += 1
        windows.append(
            TimeWindow(
                task_id=current_task_id,
                sat_id=sat_id,
                window_id=window_id,
                start=start,
                end=end,
                duration=duration,
            )
        )
    return windows


def read_all_timewindows(data_dir: str | Path) -> List[TimeWindow]:
    """Read all outputtimewindow_*.txt files under data_dir."""

    data_path = Path(data_dir)
    files = sorted(data_path.glob("outputtimewindow_*.txt"))
    if not files:
        raise FileNotFoundError(f"No outputtimewindow_*.txt files found in {data_path}")
    all_windows: List[TimeWindow] = []
    for file_path in files:
        all_windows.extend(read_timewindow_file(file_path))
    return all_windows


def build_candidate_nodes(tasks: Dict[int, Task], time_windows: Iterable[TimeWindow]) -> List[CandidateNode]:
    """Expand time windows into task-satellite-window candidate nodes."""

    nodes: List[CandidateNode] = []
    for tw in sorted(time_windows, key=lambda w: (w.sat_id, w.task_id, w.start, w.end)):
        task = tasks.get(tw.task_id)
        if task is None:
            # Ignore windows for tasks not present in tasklist.
            continue
        if tw.duration + 1e-9 < task.duration:
            # The observation cannot finish inside this visibility window.
            continue
        node = CandidateNode(
            node_id=len(nodes),
            task_id=tw.task_id,
            sat_id=tw.sat_id,
            window_id=tw.window_id,
            start=tw.start,
            end=tw.end,
            duration=tw.duration,
            profit=task.profit,
            task_duration=task.duration,
            coord_1=task.coord_1,
            coord_2=task.coord_2,
        )
        nodes.append(node)
    return nodes


def save_candidate_nodes_csv(nodes: Iterable[CandidateNode], file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "node_id", "task_id", "sat_id", "window_id", "start", "end", "duration",
            "profit", "task_duration", "coord_1", "coord_2",
        ])
        for n in nodes:
            writer.writerow([
                n.node_id, n.task_id, n.sat_id, n.window_id, n.start, n.end, n.duration,
                n.profit, n.task_duration, n.coord_1, n.coord_2,
            ])


def save_tasks_csv(tasks: Iterable[Task], file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "coord_1", "coord_2", "task_type", "duration", "profit"])
        for task in sorted(tasks, key=lambda t: t.task_id):
            writer.writerow([
                task.task_id,
                task.coord_1,
                task.coord_2,
                task.task_type,
                task.duration,
                task.profit,
            ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess one Local_Data subdataset into CSV_DATA.")
    parser.add_argument(
        "--satellites",
        "-s",
        type=int,
        default=SATELLITE_COUNT,
        help="Satellite count in dataset name, e.g. 5 for area_s5_t300.",
    )
    parser.add_argument(
        "--tasks",
        "-t",
        type=int,
        default=TASK_COUNT,
        help="Task count in dataset name, e.g. 300 for area_s5_t300.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_dataset_dir(LOCAL_DATA_ROOT, args.satellites, args.tasks)
    tasks, nodes, csv_dataset_dir = prepare_candidate_nodes_csv(
        LOCAL_DATA_ROOT,
        CSV_DATA_ROOT,
        args.satellites,
        args.tasks,
    )
    print(f"Read: {dataset_dir}")
    print(f"Tasks: {len(tasks)}, candidate nodes: {len(nodes)}")
    print(f"Saved: {csv_dataset_dir / 'tasks.csv'}")
    print(f"Saved: {csv_dataset_dir / 'candidate_nodes.csv'}")


if __name__ == "__main__":
    main()
