from __future__ import annotations


# =========================================================
# Comparison dataset selection
#
# Both generate_metrics_table.py and plot_pareto_front.py use this file.
# Select one dataset prefix and list the satellite/task cases to include.
# =========================================================

DEFAULT_DATASET_PREFIX = "area"
CASE_LIST = [
    (5, 100),
    (5, 200),
    (5, 300),
    (5, 400),
    (5, 500),
    (5, 600),
    (5, 700),
    (5, 800),
    (5, 900),
    (5, 1000),
]

# DEFAULT_DATASET_PREFIX = "world"
# CASE_LIST = [
#     (5, 200),
#     (5, 400),
#     (5, 600),
#     (5, 800),
#     (5, 1000),
# ]


def selected_case_names() -> set[str]:
    prefix = DEFAULT_DATASET_PREFIX.strip()
    if not prefix:
        raise ValueError("DEFAULT_DATASET_PREFIX cannot be empty")
    if not CASE_LIST:
        raise ValueError("CASE_LIST cannot be empty")

    names = set()
    for satellite_count, task_count in CASE_LIST:
        if satellite_count <= 0 or task_count <= 0:
            raise ValueError(f"Invalid case: {(satellite_count, task_count)}")
        names.add(f"{prefix}_s{satellite_count}_t{task_count}")
    return names


SELECTED_CASE_NAMES = selected_case_names()
