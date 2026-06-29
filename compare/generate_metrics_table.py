from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_postprocess import generate_metrics_table


EXPERIMENT_DIR = Path(__file__).resolve().parent


TAG_ALIASES = {
    "cg_moaco": ("cg_moaco", "CG-MOACO"),
    "modbo": ("modbo", "MODBO"),
    "sfmodbo": ("modbo", "MODBO"),
    "mopso": ("mopso", "MOPSO"),
    "spea2": ("spea2", "SPEA2"),
    # "moaco": ("moaco", "MOACO"),
    "nsga2": ("nsga2", "NSGA-II"),
}


ALGORITHM_ORDER = [
    "CG-MOACO",
    # "MOACO",
    "MODBO",
    "MOPSO",
    "SPEA2",
    "NSGA-II",
]

ALGORITHM_STYLE = {
    "CG-MOACO": ("#0F4D92", "o", 2.4),
    "MOACO": ("#3775BA", "s", 1.5),
    "MODBO": ("#B64342", "^", 1.5),
    "MOPSO": ("#42949E", "D", 1.5),
    "SPEA2": ("#9A4D8E", "P", 1.5),
    "NSGA-II": ("#767676", "X", 1.5),
}

METRIC_PANELS = [
    ("HV_mean", "HV_std", "HV (higher is better)", 1.0),
    ("IGD_mean", "IGD_std", "IGD (lower is better)", 1.0),
    ("Best_f1_mean", None, "Total profit (higher is better)", 1.0),
    ("Best_f2_mean", None, "Maneuver cost (lower is better)", 1.0),
    ("Best_f3_mean", None, "Load imbalance (lower is better)", 1.0),
    ("Completion_mean", None, "Completion rate (%)", 100.0),
]


def task_count(case_name: str) -> int:
    match = re.search(r"t(\d+)$", case_name)
    if match is None:
        raise ValueError(f"Cannot parse task count from case: {case_name}")
    return int(match.group(1))


def plot_metrics_comparison(rows: list[dict], output_dir: Path) -> list[Path]:
    """Plot HV, IGD, objectives, and completion trends across task scales."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )

    cases = sorted({str(row["Case"]) for row in rows}, key=task_count)
    indexed = {(str(row["Case"]), str(row["Algorithm"])): row for row in rows}
    missing = [
        (case_name, algorithm)
        for case_name in cases
        for algorithm in ALGORITHM_ORDER
        if (case_name, algorithm) not in indexed
    ]
    if missing:
        missing_text = ", ".join(f"{case}/{algorithm}" for case, algorithm in missing)
        raise ValueError(f"Missing metric rows: {missing_text}")

    x_values = [task_count(case_name) for case_name in cases]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), sharex=True)

    for panel_index, (ax, panel) in enumerate(zip(axes.flat, METRIC_PANELS)):
        metric_key, std_key, ylabel, scale = panel
        for algorithm in ALGORITHM_ORDER:
            color, marker, linewidth = ALGORITHM_STYLE[algorithm]
            values = [float(indexed[(case_name, algorithm)][metric_key]) * scale for case_name in cases]
            ax.plot(
                x_values,
                values,
                color=color,
                marker=marker,
                markersize=4.2,
                markeredgecolor="white",
                markeredgewidth=0.45,
                linewidth=linewidth,
                alpha=1.0 if algorithm == "CG-MOACO" else 0.88,
                label=algorithm,
                zorder=4 if algorithm == "CG-MOACO" else 2,
            )
            if std_key is not None:
                std_values = [
                    float(indexed[(case_name, algorithm)][std_key]) * scale
                    for case_name in cases
                ]
                lower = [value - std for value, std in zip(values, std_values)]
                upper = [value + std for value, std in zip(values, std_values)]
                ax.fill_between(
                    x_values,
                    lower,
                    upper,
                    color=color,
                    alpha=0.10 if algorithm == "CG-MOACO" else 0.045,
                    linewidth=0,
                    zorder=1,
                )

        ax.set_ylabel(ylabel)
        ax.set_xticks(x_values)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.7)
        ax.tick_params(direction="out", length=2.5, width=0.7)
        ax.text(
            -0.14,
            1.04,
            chr(ord("a") + panel_index),
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )

    for ax in axes[1, :]:
        ax.set_xlabel("Number of tasks")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 1.005),
        columnspacing=1.2,
        handlelength=2.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.35, w_pad=1.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "metrics_comparison"
    output_files = []
    for extension, dpi in (("svg", 300), ("pdf", 300), ("png", 600)):
        output_file = output_base.with_suffix(f".{extension}")
        fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
        output_files.append(output_file)
        print(f"Metrics figure saved to {output_file}")
    plt.close(fig)
    return output_files


if __name__ == "__main__":
    metric_rows = generate_metrics_table(
        input_dir=EXPERIMENT_DIR,
        output_dir=EXPERIMENT_DIR,
        tag_aliases=TAG_ALIASES,
        title="Comparison experiment metrics",
    )
    plot_metrics_comparison(metric_rows, EXPERIMENT_DIR)
