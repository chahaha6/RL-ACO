from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


OBJECTIVE_KEYS = ("f1", "f2", "f3")
ARCHIVE_PATTERN = re.compile(
    r"(?P<tag>.+?)_t(?P<tasks>\d+)(?:_.*?)?_(?:archive|per)\.csv$",
    re.IGNORECASE,
)


def normalize_tag(raw_tag: str) -> str:
    return raw_tag.strip().lower().replace("-", "_")


def tex_escape(text: object) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def fmt(value: float | str | None, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number) or math.isinf(number):
        return ""
    return f"{number:.{digits}f}"


def parse_archive_filename(path: Path) -> tuple[str, int] | None:
    match = ARCHIVE_PATTERN.match(path.name)
    if not match:
        return None
    return normalize_tag(match.group("tag")), int(match.group("tasks"))


def read_archive_points(path: Path) -> list[dict]:
    points = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Pareto file has no header: {path}")

        has_f_objectives = all(key in reader.fieldnames for key in OBJECTIVE_KEYS)
        has_named_objectives = all(
            key in reader.fieldnames
            for key in ("profit", "load", "attitude")
        )
        if not has_f_objectives and not has_named_objectives:
            raise ValueError(
                f"Pareto file has no recognized objective columns: {path}; "
                f"columns={reader.fieldnames}"
            )

        for line_no, row in enumerate(reader, start=2):
            try:
                if has_f_objectives:
                    f1 = float(row["f1"])
                    f2 = float(row["f2"])
                    f3 = float(row["f3"])
                else:
                    profit = float(row["profit"])
                    f1 = -profit if profit < 0 else profit
                    f2 = float(row["attitude"])
                    f3 = float(row["load"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid objective value in {path} at line {line_no}: {row}"
                ) from exc
            if not all(math.isfinite(value) for value in (f1, f2, f3)):
                raise ValueError(
                    f"Non-finite objective value in {path} at line {line_no}: "
                    f"{(f1, f2, f3)}"
                )
            point = {
                "run": str(row.get("Run", "1")),
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "node_ids": row.get("Node_ids", ""),
            }
            points.append(point)
    return points


def discover_archives(
    input_dir: Path,
    tag_aliases: dict[str, tuple[str, str]],
    case_names: Iterable[str] | None = None,
) -> list[dict]:
    archives = []
    if not input_dir.exists():
        return archives
    allowed_cases = set(case_names) if case_names is not None else None

    for path in sorted(input_dir.glob("**/*.csv")):
        parsed = parse_archive_filename(path)
        if parsed is None:
            continue
        raw_tag, task_count = parsed
        if raw_tag not in tag_aliases:
            continue
        canonical_tag, label = tag_aliases[raw_tag]
        points = read_archive_points(path)
        if not points:
            continue
        if path.parent.resolve() == input_dir.resolve():
            case_name = f"t{task_count}"
        else:
            case_name = path.parent.name
        if allowed_cases is not None and case_name not in allowed_cases:
            continue
        archives.append(
            {
                "path": path,
                "raw_tag": raw_tag,
                "tag": canonical_tag,
                "label": label,
                "case": case_name,
                "task_count": task_count,
                "points": points,
            }
        )
    return archives


def validate_archive_coverage(
    archives: list[dict],
    tag_aliases: dict[str, tuple[str, str]],
    case_names: Iterable[str] | None = None,
) -> None:
    """Require every selected case to contain every configured algorithm/run."""

    if not archives:
        raise FileNotFoundError("No Pareto archive files were found.")

    expected_tags = {
        canonical_tag: label
        for canonical_tag, label in tag_aliases.values()
    }
    selected_cases = set(case_names) if case_names is not None else {
        archive["case"] for archive in archives
    }
    by_case: dict[str, list[dict]] = defaultdict(list)
    for archive in archives:
        by_case[archive["case"]].append(archive)

    missing = []
    duplicate = []
    run_mismatches = []
    for case_name in sorted(selected_cases):
        case_archives = by_case.get(case_name, [])
        by_tag: dict[str, list[dict]] = defaultdict(list)
        for archive in case_archives:
            by_tag[archive["tag"]].append(archive)

        for tag, label in expected_tags.items():
            matches = by_tag.get(tag, [])
            if not matches:
                missing.append(f"{case_name}/{label}")
            elif len(matches) > 1:
                duplicate.append(
                    f"{case_name}/{label}: "
                    + ", ".join(str(item["path"]) for item in matches)
                )

        complete_archives = [
            by_tag[tag][0]
            for tag in expected_tags
            if len(by_tag.get(tag, [])) == 1
        ]
        if complete_archives:
            expected_runs = {
                point["run"] for point in complete_archives[0]["points"]
            }
            for archive in complete_archives[1:]:
                actual_runs = {point["run"] for point in archive["points"]}
                if actual_runs != expected_runs:
                    run_mismatches.append(
                        f"{case_name}/{archive['label']}: "
                        f"runs={sorted(actual_runs)}, expected={sorted(expected_runs)}"
                    )

    errors = []
    if missing:
        errors.append("Missing Pareto data: " + ", ".join(missing))
    if duplicate:
        errors.append("Duplicate Pareto data: " + "; ".join(duplicate))
    if run_mismatches:
        errors.append("Inconsistent run sets: " + "; ".join(run_mismatches))
    if errors:
        raise ValueError("\n".join(errors))


def dominates(a: tuple[float, ...], b: tuple[float, ...], eps: float = 1e-12) -> bool:
    return all(x <= y + eps for x, y in zip(a, b)) and any(x < y - eps for x, y in zip(a, b))


def non_dominated(points: Iterable[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    unique = sorted(set(points))
    kept = []
    for i, point in enumerate(unique):
        if any(i != j and dominates(other, point) for j, other in enumerate(unique)):
            continue
        kept.append(point)
    return kept


def metric_point(raw_point: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert displayed objectives to the minimization form used by HV/IGD."""

    f1_profit, f2_maneuver, f3_load = raw_point
    return (-f1_profit, f2_maneuver, f3_load)


def metric_point_from_dict(point: dict) -> tuple[float, float, float]:
    return metric_point((point["f1"], point["f2"], point["f3"]))


def non_dominated_raw(points: Iterable[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Non-dominated filtering for displayed points where f1 is total profit."""

    unique = sorted(set(points))
    kept = []
    for i, point in enumerate(unique):
        point_metric = metric_point(point)
        if any(
            i != j and dominates(metric_point(other), point_metric)
            for j, other in enumerate(unique)
        ):
            continue
        kept.append(point)
    return kept


def bounds_for_case(archives: list[dict]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values = []
    for archive in archives:
        for point in archive["points"]:
            values.append(metric_point_from_dict(point))
    if not values:
        return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)

    ideal = tuple(min(point[i] for point in values) for i in range(3))
    nadir = tuple(max(point[i] for point in values) for i in range(3))
    nadir = tuple(
        nadir[i] if abs(nadir[i] - ideal[i]) > 1e-12 else ideal[i] + 1.0
        for i in range(3)
    )
    return ideal, nadir


def normalize_points(
    points: Iterable[tuple[float, float, float]],
    ideal: tuple[float, ...],
    nadir: tuple[float, ...],
) -> list[tuple[float, float, float]]:
    normalized = []
    for point in points:
        normalized.append(
            tuple(
                max(0.0, min(1.2, (point[i] - ideal[i]) / (nadir[i] - ideal[i])))
                for i in range(3)
            )
        )
    return normalized


def rectangle_union_area_yz(
    points: list[tuple[float, float, float]],
    ref_y: float,
    ref_z: float,
) -> float:
    if not points:
        return 0.0

    y_values = sorted({p[1] for p in points if p[1] < ref_y} | {ref_y})
    area = 0.0
    for left, right in zip(y_values, y_values[1:]):
        active = [p for p in points if p[1] <= left and p[2] < ref_z]
        if not active:
            continue
        z_min = min(p[2] for p in active)
        area += (right - left) * max(0.0, ref_z - z_min)
    return area


def hypervolume_3d(
    points: Iterable[tuple[float, float, float]],
    ref: tuple[float, float, float] = (1.1, 1.1, 1.1),
) -> float:
    clean = [p for p in non_dominated(points) if all(p[i] < ref[i] for i in range(3))]
    if not clean:
        return 0.0

    x_values = sorted({p[0] for p in clean} | {ref[0]})
    hv = 0.0
    for left, right in zip(x_values, x_values[1:]):
        active = [p for p in clean if p[0] <= left]
        if not active:
            continue
        yz_area = rectangle_union_area_yz(active, ref[1], ref[2])
        hv += (right - left) * yz_area
    return hv


def igd(
    points: Iterable[tuple[float, float, float]],
    reference_points: Iterable[tuple[float, float, float]],
) -> float:
    point_list = list(points)
    ref_list = list(reference_points)
    if not ref_list:
        raise ValueError("IGD reference front is empty")
    if not point_list:
        return float("inf")

    distances = []
    for ref in ref_list:
        distances.append(
            min(
                math.sqrt(sum((ref[i] - point[i]) ** 2 for i in range(3)))
                for point in point_list
            )
        )
    return mean(distances)


def node_count(node_ids: str) -> int:
    if not node_ids:
        return 0
    return len([item for item in node_ids.split() if item.strip()])


def metric_rows_for_archives(archives: list[dict]) -> list[dict]:
    rows = []
    by_case: dict[str, list[dict]] = defaultdict(list)
    for archive in archives:
        by_case[archive["case"]].append(archive)

    for case_name, case_archives in sorted(by_case.items()):
        ideal, nadir = bounds_for_case(case_archives)

        all_norm_points = []
        for archive in case_archives:
            metric_points = [metric_point_from_dict(p) for p in archive["points"]]
            all_norm_points.extend(normalize_points(metric_points, ideal, nadir))
        reference_front = non_dominated(all_norm_points)

        for archive in sorted(case_archives, key=lambda item: item["label"]):
            by_run: dict[str, list[dict]] = defaultdict(list)
            for point in archive["points"]:
                by_run[point["run"]].append(point)

            hv_values = []
            igd_values = []
            archive_sizes = []
            best_f1_values = []
            best_f2_values = []
            best_f3_values = []
            completion_values = []

            for run_id, run_points in sorted(by_run.items()):
                metric_points = [metric_point_from_dict(p) for p in run_points]
                norm_points = normalize_points(metric_points, ideal, nadir)
                hv_values.append(hypervolume_3d(norm_points))
                igd_values.append(igd(norm_points, reference_front))
                archive_sizes.append(float(len(run_points)))
                best_f1_values.append(max(p["f1"] for p in run_points))
                best_f2_values.append(min(p["f2"] for p in run_points))
                best_f3_values.append(min(p["f3"] for p in run_points))
                if archive["task_count"] > 0:
                    completion_values.append(
                        max(node_count(p["node_ids"]) for p in run_points) / archive["task_count"]
                    )

            rows.append(
                {
                    "Case": case_name,
                    "Algorithm": archive["label"],
                    "Tag": archive["tag"],
                    "Runs": len(by_run),
                    "HV_mean": mean(hv_values) if hv_values else 0.0,
                    "HV_std": pstdev(hv_values) if len(hv_values) > 1 else 0.0,
                    "IGD_mean": mean(igd_values) if igd_values else 0.0,
                    "IGD_std": pstdev(igd_values) if len(igd_values) > 1 else 0.0,
                    "Archive_size_mean": mean(archive_sizes) if archive_sizes else 0.0,
                    "Best_f1_mean": mean(best_f1_values) if best_f1_values else 0.0,
                    "Best_f2_mean": mean(best_f2_values) if best_f2_values else 0.0,
                    "Best_f3_mean": mean(best_f3_values) if best_f3_values else 0.0,
                    "Completion_mean": mean(completion_values) if completion_values else 0.0,
                }
            )
    return rows


def write_metrics_csv(rows: list[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Case",
        "Algorithm",
        "Tag",
        "Runs",
        "HV_mean",
        "HV_std",
        "IGD_mean",
        "IGD_std",
        "Archive_size_mean",
        "Best_f1_mean",
        "Best_f2_mean",
        "Best_f3_mean",
        "Completion_mean",
    ]
    with output_file.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_tex(rows: list[dict], output_file: Path, title: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"\\subsection*{{{tex_escape(title)}}}",
        "\\begin{longtable}{llrrrrr}",
        "\\toprule",
        "Case & Algorithm & HV & IGD & Best profit & Best $f_2$ & Completion \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Case & Algorithm & HV & IGD & Best profit & Best $f_2$ & Completion \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    tex_escape(row["Case"]),
                    tex_escape(row["Algorithm"]),
                    fmt(row["HV_mean"]),
                    fmt(row["IGD_mean"]),
                    fmt(row["Best_f1_mean"]),
                    fmt(row["Best_f2_mean"], digits=2),
                    fmt(row["Completion_mean"]),
                ]
            )
            + r" \\"
        )
    lines.extend(["\\bottomrule", "\\end{longtable}", ""])
    output_file.write_text("\n".join(lines), encoding="utf-8")


def generate_metrics_table(
    *,
    root: Path | None = None,
    input_dir: Path | None = None,
    output_dir: Path,
    tag_aliases: dict[str, tuple[str, str]],
    title: str,
    case_names: Iterable[str] | None = None,
) -> list[dict]:
    search_dir = input_dir if input_dir is not None else (root / "results" if root is not None else None)
    if search_dir is None:
        raise ValueError("Either input_dir or root must be provided.")
    selected_cases = set(case_names) if case_names is not None else None
    archives = discover_archives(search_dir, tag_aliases, selected_cases)
    validate_archive_coverage(archives, tag_aliases, selected_cases)
    rows = metric_rows_for_archives(archives)
    write_metrics_csv(rows, output_dir / "metrics_table.csv")
    write_metrics_tex(rows, output_dir / "metrics_table.tex", title)
    print(f"Metrics input directory: {search_dir}")
    print(f"Metrics CSV saved to {output_dir / 'metrics_table.csv'}")
    print(f"Metrics TeX saved to {output_dir / 'metrics_table.tex'}")
    return rows


def collect_plot_points(archives: list[dict]) -> dict[str, dict[str, list[tuple[float, float, float]]]]:
    by_case: dict[str, dict[str, list[tuple[float, float, float]]]] = defaultdict(lambda: defaultdict(list))
    for archive in archives:
        points = [(p["f1"], p["f2"], p["f3"]) for p in archive["points"]]
        by_case[archive["case"]][archive["label"]].extend(non_dominated_raw(points))
    return by_case


def write_pareto_tex(figure_files: list[Path], output_file: Path, title: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"\\subsection*{{{tex_escape(title)}}}"]
    for figure_file in figure_files:
        rel_path = f"../{figure_file.parent.name}/{figure_file.name}"
        caption = figure_file.stem.replace("pareto_front_", "").replace("_", r"\_")
        lines.extend(
            [
                "\\begin{figure}[htbp]",
                "\\centering",
                f"\\includegraphics[width=0.86\\linewidth]{{{rel_path}}}",
                f"\\caption{{Pareto front for {caption}}}",
                "\\end{figure}",
                "",
            ]
        )
    output_file.write_text("\n".join(lines), encoding="utf-8")


def generate_pareto_plots(
    *,
    root: Path | None = None,
    input_dir: Path | None = None,
    output_dir: Path,
    tag_aliases: dict[str, tuple[str, str]],
    title: str,
    case_names: Iterable[str] | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "legend.frameon": False,
        }
    )

    search_dir = input_dir if input_dir is not None else (root / "results" if root is not None else None)
    if search_dir is None:
        raise ValueError("Either input_dir or root must be provided.")
    selected_cases = set(case_names) if case_names is not None else None
    archives = discover_archives(search_dir, tag_aliases, selected_cases)
    validate_archive_coverage(archives, tag_aliases, selected_cases)
    plot_data = collect_plot_points(archives)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_files = []
    algorithm_styles = {
        "CG-MOACO": ("#0F4D92", "o"),
        "MOACO": ("#3775BA", "s"),
        "MODBO": ("#B64342", "^"),
        "MOPSO": ("#42949E", "D"),
        "SPEA2": ("#9A4D8E", "P"),
        "NSGA-II": ("#767676", "X"),
    }
    for case_name, by_algorithm in sorted(plot_data.items()):
        fig = plt.figure(figsize=(7.2, 5.4))
        ax = fig.add_subplot(111, projection="3d")
        for label, points in sorted(by_algorithm.items()):
            nd_points = non_dominated_raw(points)
            if len(nd_points) > 400:
                step = max(1, len(nd_points) // 400)
                nd_points = nd_points[::step]
            if not nd_points:
                continue
            xs = [p[0] for p in nd_points]
            ys = [p[1] for p in nd_points]
            zs = [p[2] for p in nd_points]
            color, marker = algorithm_styles.get(label, (None, "o"))
            ax.scatter(
                xs,
                ys,
                zs,
                s=14,
                alpha=0.72,
                color=color,
                marker=marker,
                label=label,
                depthshade=False,
            )

        ax.set_xlabel("Total profit", labelpad=6)
        ax.set_ylabel("Maneuver cost", labelpad=7)
        ax.set_zlabel("")
        ax.text2D(
            1.08,
            0.53,
            "Load imbalance",
            transform=ax.transAxes,
            rotation=90,
            ha="center",
            va="center",
        )
        ax.set_title(f"Pareto front: {case_name}", fontsize=10, pad=8)
        ax.view_init(elev=24, azim=-55)
        ax.tick_params(labelsize=7, pad=1)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.01, 0.98),
            fontsize=7,
            markerscale=0.9,
            borderaxespad=0,
        )
        fig.subplots_adjust(left=0.01, right=0.90, bottom=0.03, top=0.92)

        figure_file = output_dir / f"pareto_front_{case_name}.png"
        fig.savefig(figure_file, dpi=300, bbox_inches="tight")
        fig.savefig(figure_file.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(figure_file.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        figure_files.append(figure_file)
        print(f"Pareto figure saved to {figure_file}")

    write_pareto_tex(figure_files, output_dir / "pareto_figures.tex", title)
    print(f"Pareto input directory: {search_dir}")
    print(f"Pareto TeX saved to {output_dir / 'pareto_figures.tex'}")
    return figure_files
