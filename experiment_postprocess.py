from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


OBJECTIVE_KEYS = ("f1", "f2", "f3")
ARCHIVE_PATTERN = re.compile(r"(?P<tag>.+)_t(?P<tasks>\d+)_archive\.csv$", re.IGNORECASE)


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
        if not reader.fieldnames or not all(key in reader.fieldnames for key in OBJECTIVE_KEYS):
            return []

        for row in reader:
            try:
                point = {
                    "run": str(row.get("Run", "1")),
                    "f1": float(row["f1"]),
                    "f2": float(row["f2"]),
                    "f3": float(row["f3"]),
                    "node_ids": row.get("Node_ids", ""),
                }
            except (TypeError, ValueError):
                continue
            points.append(point)
    return points


def discover_archives(
    input_dir: Path,
    tag_aliases: dict[str, tuple[str, str]],
) -> list[dict]:
    archives = []
    if not input_dir.exists():
        return archives

    for path in sorted(input_dir.glob("**/*_archive.csv")):
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


def bounds_for_case(archives: list[dict]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values = []
    for archive in archives:
        for point in archive["points"]:
            values.append((point["f1"], point["f2"], point["f3"]))
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
    if not point_list or not ref_list:
        return 0.0

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
            raw_points = [(p["f1"], p["f2"], p["f3"]) for p in archive["points"]]
            all_norm_points.extend(normalize_points(raw_points, ideal, nadir))
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
                raw_points = [(p["f1"], p["f2"], p["f3"]) for p in run_points]
                norm_points = normalize_points(raw_points, ideal, nadir)
                hv_values.append(hypervolume_3d(norm_points))
                igd_values.append(igd(norm_points, reference_front))
                archive_sizes.append(float(len(run_points)))
                best_f1_values.append(min(p["f1"] for p in run_points))
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
        "Case & Algorithm & HV & IGD & Best $f_1$ & Best $f_2$ & Completion \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Case & Algorithm & HV & IGD & Best $f_1$ & Best $f_2$ & Completion \\\\",
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
) -> list[dict]:
    search_dir = input_dir if input_dir is not None else (root / "results" if root is not None else None)
    if search_dir is None:
        raise ValueError("Either input_dir or root must be provided.")
    archives = discover_archives(search_dir, tag_aliases)
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
        by_case[archive["case"]][archive["label"]].extend(non_dominated(points))
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
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    search_dir = input_dir if input_dir is not None else (root / "results" if root is not None else None)
    if search_dir is None:
        raise ValueError("Either input_dir or root must be provided.")
    archives = discover_archives(search_dir, tag_aliases)
    plot_data = collect_plot_points(archives)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_files = []
    for case_name, by_algorithm in sorted(plot_data.items()):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        for label, points in sorted(by_algorithm.items()):
            nd_points = non_dominated(points)
            if len(nd_points) > 400:
                step = max(1, len(nd_points) // 400)
                nd_points = nd_points[::step]
            if not nd_points:
                continue
            xs = [p[0] for p in nd_points]
            ys = [p[1] for p in nd_points]
            zs = [p[2] for p in nd_points]
            ax.scatter(xs, ys, zs, s=12, alpha=0.75, label=label)

        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
        ax.set_title(f"{title}: {case_name}")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        figure_file = output_dir / f"pareto_front_{case_name}.png"
        fig.savefig(figure_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figure_files.append(figure_file)
        print(f"Pareto figure saved to {figure_file}")

    write_pareto_tex(figure_files, output_dir / "pareto_figures.tex", title)
    print(f"Pareto input directory: {search_dir}")
    print(f"Pareto TeX saved to {output_dir / 'pareto_figures.tex'}")
    return figure_files
