from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
import numpy as np

plt.rcParams.update(
    {
        "font.size": 42,
        "axes.labelsize": 46,
        "xtick.labelsize": 38,
        "ytick.labelsize": 38,
        "legend.fontsize": 30,
    }
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = BASE_DIR / "results"
DEFAULT_SUMMARY_DIR = DEFAULT_RESULTS_ROOT / "summaries"
DEFAULT_REAL_SUMMARY_DIR = DEFAULT_RESULTS_ROOT / "real_summaries"
DEFAULT_FIGURE_DIR = DEFAULT_RESULTS_ROOT / "figures"

METHOD_LABELS = {
    "baseline_HD-PI": "HD-PI",
    "baseline_RH": "RH",
    "baseline_UH-Simplex": "UH-Simplex",
    "baseline_UH-Random": "UH-Random",
    "shared_utility_range_HD-PI": "SUR-HD-PI",
    "shared_utility_range_RH": "SUR-RH",
    "shared_utility_range_UH-Simplex": "SUR-UH-Simplex",
    "shared_utility_range_UH-Random": "SUR-UH-Random",
    "sharedq": "SharedQ",
}

METHOD_ORDER = list(METHOD_LABELS)
METHOD_ALIASES = {
    "sharedq": "sharedq",
    "SharedQ": "sharedq",
}
MARKERS = ["+", "x", "o", "s", "^", "D", "v", "*", "h"]
LINESTYLES = ["-", "-", "-", "-", "--", "--", "--", "--", "-."]
LINE_COLORS = {
    "baseline_HD-PI": "#1f77b4",
    "baseline_RH": "#ff7f0e",
    "baseline_UH-Simplex": "#2ca02c",
    "baseline_UH-Random": "#d62728",
    "shared_utility_range_HD-PI": "#9467bd",
    "shared_utility_range_RH": "#8c564b",
    "shared_utility_range_UH-Simplex": "#e377c2",
    "shared_utility_range_UH-Random": "#17becf",
    "sharedq": "#111111",
}
METHOD_LINE_STYLES = {
    "shared_utility_range_UH-Simplex": {
        "marker": "X",
        "linestyle": (0, (5, 2)),
        "linewidth": 2.6,
        "markersize": 12,
    },
    "sharedq": {
        "marker": "P",
        "linestyle": (0, (3, 1, 1, 1)),
        "linewidth": 2.8,
        "markersize": 12,
    },
}
HATCHES = ["", "////", "\\\\\\\\", "xxxx", "....", "++++", "----", "oooo", "****"]
BAR_FACES = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#111111",
]
AXIS_LABELS = {
    "alpha_beta_ratio": "\u03b1/\u03b2",
}


def save_figure_both(fig: plt.Figure, output_path: Path, **kwargs: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, **kwargs)
    print(f"wrote {pdf_path}", flush=True)


def normalize_method_name(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


def read_table(path: Path) -> tuple[str, list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not fieldnames:
        return "", [], []
    first_col = fieldnames[0]
    rows: list[dict[str, str]] = []
    for row in raw_rows:
        normalized = dict(row)
        for old, new in METHOD_ALIASES.items():
            if old in normalized and new not in normalized:
                normalized[new] = normalized.pop(old)
        rows.append(normalized)
    method_candidates = [normalize_method_name(col) for col in fieldnames[1:]]
    methods = [
        method
        for method in dict.fromkeys(method_candidates)
        if any(str(row.get(method, "")).strip() for row in rows)
    ]
    methods = [method for method in METHOD_ORDER if method in methods] + sorted(
        method for method in methods if method not in METHOD_ORDER
    )
    return first_col, methods, rows


def parse_float(value: object) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        result = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def numeric_x(labels: list[str]) -> tuple[np.ndarray, list[str]]:
    values: list[float] = []
    all_numeric = True
    for label in labels:
        text = label.lower().replace("d", "")
        try:
            if text.endswith("k"):
                values.append(float(text[:-1]) * 1_000)
            elif text.endswith("w"):
                values.append(float(text[:-1]) * 10_000)
            elif text.endswith("m"):
                values.append(float(text[:-1]) * 1_000_000)
            else:
                values.append(float(text))
        except ValueError:
            all_numeric = False
            break
    if all_numeric:
        return np.asarray(values, dtype=float), labels
    return np.arange(len(labels), dtype=float), labels


def set_log_if_positive(ax: plt.Axes, series_values: Iterable[float]) -> None:
    values = [value for value in series_values if value > 0]
    if len(values) >= 2:
        ax.set_yscale("log", base=2)


def line_style_for(method: str, index: int) -> dict[str, object]:
    style = {
        "marker": MARKERS[index % len(MARKERS)],
        "linestyle": LINESTYLES[index % len(LINESTYLES)],
        "linewidth": 2.4,
        "markersize": 11,
        "color": LINE_COLORS.get(method, "#111111"),
    }
    style.update(METHOD_LINE_STYLES.get(method, {}))
    return style


def plot_lines(ax: plt.Axes, table_path: Path, ylabel: str) -> None:
    first_col, methods, rows = read_table(table_path)
    if first_col == "dimension":
        methods = [method for method in methods
                   if method.lower().replace("-", "_") not in {"b_ave", "sur_ave"}]
    if first_col == "alpha_beta_ratio":
        rows = [
            row
            for row in rows
            if (parse_float(row.get(first_col)) is not None and parse_float(row.get(first_col)) <= 5.0)
        ]
    labels = [row[first_col] for row in rows]
    if first_col == "dataset_size":
        x = np.arange(len(labels), dtype=float)
        display_labels = ["500k" if label.lower() == "50w" else label for label in labels]
    else:
        x, display_labels = numeric_x(labels)
    all_values: list[float] = []

    for i, method in enumerate(methods):
        style = line_style_for(method, i)
        y = [parse_float(row.get(method)) for row in rows]
        valid_x = np.asarray([x[j] for j, value in enumerate(y) if value is not None], dtype=float)
        valid_y = np.asarray([float(value) for value in y if value is not None], dtype=float)
        if len(valid_y) == 0:
            continue
        all_values.extend(valid_y.tolist())
        ax.plot(
            valid_x,
            valid_y,
            marker=style["marker"],
            color=style["color"],
            markerfacecolor="none",
            markeredgecolor=style["color"],
            linestyle=style["linestyle"],
            linewidth=float(style["linewidth"]) * 1.7,
            markersize=float(style["markersize"]) * 2.3,
            label=METHOD_LABELS.get(method, method),
        )

    ax.set_xlabel(AXIS_LABELS.get(first_col, first_col.replace("_", " ")))
    ax.set_ylabel(ylabel)
    if first_col == "alpha_beta_ratio":
        ax.set_xticks(range(1, 6))
    else:
        ax.set_xticks(x, display_labels, rotation=30 if first_col == "dataset_size" else 0)
    ax.tick_params(direction="in", top=True, right=True)
    set_log_if_positive(ax, all_values)
    if first_col == "alpha_beta_ratio":
        # Narrow ranges may contain no powers of two for the default log ticks.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(NullLocator())


def plot_bars(ax: plt.Axes, table_path: Path, ylabel: str) -> None:
    first_col, methods, rows = read_table(table_path)
    if first_col == "type_ratio":
        selected_ratios = ["2:4:4", "4:2:4", "4:4:2"]
        rows_by_ratio = {row[first_col]: row for row in rows}
        rows = [rows_by_ratio[ratio] for ratio in selected_ratios]
    if first_col == "alpha_beta_ratio":
        rows = [
            row
            for row in rows
            if (parse_float(row.get(first_col)) is not None and parse_float(row.get(first_col)) <= 5.0)
        ]
    labels = [row[first_col] for row in rows]
    x = np.arange(len(labels), dtype=float)
    width = min(0.8 / max(len(methods), 1), 0.11)

    all_values: list[float] = []
    for i, method in enumerate(methods):
        y = [parse_float(row.get(method)) for row in rows]
        values = [0.0 if value is None else float(value) for value in y]
        all_values.extend(value for value in values if value > 0)
        offsets = x + (i - (len(methods) - 1) / 2.0) * width
        ax.bar(
            offsets,
            values,
            width=width,
            color=BAR_FACES[i % len(BAR_FACES)],
            edgecolor="black",
            linewidth=0.45,
            alpha=0.88,
            label=METHOD_LABELS.get(method, method),
        )

    ax.set_xlabel(AXIS_LABELS.get(first_col, first_col.replace("_", " ")))
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    if first_col == "type_ratio":
        ax.set_xticklabels(labels, rotation=45, ha="right")
    else:
        ax.set_xticklabels(labels, rotation=20 if any(len(label) > 6 for label in labels) else 0)
    ax.tick_params(direction="in", top=True, right=True)
    set_log_if_positive(ax, all_values)


def paired_plot(
    time_path: Path,
    question_path: Path,
    output_path: Path,
    title: str,
    style: str,
    show_legend: bool = True,
) -> None:
    if not time_path.exists() or not question_path.exists():
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=False)
    if style == "bar":
        plot_bars(axes[0], time_path, "Time (seconds)")
        plot_bars(axes[1], question_path, "# of questions")
    else:
        plot_lines(axes[0], time_path, "Time (seconds)")
        plot_lines(axes[1], question_path, "# of questions")

    handles, labels = axes[1].get_legend_handles_labels()
    if handles and show_legend:
        ncol = min(len(labels), 3)
        top_margin = 0.68 if len(labels) > 4 else 0.80
        fig.subplots_adjust(top=top_margin, bottom=0.20, left=0.12, right=0.98, wspace=0.50)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=ncol,
            frameon=False,
            handlelength=1.8,
            markerscale=0.5,
            columnspacing=0.8,
            labelspacing=0.45,
        )
    else:
        fig.subplots_adjust(top=0.68, bottom=0.20, left=0.12, right=0.98, wspace=0.50)
    # Explicitly include the full rotated axis-label extents. Without a legend,
    # the default tight crop can truncate labels taller than the plotting area.
    if not show_legend:
        from matplotlib.transforms import Bbox
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = [fig.get_tightbbox(renderer)]
        bounds.extend(label.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
                      for ax in axes for label in (ax.xaxis.label, ax.yaxis.label))
        crop = Bbox.union(bounds).padded(0.15)
        save_figure_both(fig, output_path, dpi=240, bbox_inches=crop)
    else:
        save_figure_both(fig, output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_synthetic(summary_dir: Path, figure_dir: Path) -> None:
    pairs = [
        ("02_alpha_beta_ratio", "SharedQ alpha/beta ratio", "line", "alpha_beta.pdf"),
        ("03_dataset_count", "Dataset count", "line", "dataset_count.pdf"),
        ("04_dimension", "Dimension", "line", "dimension.pdf"),
        ("05_dataset_size", "Dataset size", "line", "dataset_size.pdf"),
        ("06_data_type_ratio", "Data type ratio", "bar", "data_type.pdf"),
    ]
    for item in pairs:
        if len(item) == 4:
            prefix, title, style, output_name = item
        else:
            prefix, title, style = item
            output_name = f"{prefix}.png"
        paired_plot(
            summary_dir / f"{prefix}_time.csv",
            summary_dir / f"{prefix}_questions.csv",
            figure_dir / output_name,
            title,
            style,
            show_legend=prefix != "02_alpha_beta_ratio",
        )


def plot_real(real_summary_dir: Path, figure_dir: Path) -> None:
    for time_path in sorted(real_summary_dir.glob("*_time.csv")):
        family = time_path.name[: -len("_time.csv")]
        question_path = real_summary_dir / f"{family}_questions.csv"
        paired_plot(
            time_path,
            question_path,
            figure_dir / f"{family}.pdf",
            f"{family} by dimension",
            "line",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot summary CSV files as PDF figures.")
    parser.add_argument("--summary-dir", default=str(DEFAULT_SUMMARY_DIR))
    parser.add_argument("--real-summary-dir", default=str(DEFAULT_REAL_SUMMARY_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir).resolve()
    real_summary_dir = Path(args.real_summary_dir).resolve()
    figure_dir = Path(args.figure_dir).resolve()
    plot_synthetic(summary_dir, figure_dir)
    plot_real(real_summary_dir, figure_dir)
    print("done", flush=True)


if __name__ == "__main__":
    main()
