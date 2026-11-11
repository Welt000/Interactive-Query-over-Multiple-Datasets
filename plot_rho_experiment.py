from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_DIR = BASE_DIR / "results" / "4d_100k_10_result"
DEFAULT_FIGURE_DIR = BASE_DIR / "results" / "figures"

QUESTION_COL = "avg_total_questions_per_utility"
TIME_COL = "avg_total_elapsed_per_utility"

METHOD_LABELS = {
    "sharedq": "SharedQ",
    "baseline_HD-PI": "HD-PI",
    "baseline_RH": "RH",
    "shared_utility_range_HD-PI": "SUR-HD-PI",
    "shared_utility_range_RH": "SUR-RH",
}

COLORS = {
    "sharedq": "#111111",
    "baseline_HD-PI": "#1f77b4",
    "baseline_RH": "#ff7f0e",
    "shared_utility_range_HD-PI": "#9467bd",
    "shared_utility_range_RH": "#8c564b",
}

MARKERS = {
    "sharedq": "P",
    "baseline_HD-PI": "+",
    "baseline_RH": "x",
    "shared_utility_range_HD-PI": "D",
    "shared_utility_range_RH": "v",
}
LINESTYLES = {
    "sharedq": (0, (3, 1, 1, 1)),
    "baseline_HD-PI": "-",
    "baseline_RH": "-",
    "shared_utility_range_HD-PI": "--",
    "shared_utility_range_RH": "--",
}

METHOD_ALIASES = {
    "sharedq": "sharedq",
    "SharedQ": "sharedq",
}


def save_figure_both(fig: plt.Figure, output_path: Path, **kwargs: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, **kwargs)
    print(f"wrote {pdf_path}", flush=True)


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


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"rho summary not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if "rho" not in fieldnames:
        raise ValueError(f"{path} does not contain a rho column")
    if not rows:
        raise ValueError(f"{path} is empty")
    result: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("status", "ok")).strip().lower() not in {"", "ok"}:
            continue
        if parse_float(row.get("rho")) is None:
            continue
        normalized = dict(row)
        normalized["method"] = METHOD_ALIASES.get(str(row.get("method", "")).strip(), str(row.get("method", "")).strip())
        result.append(normalized)
    return result


def group_by_method(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        method = str(row.get("method", "")).strip()
        if method:
            grouped[method].append(row)
    for method in grouped:
        grouped[method].sort(key=lambda row: parse_float(row.get("rho")) or 0.0)
    return dict(grouped)


def plot_metric(
    ax: plt.Axes,
    grouped: dict[str, list[dict[str, str]]],
    metric_col: str,
    ylabel: str,
) -> None:
    for idx, (method, rows) in enumerate(grouped.items()):
        x: list[float] = []
        y: list[float] = []
        for row in rows:
            rho = parse_float(row.get("rho"))
            value = parse_float(row.get(metric_col))
            if rho is None or value is None:
                continue
            x.append(rho)
            y.append(value)
        if not x:
            continue
        ax.plot(
            np.asarray(x),
            np.asarray(y),
            marker=MARKERS.get(method, "o"),
            markersize=28,
            linewidth=4.0 if method == "sharedq" else 3.4,
            color=COLORS.get(method, "#333333"),
            markerfacecolor="none",
            markeredgewidth=1.8,
            linestyle=LINESTYLES.get(method, "-"),
            label=METHOD_LABELS.get(method, method),
        )
    ax.set_xlabel("Selection rate \u03c1")
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)


def set_rho_ticks(axes: np.ndarray, rows: list[dict[str, str]]) -> None:
    rho_values = [value for value in (parse_float(row.get("rho")) for row in rows) if value is not None]
    if not rho_values:
        return
    start = math.floor((min(rho_values) + 1e-12) * 10.0) / 10.0
    end = math.ceil((max(rho_values) - 1e-12) * 10.0) / 10.0
    ticks = np.round(np.arange(start, end + 0.05, 0.1), 10)
    labels = [f"{tick:.1f}" for tick in ticks]
    for ax in axes:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)


def plot_rho_summary(input_path: Path, output_path: Path) -> None:
    rows = read_rows(input_path)
    grouped = group_by_method(rows)
    if not grouped:
        raise ValueError("no ok rows with method and rho found")

    plt.rcParams.update(
        {
            "font.size": 42,
            "axes.labelsize": 46,
            "xtick.labelsize": 38,
            "ytick.labelsize": 38,
            "legend.fontsize": 30,
        }
    )
    # Tuned with bbox_inches="tight" so the saved rho figure matches
    # the alpha/beta figure's final canvas size.
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    plot_metric(axes[0], grouped, TIME_COL, "Time (seconds)")
    plot_metric(axes[1], grouped, QUESTION_COL, "# of Questions")
    set_rho_ticks(axes, rows)

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=min(len(labels), 4),
            frameon=False,
            handlelength=2.8,
            markerscale=0.5,
            columnspacing=1.6,
        )
    fig.subplots_adjust(top=0.80, bottom=0.20, left=0.12, right=0.98, wspace=0.50)

    save_figure_both(fig, output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SharedQ results under different selection rates rho.")
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--input-csv", default="")
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--output-name", default="rho.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir).resolve()
    input_path = Path(args.input_csv).resolve() if args.input_csv else result_dir / "experiment_rho_summary.csv"
    output_path = Path(args.figure_dir).resolve() / args.output_name
    plot_rho_summary(input_path, output_path)


if __name__ == "__main__":
    main()
