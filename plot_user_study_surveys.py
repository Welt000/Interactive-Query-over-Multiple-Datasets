from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SURVEY_PATH = BASE_DIR / "user_study" / "results" / "surveys.csv"
DEFAULT_FIGURE_DIR = BASE_DIR / "results" / "figures"

METHOD_ORDER = ["SharedQ", "SUR-HDPI", "SUR-RH", "SUR-UH-Random", "SUR-UH-Simplex"]

METHOD_LABELS = {
    "SharedQ": "SharedQ",
    "SUR-HDPI": "SUR-HDPI",
    "SUR-RH": "SUR-RH",
    "SUR-UH-Random": "SUR-UH-Random",
    "SUR-UH-Simplex": "SUR-UH-Simplex",
}

COLORS = {
    "SharedQ": "#1f77b4",
    "SUR-HDPI": "#ff7f0e",
    "SUR-RH": "#2ca02c",
    "SUR-UH-Random": "#9467bd",
    "SUR-UH-Simplex": "#d62728",
}

HATCHES = {
    "SharedQ": "////",
    "SUR-HDPI": "xxxx",
    "SUR-RH": "----",
    "SUR-UH-Random": "....",
    "SUR-UH-Simplex": "\\\\\\",
}

METHOD_ALIASES = {
    "SharedQ": "SharedQ",
    "sharedq": "SharedQ",
    "sharedq": "SharedQ",
}

METRICS = ["questions", "annoyance", "satisfaction"]
MEASUREMENT_LABELS = ["# of Questions", "Annoyance Score", "Satisfaction Score"]


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


def read_surveys(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"survey file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise ValueError(f"survey file is empty: {path}")
    if "method" not in fieldnames:
        raise ValueError(
            "surveys.csv does not contain a method column; cannot draw per-method bars."
        )
    return rows


def summarize_by_method(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        method = str(row.get("method", "")).strip()
        method = METHOD_ALIASES.get(method, method)
        if method:
            grouped[method].append(row)

    summary: dict[str, dict[str, float]] = {}
    for method, group in grouped.items():
        metric_values: dict[str, float] = {}
        for metric in METRICS:
            values = [
                value
                for value in (parse_float(row.get(metric)) for row in group)
                if value is not None
            ]
            if values:
                metric_values[metric] = mean(values)
        metric_values["n"] = float(len(group))
        summary[method] = metric_values
    return summary


def ordered_methods(summary: dict[str, dict[str, float]]) -> list[str]:
    known = [method for method in METHOD_ORDER if method in summary]
    extra = sorted(method for method in summary if method not in METHOD_ORDER)
    return known + extra


def plot_combined(summary: dict[str, dict[str, float]], methods: list[str], output_path: Path) -> None:
    x = np.arange(len(METRICS), dtype=float)
    width = min(0.78 / max(len(methods), 1), 0.14)

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 16,
            "xtick.labelsize": 15,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "hatch.linewidth": 1.0,
        }
    )

    fig, ax_questions = plt.subplots(figsize=(10.2, 5.2))
    ax_scores = ax_questions.twinx()

    question_values = [
        summary.get(method, {}).get("questions", 0.0)
        for method in methods
    ]
    score_values = [
        summary.get(method, {}).get(metric, 0.0)
        for method in methods
        for metric in ("annoyance", "satisfaction")
    ]

    for i, method in enumerate(methods):
        color = COLORS.get(method, "#333333")
        hatch = HATCHES.get(method, "////")
        offset = (i - (len(methods) - 1) / 2.0) * width
        for j, metric in enumerate(METRICS):
            value = summary.get(method, {}).get(metric, 0.0)
            axis = ax_questions if metric == "questions" else ax_scores
            axis.bar(
                x[j] + offset,
                value,
                width=width,
                facecolor="white",
                edgecolor=color,
                linewidth=1.0,
                hatch=hatch,
                zorder=3,
            )

    ax_questions.set_ylabel("# of Questions")
    ax_scores.set_ylabel("Survey Score")
    ax_questions.set_xlabel("Measurement")
    ax_questions.set_xticks(x)
    ax_questions.set_xticklabels(MEASUREMENT_LABELS)
    ax_questions.tick_params(direction="in", top=True)
    ax_scores.tick_params(direction="in", top=True)
    ax_questions.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3, zorder=0)

    question_max = max(question_values) if question_values else 1.0
    ax_questions.set_ylim(0, max(1.0, question_max * 1.28))
    ax_scores.set_ylim(0, 7.0)

    legend_handles = [
        Patch(
            facecolor="white",
            edgecolor=COLORS.get(method, "#333333"),
            hatch=HATCHES.get(method, "////"),
            linewidth=1.0,
            label=METHOD_LABELS.get(method, method),
        )
        for method in methods
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=math.ceil(len(methods) / 2),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.09, right=0.9)

    save_figure_both(fig, output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot user-study survey metrics.")
    parser.add_argument("--survey-csv", default=str(DEFAULT_SURVEY_PATH))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    parser.add_argument("--output-name", default="user_study.pdf")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    survey_path = Path(args.survey_csv).resolve()
    figure_dir = Path(args.figure_dir).resolve()
    rows = read_surveys(survey_path)
    summary = summarize_by_method(rows)
    methods = ordered_methods(summary)
    if not methods:
        raise ValueError("no method data found in surveys.csv")

    plot_combined(summary, methods, figure_dir / args.output_name)


if __name__ == "__main__":
    main()
