"""Read ablation result files, verify means against individual runs, and plot."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext


OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
DEFAULT_RESULTS = OUT.parent / "ablation_sharedq_10u_20261024"
GROUPS = [
    ("neither", "None"),
    ("pca_only", "PCA only"),
    ("detect_divide_only", "Detect-and-divide only"),
    ("full", "Full"),
]


def read_results(results_dir):
    source = results_dir / "verified_summary.csv"
    with source.open(newline="", encoding="utf-8-sig") as f:
        summaries = {row["group"]: row for row in csv.DictReader(f)}
    rows = []
    for group, label in GROUPS:
        summary = summaries[group]
        runs = [json.loads(path.read_text(encoding="utf-8-sig"))
                for path in sorted((results_dir / group).glob("u[0-9]*.json"))]
        successful = [run for run in runs if run["status"] == "ok"]
        if len(successful) != int(summary["successful"]) or not successful:
            raise ValueError(f"{group}: individual run count does not match summary")
        questions = mean(run["questions"] for run in successful)
        seconds = mean(run["elapsed"] for run in successful)
        if not (math.isclose(questions, float(summary["avg_questions"]), rel_tol=1e-9)
                and math.isclose(seconds, float(summary["avg_seconds"]), rel_tol=1e-9)):
            raise ValueError(f"{group}: individual run means do not match summary")
        rows.append((label, questions, seconds))
        print(f"{group}: n={len(successful)}, questions={questions:.1f}, seconds={seconds:.9f}")
    print(f"Source: {source}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    rows = read_results(args.results_dir.resolve())
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "ablation_four_configs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", "mean_questions", "mean_algorithm_seconds"])
        writer.writerows(rows)

    plt.rcParams.update({"font.size": 21, "axes.labelsize": 24,
                         "xtick.labelsize": 21, "ytick.labelsize": 21,
                         "legend.fontsize": 21,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), layout="constrained")
    labels = ["None", "PCA\nonly", "Detect-and-\ndivide only", "Full"]
    colors = ["#8c8c8c", "#4c78a8", "#f2a541", "#3a9278"]
    hatches = ["", "///", "\\\\", "xx"]
    for ax, col, ylabel in zip(axes, [2, 1],
                              ["Time(seconds)", "# of questions"]):
        values = [row[col] for row in rows]
        bars = ax.bar(range(4), values, width=0.62, color=colors,
                      edgecolor="#333333", linewidth=0.8, zorder=3)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_xticks(range(4), labels)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=9)
        if col == 2:
            ax.set_yscale("log")
            ax.set_ylim(1, 4000)
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        else:
            ax.set_ylim(0, 25)
    for extension in ["png", "pdf"]:
        path = OUT / f"ablation_four_configs.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
