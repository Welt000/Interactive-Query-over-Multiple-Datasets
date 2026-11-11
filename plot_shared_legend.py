"""Export Figures 11--13 for the Overleaf shared-legend layout."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_experiment_summaries as p


def main():
    out = p.DEFAULT_FIGURE_DIR
    for prefix, name, style in [("04_dimension", "dimension_no_legend", "line"),
                               ("05_dataset_size", "dataset_size_no_legend", "line"),
                               ("06_data_type_ratio", "data_type_3_no_legend", "bar")]:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
        plot = p.plot_bars if style == "bar" else p.plot_lines
        plot(axes[0], p.DEFAULT_SUMMARY_DIR / f"{prefix}_time.csv", "Time (seconds)")
        plot(axes[1], p.DEFAULT_SUMMARY_DIR / f"{prefix}_questions.csv", "# of questions")
        # Fixed full canvas and identical axes rectangles keep all six panels aligned.
        fig.subplots_adjust(top=0.94, bottom=0.32, left=0.14, right=0.99, wspace=0.55)
        fig.savefig(out / f"{name}.pdf", bbox_inches=None)
        plt.close(fig)

    # Obtain exactly the same line styles and method order as the plotted data.
    source, ax = plt.subplots()
    p.plot_lines(ax, p.DEFAULT_SUMMARY_DIR / "04_dimension_time.csv", "")
    handles, labels = ax.get_legend_handles_labels()
    legend_fig = plt.figure(figsize=(16, 2))
    legend = legend_fig.legend(handles, labels, loc="center", ncol=6,
                               fontsize=20, frameon=False, handlelength=1.8,
                               markerscale=0.5, columnspacing=1.2, labelspacing=0.5)
    legend_fig.canvas.draw()
    bounds = legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted())
    legend_fig.savefig(out / "shared_legend.pdf", bbox_inches=bounds.expanded(1.02, 1.05))
    plt.close(source)
    plt.close(legend_fig)

    source, ax = plt.subplots()
    p.plot_bars(ax, p.DEFAULT_SUMMARY_DIR / "06_data_type_ratio_time.csv", "")
    handles, labels = ax.get_legend_handles_labels()
    legend_fig = plt.figure(figsize=(16, 2))
    legend = legend_fig.legend(handles, labels, loc="center", ncol=3,
                               fontsize=20, frameon=False, handlelength=1.8,
                               columnspacing=1.2, labelspacing=0.5)
    legend_fig.canvas.draw()
    bounds = legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted())
    legend_fig.savefig(out / "bar_legend.pdf", bbox_inches=bounds.expanded(1.02, 1.05))
    plt.close(source)
    plt.close(legend_fig)
    print(f"Shared legend files saved in {out}")


if __name__ == "__main__":
    main()
