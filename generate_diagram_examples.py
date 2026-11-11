from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "overleaf_figures" / "diagram_examples"

BLUE = "#4C78A8"
ORANGE = "#F58518"
RED = "#E45756"
GREEN = "#54A24B"
GRAY = "#B8B8B8"
DARK = "#222222"
LIGHT_GRAY = "#E8E8E8"

TRIANGLE = np.array(
    [
        [0.06, 0.06],
        [0.94, 0.06],
        [0.50, 0.82],
    ]
)


def setup_ax() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.9)
    ax.axis("off")
    return fig, ax


def draw_simplex(ax: plt.Axes, labels: bool = True) -> None:
    ax.add_patch(
        Polygon(
            TRIANGLE,
            closed=True,
            facecolor="white",
            edgecolor=DARK,
            linewidth=1.8,
            joinstyle="round",
        )
    )
    if labels:
        ax.text(0.02, 0.015, r"$u_1$", fontsize=13)
        ax.text(0.94, 0.015, r"$u_2$", fontsize=13)
        ax.text(0.49, 0.845, r"$u_3$", fontsize=13)


def draw_region(ax: plt.Axes, vertices: np.ndarray, color: str = BLUE, label: str = r"$R$") -> None:
    ax.add_patch(
        Polygon(
            vertices,
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=2.2,
            alpha=0.20,
            joinstyle="round",
        )
    )
    center = vertices.mean(axis=0)
    ax.text(center[0] - 0.015, center[1], label, color=color, fontsize=16, weight="bold")


def add_partition(
    ax: plt.Axes,
    vertices: list[tuple[float, float]],
    edge: str = GRAY,
    face: str = "white",
    alpha: float = 1.0,
    linewidth: float = 1.0,
    linestyle: str = "-",
) -> None:
    ax.add_patch(
        Polygon(
            np.asarray(vertices),
            closed=True,
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            joinstyle="round",
        )
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = OUT_DIR / f"{name}{suffix}"
        fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=0.03)
        print(f"wrote {path}", flush=True)
    plt.close(fig)


def utility_space() -> None:
    fig, ax = setup_ax()
    draw_simplex(ax)
    region = np.array([[0.27, 0.20], [0.64, 0.18], [0.72, 0.36], [0.48, 0.57], [0.27, 0.41]])
    draw_region(ax, region)
    center = np.array([0.48, 0.34])
    true_u = np.array([0.61, 0.45])
    ax.add_patch(Circle(center, 0.19, facecolor=ORANGE, edgecolor=ORANGE, linewidth=2.0, alpha=0.16))
    ax.plot(center[0], center[1], "o", color=BLUE, markersize=8)
    ax.text(center[0] - 0.06, center[1] - 0.055, r"$c$", color=BLUE, fontsize=15)
    ax.plot(true_u[0], true_u[1], "*", color=RED, markersize=14)
    ax.text(true_u[0] + 0.025, true_u[1] + 0.015, r"$u^*$", color=RED, fontsize=15)
    ax.annotate("", xy=true_u, xytext=center, arrowprops=dict(arrowstyle="->", color=RED, lw=2.2))
    ax.text(0.64, 0.23, r"$B(c,r)$", color=ORANGE, fontsize=14)
    save(fig, "utility_space_example")


def partition_range() -> None:
    fig, ax = setup_ax()
    draw_simplex(ax, labels=False)
    partitions = [
        [(0.16, 0.09), (0.33, 0.09), (0.31, 0.27), (0.20, 0.28)],
        [(0.33, 0.09), (0.52, 0.10), (0.50, 0.27), (0.31, 0.27)],
        [(0.52, 0.10), (0.75, 0.10), (0.67, 0.28), (0.50, 0.27)],
        [(0.20, 0.28), (0.31, 0.27), (0.37, 0.45), (0.25, 0.48)],
        [(0.31, 0.27), (0.50, 0.27), (0.53, 0.45), (0.37, 0.45)],
        [(0.50, 0.27), (0.67, 0.28), (0.62, 0.45), (0.53, 0.45)],
        [(0.25, 0.48), (0.37, 0.45), (0.50, 0.66), (0.38, 0.61)],
        [(0.37, 0.45), (0.53, 0.45), (0.50, 0.66)],
        [(0.53, 0.45), (0.62, 0.45), (0.50, 0.66)],
    ]
    kept = {4, 5, 7}
    for idx, poly in enumerate(partitions):
        if idx in kept:
            add_partition(ax, poly, edge=BLUE, face=BLUE, alpha=0.10, linewidth=2.0)
        else:
            add_partition(ax, poly, edge=GRAY, face="white", alpha=0.95, linewidth=1.0, linestyle="--")
    region = np.array([[0.38, 0.29], [0.62, 0.30], [0.58, 0.49], [0.43, 0.54], [0.34, 0.43]])
    draw_region(ax, region, BLUE, r"$R$")
    ax.text(0.57, 0.55, "kept", color=BLUE, fontsize=13)
    ax.text(0.16, 0.56, "pruned", color=GRAY, fontsize=13)
    save(fig, "partition_range_example")


def detect() -> None:
    fig, ax = setup_ax()
    draw_simplex(ax, labels=False)
    region = np.array([[0.26, 0.18], [0.70, 0.19], [0.75, 0.38], [0.53, 0.62], [0.30, 0.44]])
    draw_region(ax, region, BLUE, r"$R_t$")
    center = np.array([0.48, 0.36])
    ax.add_patch(Circle(center, 0.22, facecolor=ORANGE, edgecolor=ORANGE, linewidth=2.2, alpha=0.16))
    ax.plot(center[0], center[1], "o", color=BLUE, markersize=8)
    ax.text(center[0] - 0.045, center[1] - 0.055, r"$c_t$", color=BLUE, fontsize=14)
    boundary = [
        [(0.58, 0.51), (0.70, 0.46), (0.67, 0.31), (0.55, 0.34)],
        [(0.28, 0.37), (0.38, 0.53), (0.47, 0.44), (0.39, 0.30)],
    ]
    inner = [
        [(0.39, 0.30), (0.55, 0.34), (0.47, 0.44), (0.38, 0.53)],
        [(0.36, 0.20), (0.53, 0.20), (0.55, 0.34), (0.39, 0.30)],
    ]
    for poly in inner:
        add_partition(ax, poly, edge=BLUE, face=BLUE, alpha=0.10, linewidth=1.6)
    for poly in boundary:
        add_partition(ax, poly, edge=RED, face=RED, alpha=0.16, linewidth=2.2)
    ax.text(0.62, 0.57, "boundary", color=RED, fontsize=13)
    ax.text(0.56, 0.19, "detect ball", color=ORANGE, fontsize=13)
    save(fig, "detect_example")


def divide_continue() -> None:
    fig, ax = setup_ax()
    draw_simplex(ax, labels=False)
    region = np.array([[0.24, 0.17], [0.73, 0.18], [0.70, 0.43], [0.49, 0.65], [0.27, 0.43]])
    draw_region(ax, region, BLUE, r"$R_t$")
    ax.plot([0.27, 0.76], [0.52, 0.24], color=DARK, lw=2.3)
    ax.text(0.73, 0.23, r"$h$", color=DARK, fontsize=15)
    ax.annotate("", xy=(0.62, 0.44), xytext=(0.50, 0.37), arrowprops=dict(arrowstyle="->", lw=2.0, color=RED))
    add_partition(ax, [(0.34, 0.25), (0.51, 0.27), (0.55, 0.43), (0.39, 0.47)], edge=GREEN, face=GREEN, alpha=0.18, linewidth=2.3)
    add_partition(ax, [(0.55, 0.43), (0.70, 0.38), (0.66, 0.24), (0.51, 0.27)], edge=RED, face=RED, alpha=0.12, linewidth=2.0)
    ax.text(0.28, 0.61, "continue divide", color=GREEN, fontsize=14, weight="bold")
    ax.text(0.32, 0.22, "interior candidate", color=GREEN, fontsize=12)
    save(fig, "divide_continue_example")


def divide_stop() -> None:
    fig, ax = setup_ax()
    draw_simplex(ax, labels=False)
    region = np.array([[0.24, 0.17], [0.73, 0.18], [0.70, 0.43], [0.49, 0.65], [0.27, 0.43]])
    draw_region(ax, region, BLUE, r"$R_t$")
    for poly in [
        [(0.62, 0.36), (0.72, 0.31), (0.70, 0.18), (0.58, 0.20)],
        [(0.28, 0.42), (0.37, 0.55), (0.49, 0.65), (0.39, 0.44)],
        [(0.49, 0.65), (0.64, 0.50), (0.70, 0.43), (0.55, 0.45)],
    ]:
        add_partition(ax, poly, edge=RED, face=RED, alpha=0.16, linewidth=2.2)
    anchor = np.array([0.49, 0.40])
    new_center = np.array([0.34, 0.22])
    ax.plot(anchor[0], anchor[1], "o", color=BLUE, markersize=7)
    ax.annotate("", xy=new_center, xytext=anchor, arrowprops=dict(arrowstyle="->", lw=2.3, color=RED))
    ax.plot(new_center[0], new_center[1], "o", color=RED, markersize=8)
    ax.text(new_center[0] - 0.08, new_center[1] - 0.055, r"$c_{t+1}$", color=RED, fontsize=14)
    ax.text(0.31, 0.61, "stop divide", color=RED, fontsize=14, weight="bold")
    ax.text(0.35, 0.14, "next detect", color=RED, fontsize=13)
    save(fig, "divide_stop_example")


def main() -> None:
    utility_space()
    partition_range()
    detect()
    divide_continue()
    divide_stop()


if __name__ == "__main__":
    main()
