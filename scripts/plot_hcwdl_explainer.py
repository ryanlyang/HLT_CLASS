"""Create compact Matplotlib diagrams for the HCWDL method.

The figures are intentionally schematic. They explain the scientific
semantics without depending on campaign artifacts or exposing final-test data.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


BG = "#FCFBF7"
INK = "#263238"
HLT = "#2878C8"
OFFLINE = "#F28E5B"
PURPLE = "#8267C7"
GREEN = "#35A56F"
AMBER = "#D7A62A"
GRAY = "#8A9499"
RED = "#D95C5C"
PALE_BLUE = "#EAF3FB"
PALE_ORANGE = "#FDEFE7"


def _blend(left: str, right: str, weight: float) -> tuple[float, float, float]:
    a = to_rgb(left)
    b = to_rgb(right)
    return tuple((1.0 - weight) * x + weight * y for x, y in zip(a, b))


def _canvas(title: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(
        0.55,
        8.42,
        title,
        color=INK,
        fontsize=27,
        fontweight="bold",
        ha="left",
        va="center",
    )
    return fig, ax


def _pill(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    width: float,
) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - 0.27),
        width,
        0.54,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        facecolor=color,
        edgecolor="none",
        alpha=0.14,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, color=color, fontsize=13, fontweight="bold", ha="center", va="center")


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor=BG)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def matching_figure(output_dir: Path) -> None:
    fig, ax = _canvas("1. Constituent matching")

    hlt_points = [
        (2.1, 6.7),
        (3.6, 6.0),
        (2.6, 4.8),
        (4.0, 4.1),
        (2.0, 3.2),
        (3.6, 2.4),
    ]
    offline_points = [
        (12.5, 6.8),
        (14.0, 6.0),
        (12.8, 4.9),
        (14.1, 4.0),
        (12.4, 3.1),
        (14.0, 2.3),
    ]
    sizes = [0.31, 0.24, 0.28, 0.21, 0.26, 0.20]

    ax.text(2.9, 7.55, "HLT", color=HLT, fontsize=20, fontweight="bold", ha="center")
    ax.text(13.25, 7.55, "Offline", color=OFFLINE, fontsize=20, fontweight="bold", ha="center")

    candidates = [
        (0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2),
        (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (4, 5),
        (5, 4), (5, 5),
    ]
    for i, j in candidates:
        ax.plot(
            [hlt_points[i][0], offline_points[j][0]],
            [hlt_points[i][1], offline_points[j][1]],
            color=GRAY,
            alpha=0.12,
            linewidth=1.1,
            zorder=1,
        )

    chosen = [
        (0, 0, GREEN, "-", 3.2),
        (1, 1, GREEN, "-", 3.2),
        (2, 2, GREEN, "-", 3.2),
        (3, 3, AMBER, (0, (4, 3)), 2.8),
        (4, 4, AMBER, (0, (4, 3)), 2.8),
    ]
    for i, j, color, style, width in chosen:
        ax.plot(
            [hlt_points[i][0], offline_points[j][0]],
            [hlt_points[i][1], offline_points[j][1]],
            color=color,
            linestyle=style,
            linewidth=width,
            alpha=0.95,
            zorder=2,
        )

    for (x, y), radius in zip(hlt_points, sizes):
        ax.add_patch(Circle((x, y), radius, facecolor=HLT, edgecolor="white", linewidth=2.2, zorder=4))
    for (x, y), radius in zip(offline_points, sizes):
        ax.add_patch(Circle((x, y), radius, facecolor=OFFLINE, edgecolor="white", linewidth=2.2, zorder=4))

    for x, y in (hlt_points[-1], offline_points[-1]):
        ax.add_patch(Circle((x, y), 0.31, facecolor="none", edgecolor=GRAY, linewidth=2.2, zorder=5))
        ax.plot([x - 0.15, x + 0.15], [y - 0.15, y + 0.15], color=GRAY, linewidth=2.0, zorder=6)
        ax.plot([x - 0.15, x + 0.15], [y + 0.15, y - 0.15], color=GRAY, linewidth=2.0, zorder=6)

    center = FancyBboxPatch(
        (5.55, 3.25),
        4.9,
        2.45,
        boxstyle="round,pad=0.12,rounding_size=0.24",
        facecolor="white",
        edgecolor="#D8DDDF",
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(center)
    ax.text(8.0, 5.10, "pair + context + consensus", color=INK, fontsize=16, fontweight="bold", ha="center")
    ax.text(8.0, 4.45, "global 1:1 assignment", color=INK, fontsize=18, fontweight="bold", ha="center")
    ax.text(8.0, 3.82, r"confidence  $q \in [0,1]$", color=PURPLE, fontsize=19, fontweight="bold", ha="center")

    _pill(ax, 6.35, 2.22, "high q", GREEN, 1.55)
    _pill(ax, 8.0, 2.22, "low q", AMBER, 1.55)
    _pill(ax, 9.85, 2.22, "dustbin", GRAY, 1.75)
    ax.text(8.0, 1.25, "not nearest-neighbor matching", color=GRAY, fontsize=14, ha="center")

    _save(fig, output_dir, "hcwdl_step1_matching")


def balanced_matching_figure(output_dir: Path) -> None:
    """Show the selected constituent assignment without confidence tiers."""
    fig, ax = _canvas("1. Constituent matching")

    hlt_points = [
        (2.1, 6.7),
        (3.6, 6.0),
        (2.6, 4.8),
        (4.0, 4.1),
        (2.0, 3.2),
        (3.6, 2.4),
    ]
    offline_points = [
        (12.5, 6.8),
        (14.0, 6.0),
        (12.8, 4.9),
        (14.1, 4.0),
        (12.4, 3.1),
        (14.0, 2.3),
    ]
    sizes = [0.31, 0.24, 0.28, 0.21, 0.26, 0.20]

    ax.text(2.9, 7.55, "HLT", color=HLT, fontsize=20, fontweight="bold", ha="center")
    ax.text(13.25, 7.55, "Offline", color=OFFLINE, fontsize=20, fontweight="bold", ha="center")

    candidates = [
        (0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2),
        (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (4, 5),
        (5, 4), (5, 5),
    ]
    for i, j in candidates:
        ax.plot(
            [hlt_points[i][0], offline_points[j][0]],
            [hlt_points[i][1], offline_points[j][1]],
            color=GRAY,
            alpha=0.12,
            linewidth=1.1,
            zorder=1,
        )

    # Every selected match is drawn identically. There is no high/low
    # confidence visual tier and confidence does not control the assignment.
    for i, j in ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4)):
        ax.plot(
            [hlt_points[i][0], offline_points[j][0]],
            [hlt_points[i][1], offline_points[j][1]],
            color=GREEN,
            linestyle="-",
            linewidth=3.2,
            alpha=0.95,
            zorder=2,
        )

    for (x, y), radius in zip(hlt_points, sizes):
        ax.add_patch(Circle((x, y), radius, facecolor=HLT, edgecolor="white", linewidth=2.2, zorder=4))
    for (x, y), radius in zip(offline_points, sizes):
        ax.add_patch(Circle((x, y), radius, facecolor=OFFLINE, edgecolor="white", linewidth=2.2, zorder=4))

    for x, y in (hlt_points[-1], offline_points[-1]):
        ax.add_patch(Circle((x, y), 0.31, facecolor="none", edgecolor=GRAY, linewidth=2.2, zorder=5))
        ax.plot([x - 0.15, x + 0.15], [y - 0.15, y + 0.15], color=GRAY, linewidth=2.0, zorder=6)
        ax.plot([x - 0.15, x + 0.15], [y + 0.15, y - 0.15], color=GRAY, linewidth=2.0, zorder=6)

    center = FancyBboxPatch(
        (5.35, 3.15),
        5.3,
        2.75,
        boxstyle="round,pad=0.12,rounding_size=0.24",
        facecolor="white",
        edgecolor="#D8DDDF",
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(center)
    ax.text(8.0, 5.28, r"first: nearby candidates in  $\Delta R$", color=INK, fontsize=16, fontweight="bold", ha="center")
    ax.text(8.0, 4.72, r"then: $p_T$ and energy response", color=PURPLE, fontsize=15.5, fontweight="bold", ha="center")
    ax.text(8.0, 4.20, r"+ PID / charge + $p_T$-rank consistency", color=PURPLE, fontsize=14.5, fontweight="bold", ha="center")
    ax.text(8.0, 3.60, "global 1:1 assignment", color=INK, fontsize=18, fontweight="bold", ha="center")

    _pill(ax, 6.15, 2.22, r"$\Delta R$ gate", GREEN, 1.65)
    _pill(ax, 8.35, 2.22, "response + identity", PURPLE, 2.35)
    _pill(ax, 10.55, 2.22, "dustbin", GRAY, 1.65)
    ax.text(8.0, 1.25, "global assignment — not greedy nearest-neighbor matching", color=GRAY, fontsize=14, ha="center")

    _save(fig, output_dir, "hcwdl_step1_balanced_matching")


def progression_figure(output_dir: Path) -> None:
    fig, ax = _canvas("2. Uniform progression")

    stages = [
        ("D100", 1.00),
        ("D75", 0.75),
        ("D50", 0.50),
        ("D25", 0.25),
        ("D0", 0.00),
    ]
    xs = [1.8, 4.9, 8.0, 11.1, 14.2]
    points = [
        (-0.55, 0.55),
        (0.12, 0.68),
        (0.58, 0.32),
        (-0.34, -0.02),
        (0.28, -0.18),
        (-0.63, -0.53),
        (0.62, -0.52),
        (0.02, -0.70),
    ]
    radii = [0.20, 0.16, 0.18, 0.14, 0.17, 0.13, 0.12, 0.13]

    for index, ((name, alpha), x) in enumerate(zip(stages, xs)):
        ax.add_patch(Circle((x, 4.75), 1.22, facecolor="white", edgecolor="#D8DDDF", linewidth=2.0))
        for point_index, ((dx, dy), radius) in enumerate(zip(points, radii)):
            if point_index == len(points) - 1:
                # A dustbin is an exact HLT token at every rung. The larger
                # double outline makes that invariant visually unmistakable.
                color = HLT
                edge = INK
                linewidth = 2.2
                ax.add_patch(
                    Circle(
                        (x + dx, 4.75 + dy),
                        radius + 0.055,
                        facecolor="none",
                        edgecolor=GRAY,
                        linewidth=2.5,
                        zorder=2,
                    )
                )
            else:
                color = _blend(HLT, OFFLINE, alpha)
                edge = "white"
                linewidth = 1.3
            ax.add_patch(
                Circle(
                    (x + dx, 4.75 + dy),
                    radius,
                    facecolor=color,
                    edgecolor=edge,
                    linewidth=linewidth,
                    zorder=3,
                )
            )
        ax.text(x, 6.38, name, color=INK, fontsize=19, fontweight="bold", ha="center")
        ax.text(x, 3.10, rf"$\alpha={alpha:g}$", color=GRAY, fontsize=14, ha="center")
        if index < len(xs) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 1.28, 4.75),
                    (xs[index + 1] - 1.28, 4.75),
                    arrowstyle="-|>",
                    mutation_scale=16,
                    linewidth=2.1,
                    color=PURPLE,
                )
            )

    ax.text(8.0, 7.33, "fixed HLT skeleton", color=INK, fontsize=17, fontweight="bold", ha="center")
    ax.text(8.0, 6.92, "same count  •  same order  •  one shared alpha", color=GRAY, fontsize=13.5, ha="center")

    dustbin_box = FancyBboxPatch(
        (4.60, 1.36),
        6.8,
        0.72,
        boxstyle="round,pad=0.06,rounding_size=0.20",
        facecolor=PALE_BLUE,
        edgecolor=HLT,
        linewidth=1.7,
    )
    ax.add_patch(dustbin_box)
    ax.add_patch(Circle((5.15, 1.72), 0.16, facecolor=HLT, edgecolor=INK, linewidth=2.0, zorder=3))
    ax.add_patch(Circle((5.15, 1.72), 0.22, facecolor="none", edgecolor=GRAY, linewidth=2.2, zorder=2))
    ax.text(
        5.55,
        1.72,
        "DUSTBIN stays exact HLT at every rung",
        color=INK,
        fontsize=13.5,
        fontweight="bold",
        ha="left",
        va="center",
    )

    _pill(ax, 6.05, 0.72, "continuous: uniform blend", PURPLE, 3.40)
    _pill(ax, 9.95, 0.72, "discrete: fixed-hash switch", PURPLE, 3.70)

    _save(fig, output_dir, "hcwdl_step2_progression")


def _node(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    color: str,
    width: float = 1.25,
    edge: str | None = None,
) -> None:
    box = FancyBboxPatch(
        (x - width / 2, y - 0.38),
        width,
        0.76,
        boxstyle="round,pad=0.05,rounding_size=0.16",
        facecolor=color,
        edgecolor=edge or "white",
        linewidth=2.8 if edge else 1.5,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, color="white", fontsize=14, fontweight="bold", ha="center", va="center", zorder=4)


def distillation_figure(output_dir: Path) -> None:
    fig, ax = _canvas("3. Dense downward distillation")

    labels = ["TOFF", "D100", "D90", "D80", "D70", "...", "D20", "D10", "D0", "M1"]
    xs = [0.95, 2.55, 4.15, 5.75, 7.35, 8.7, 10.1, 11.7, 13.3, 14.9]
    y = 4.75

    for index, (label, x) in enumerate(zip(labels, xs)):
        if label == "TOFF":
            color = OFFLINE
        elif label == "M1":
            color = GREEN
        elif label == "...":
            color = GRAY
        else:
            fraction = max(0.0, min(1.0, (x - xs[1]) / (xs[-2] - xs[1])))
            color = _blend(PURPLE, HLT, fraction)
        _node(ax, x, y, label, color, width=1.18 if label != "..." else 0.82, edge=INK if label == "M1" else None)

        if index < len(xs) - 1:
            start = x + (0.63 if label != "..." else 0.45)
            end = xs[index + 1] - (0.63 if labels[index + 1] != "..." else 0.45)
            ax.add_patch(
                FancyArrowPatch(
                    (start, y),
                    (end, y),
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=2.1,
                    color=INK,
                    zorder=2,
                )
            )

    ax.add_patch(
        FancyArrowPatch(
            (1.15, 4.25),
            (14.72, 4.25),
            connectionstyle="arc3,rad=0.35",
            arrowstyle="-|>",
            mutation_scale=17,
            linewidth=2.0,
            linestyle=(0, (5, 4)),
            color=RED,
            alpha=0.75,
            zorder=1,
        )
    )
    ax.text(8.0, 2.42, "direct transfer: large gap", color=RED, fontsize=14, fontweight="bold", ha="center")

    _pill(ax, 4.0, 6.62, "small input steps", PURPLE, 2.55)
    _pill(ax, 8.0, 6.62, "CE + predecessor KD", PURPLE, 3.15)
    _pill(ax, 12.0, 6.62, "cold start", PURPLE, 2.15)

    ax.text(8.0, 7.30, "same ParT at every rung", color=INK, fontsize=17, fontweight="bold", ha="center")
    ax.text(14.9, 3.66, "HLT only", color=GREEN, fontsize=14, fontweight="bold", ha="center")

    result = FancyBboxPatch(
        (5.8, 0.62),
        4.4,
        0.92,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor=PALE_BLUE,
        edgecolor="none",
    )
    ax.add_patch(result)
    ax.text(8.0, 1.08, "R50  1216 → 1331   (+9.5%)", color=INK, fontsize=17, fontweight="bold", ha="center", va="center")

    _save(fig, output_dir, "hcwdl_step3_distillation")


def ensemble_steps_figure(output_dir: Path) -> None:
    """Explain D-ladder ensembling with one reusable offline teacher."""
    fig, ax = _canvas("4. Ensembling each D-ladder step")

    def student_box(
        x: float,
        y: float,
        rung: str,
        teacher: str,
        color: str,
    ) -> None:
        box = FancyBboxPatch(
            (x - 1.18, y - 0.52),
            2.36,
            1.04,
            boxstyle="round,pad=0.07,rounding_size=0.18",
            facecolor="white",
            edgecolor=color,
            linewidth=2.5,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(
            x, y + 0.17, f"{rung} student", color=INK,
            fontsize=14.5, fontweight="bold", ha="center", va="center", zorder=4,
        )
        ax.text(
            x, y - 0.20, f"KD from {teacher}", color=color,
            fontsize=12.5, fontweight="bold", ha="center", va="center", zorder=4,
        )

    def ensemble_box(x: float, y: float, label: str) -> None:
        box = FancyBboxPatch(
            (x - 0.86, y - 0.61),
            1.72,
            1.22,
            boxstyle="round,pad=0.07,rounding_size=0.22",
            facecolor=GREEN,
            edgecolor="white",
            linewidth=2.0,
            zorder=4,
        )
        ax.add_patch(box)
        ax.text(
            x, y + 0.24, "50 / 50", color="white",
            fontsize=11.5, fontweight="bold", ha="center", va="center", zorder=5,
        )
        ax.text(
            x, y - 0.12, label, color="white",
            fontsize=17, fontweight="bold", ha="center", va="center", zorder=5,
        )

    def arrow(
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        color: str = INK,
        style: str | tuple[int, tuple[int, ...]] = "-",
        width: float = 2.1,
    ) -> None:
        ax.add_patch(FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=width,
            linestyle=style,
            color=color,
            zorder=2,
        ))

    # One fixed offline teacher feeds one extra student at every target rung.
    _node(ax, 1.35, 6.85, "Offline teacher", OFFLINE, width=2.15)
    ax.text(
        2.65, 7.34, "reused at every rung", color=OFFLINE,
        fontsize=14, fontweight="bold", ha="left", va="center",
    )
    ax.plot(
        [2.45, 10.15], [6.85, 6.85], color=OFFLINE,
        linewidth=2.5, linestyle=(0, (5, 4)), alpha=0.90, zorder=1,
    )

    # Ordinary predecessor ladder along the lower path.
    _node(ax, 1.35, 3.22, "D60", PURPLE, width=1.45)
    student_box(4.10, 5.42, "D40", "Offline", OFFLINE)
    student_box(4.10, 3.22, "D40", "D60", PURPLE)
    ensemble_box(7.10, 4.32, "D40E")

    student_box(10.05, 5.42, "D20", "Offline", OFFLINE)
    student_box(10.05, 3.22, "D20", "D40E", GREEN)
    ensemble_box(13.05, 4.32, "D20E")

    # Anchor-bus drops.
    arrow((4.10, 6.85), (4.10, 5.98), color=OFFLINE, style=(0, (5, 4)))
    arrow((10.05, 6.85), (10.05, 5.98), color=OFFLINE, style=(0, (5, 4)))

    # Local predecessor KD.
    arrow((2.10, 3.22), (2.88, 3.22), color=PURPLE)
    arrow((7.98, 4.02), (8.82, 3.45), color=GREEN)

    # Each pair of independently trained students becomes one probability
    # ensemble. The ensemble is then the local teacher at the next rung.
    arrow((5.30, 5.23), (6.20, 4.61), color=GREEN, width=2.5)
    arrow((5.30, 3.42), (6.20, 4.03), color=GREEN, width=2.5)
    arrow((10.55 + 0.70, 5.23), (12.15, 4.61), color=GREEN, width=2.5)
    arrow((10.55 + 0.70, 3.42), (12.15, 4.03), color=GREEN, width=2.5)
    arrow((13.93, 4.32), (15.05, 4.32), color=GREEN, width=2.5)
    ax.text(14.48, 4.72, "to D0", color=GREEN, fontsize=13, fontweight="bold", ha="center")

    ax.text(
        7.10, 5.13, "average predicted\nprobabilities", color=GREEN,
        fontsize=12.5, fontweight="bold", ha="center", va="bottom",
    )
    ax.text(
        13.05, 5.13, "average predicted\nprobabilities", color=GREEN,
        fontsize=12.5, fontweight="bold", ha="center", va="bottom",
    )

    explanation = FancyBboxPatch(
        (1.15, 0.62),
        13.70,
        1.25,
        boxstyle="round,pad=0.08,rounding_size=0.20",
        facecolor=PALE_BLUE,
        edgecolor="none",
        zorder=1,
    )
    ax.add_patch(explanation)
    ax.text(
        8.0, 1.43,
        "At each D rung: same target inputs, separate cold-start students; only the KD teacher changes.",
        color=INK, fontsize=13.3, fontweight="bold", ha="center", va="center",
    )
    ax.text(
        8.0, 0.96,
        "E = the 50/50 probability ensemble passed forward as the next local teacher.",
        color=GRAY, fontsize=13.5, ha="center", va="center",
    )

    _save(fig, output_dir, "hcwdl_step4_ensembling")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "figures",
    )
    args = parser.parse_args()
    matching_figure(args.output_dir)
    balanced_matching_figure(args.output_dir)
    progression_figure(args.output_dir)
    distillation_figure(args.output_dir)
    ensemble_steps_figure(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
