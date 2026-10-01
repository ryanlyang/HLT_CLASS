"""Create paper graphics for full-cardinality HCWDL matching and support flow.

The figures are deterministic schematics.  They do not read campaign results
or final-test data, and their synthetic particles are chosen only to explain
the registered scientific construction.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch


BG = "#FBFAF7"
INK = "#263238"
MUTED = "#66757D"
GRID = "#DDE3E6"
HLT = "#2378C3"
OFFLINE = "#EE8758"
GREEN = "#278F67"
RED = "#CB5A5A"
PURPLE = "#7259B3"
PALE_BLUE = "#E8F2FA"
PALE_ORANGE = "#FCEDE5"
PALE_GRAY = "#EEF1F2"


def _configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": BG,
    })


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor=BG,
    )
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor=BG,
    )
    plt.close(fig)


def _wrapped_delta_phi(left: float, right: float) -> float:
    return (left - right + math.pi) % (2.0 * math.pi) - math.pi


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    deta = float(left[0] - right[0])
    dphi = _wrapped_delta_phi(float(left[1]), float(right[1]))
    return math.hypot(deta, dphi)


def _lexicographic_bottleneck_assignment(
    hlt: np.ndarray,
    offline: np.ndarray,
) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...]]:
    """Solve the small synthetic example exactly by enumeration."""

    if len(hlt) > len(offline):
        raise ValueError("the explanatory example expects HLT to be smaller")
    best_key: tuple[tuple[int, ...], tuple[int, ...]] | None = None
    best_pairs: tuple[tuple[int, int], ...] | None = None
    best_distances: tuple[float, ...] | None = None
    for offline_indices in itertools.permutations(range(len(offline)), len(hlt)):
        pairs = tuple(enumerate(offline_indices))
        distances = tuple(_distance(hlt[i], offline[j]) for i, j in pairs)
        quantized = tuple(
            sorted((round(value / 1.0e-7) for value in distances), reverse=True)
        )
        # Native indices are sufficient for the final tie in this schematic;
        # its synthetic distances have a unique primary optimum.
        key = (quantized, offline_indices)
        if best_key is None or key < best_key:
            best_key = key
            best_pairs = pairs
            best_distances = distances
    if best_pairs is None or best_distances is None:
        raise RuntimeError("synthetic bottleneck assignment is empty")
    return best_pairs, best_distances


def full_cardinality_matching_figure(output_dir: Path) -> None:
    """Explain smaller-side saturation and lexicographic bottleneck matching."""

    hlt = np.asarray([
        (-0.34, 0.24),
        (-0.08, 0.31),
        (0.17, 0.22),
        (-0.27, -0.07),
        (0.02, -0.10),
        (0.31, -0.19),
    ])
    offline = np.asarray([
        (-0.30, 0.19),
        (-0.04, 0.36),
        (0.21, 0.17),
        (-0.31, -0.13),
        (0.08, -0.04),
        (0.26, -0.25),
        (-0.49, 0.44),
        (0.44, 0.38),
    ])
    pairs, distances = _lexicographic_bottleneck_assignment(hlt, offline)
    used_offline = {j for _, j in pairs}
    unused_offline = sorted(set(range(len(offline))) - used_offline)

    fig = plt.figure(figsize=(11.6, 4.8), facecolor=BG)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.25, 1.0),
        left=0.065,
        right=0.975,
        top=0.82,
        bottom=0.14,
        wspace=0.28,
    )
    ax_match = fig.add_subplot(grid[0, 0])
    ax_rank = fig.add_subplot(grid[0, 1])

    fig.suptitle(
        "Full-cardinality bottleneck particle pairing",
        x=0.5,
        y=0.955,
        fontsize=19,
        fontweight="bold",
        color=INK,
    )
    ax_match.set_title("(a) Maximum-cardinality assignment", pad=10, color=INK)
    ax_match.set_facecolor("white")
    ax_match.grid(True, color=GRID, linewidth=0.7, alpha=0.72)
    ax_match.set_axisbelow(True)
    ax_match.set_xlabel(r"jet-centered $\Delta\eta$")
    ax_match.set_ylabel(r"jet-centered $\Delta\phi$")
    ax_match.set_xlim(-0.57, 0.53)
    ax_match.set_ylim(-0.34, 0.52)
    ax_match.set_aspect("equal", adjustable="box")

    for (i, j), distance in zip(pairs, distances):
        ax_match.plot(
            [hlt[i, 0], offline[j, 0]],
            [hlt[i, 1], offline[j, 1]],
            color=GREEN,
            linewidth=2.1,
            alpha=0.78,
            zorder=2,
        )

    ax_match.scatter(
        offline[:, 0],
        offline[:, 1],
        s=95,
        marker="o",
        facecolor=OFFLINE,
        edgecolor="white",
        linewidth=1.2,
        zorder=4,
    )
    ax_match.scatter(
        hlt[:, 0],
        hlt[:, 1],
        s=125,
        marker="o",
        facecolor="white",
        edgecolor=HLT,
        linewidth=2.5,
        zorder=5,
    )
    ax_match.scatter(
        offline[unused_offline, 0],
        offline[unused_offline, 1],
        s=155,
        marker="x",
        color=MUTED,
        linewidth=2.1,
        zorder=6,
    )

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=HLT,
               markeredgewidth=2.2, markersize=8, label="HLT"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=OFFLINE, markeredgecolor="white",
               markersize=8, label="offline"),
        Line2D([0], [0], color=GREEN, linewidth=2.2, label="pair"),
        Line2D([0], [0], marker="x", color=MUTED, linewidth=0, markersize=8,
               markeredgewidth=2, label="unused"),
    ]
    ax_match.legend(
        handles=legend,
        loc="lower left",
        fontsize=8.2,
        frameon=True,
        facecolor="white",
        edgecolor=GRID,
        ncol=2,
    )
    ax_match.text(
        0.98,
        0.97,
        r"$n_{\mathrm{HLT}}=6,\ n_{\mathrm{off}}=8$" "\n" r"$k=\min(6,8)=6$",
        transform=ax_match.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color=INK,
        bbox={"boxstyle": "round,pad=0.35", "fc": PALE_BLUE, "ec": HLT, "alpha": 0.95},
    )

    ordered = sorted(distances, reverse=True)
    labels = ["1  worst", "2", "3", "4", "5", "6"]
    colors = [RED, PURPLE, GREEN, GREEN, GREEN, GREEN]
    positions = np.arange(len(ordered))
    ax_rank.set_title("(b) Bottleneck objective", pad=10, color=INK)
    ax_rank.set_facecolor("white")
    bars = ax_rank.barh(
        positions,
        ordered,
        color=colors,
        height=0.62,
        edgecolor="white",
        linewidth=0.8,
    )
    ax_rank.set_yticks(positions, labels)
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel(r"selected pair separation $\Delta R$")
    ax_rank.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.75)
    ax_rank.set_axisbelow(True)
    ax_rank.spines[["top", "right", "left"]].set_visible(False)
    ax_rank.tick_params(axis="y", length=0)
    ax_rank.set_xlim(0, max(ordered) * 1.34)
    for bar, value in zip(bars, ordered):
        ax_rank.text(
            value + max(ordered) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=9,
            color=INK,
        )
    _save(fig, output_dir, "fullcard_bottleneck_matching")


def _draw_particle(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    *,
    aligned: bool,
    extra: bool,
    removed: bool,
) -> None:
    if removed:
        ax.scatter([x], [y], s=360, marker="o", facecolor=PALE_GRAY, edgecolor=GRID, linewidth=1.2, zorder=2)
        ax.plot([x - 0.16, x + 0.16], [y - 0.16, y + 0.16], color=MUTED, linewidth=1.9, zorder=3)
        ax.plot([x - 0.16, x + 0.16], [y + 0.16, y - 0.16], color=MUTED, linewidth=1.9, zorder=3)
        color = MUTED
    elif extra:
        ax.scatter([x], [y], s=345, marker="D", facecolor=PALE_ORANGE, edgecolor=OFFLINE, linewidth=1.9, zorder=2)
        color = OFFLINE
    else:
        edge = HLT if aligned else OFFLINE
        width = 3.0 if aligned else 1.8
        ax.scatter([x], [y], s=360, marker="o", facecolor=PALE_ORANGE, edgecolor=edge, linewidth=width, zorder=2)
        color = INK
    ax.text(x, y, label, ha="center", va="center", fontsize=9.0, color=color, fontweight="bold", zorder=4)


def support_homotopy_figure(output_dir: Path) -> None:
    """Show only the U000 and U100 endpoints of the support transition."""

    stages = [
        (False, r"$\mathcal{X}_{\mathrm{off}}$", 10),
        (True, r"$\mathcal{X}_{\mathrm{align}}$", 6),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), facecolor=BG)
    fig.subplots_adjust(left=0.075, right=0.925, top=0.83, bottom=0.19, wspace=0.65)
    fig.suptitle(
        r"$\mathrm{U000}\ \longrightarrow\ \mathrm{U100}$",
        x=0.5,
        y=0.96,
        fontsize=22,
        fontweight="bold",
        color=INK,
    )
    positions = tuple((column, 4.38 - 0.82 * row) for row in range(5) for column in (0.58, 1.42))
    for stage_index, (aligned, label, active_count) in enumerate(stages):
        ax = axes[stage_index]
        ax.set_xlim(0, 2)
        ax.set_ylim(-0.05, 5.75)
        ax.axis("off")
        card = FancyBboxPatch(
            (0.05, 0.18),
            1.90,
            4.92,
            boxstyle="round,pad=0.06,rounding_size=0.16",
            facecolor="white",
            edgecolor=GRID,
            linewidth=1.25,
            zorder=0,
        )
        ax.add_patch(card)
        ax.text(1.0, 5.48, label, ha="center", va="center", fontsize=15, color=INK, fontweight="bold")

        for token in range(6):
            x, y = positions[token]
            _draw_particle(
                ax,
                x,
                y,
                rf"$p_{{{token + 1}}}$",
                aligned=aligned,
                extra=False,
                removed=False,
            )
        for extra_index in range(4):
            x, y = positions[6 + extra_index]
            _draw_particle(
                ax,
                x,
                y,
                rf"$u_{{{extra_index + 1}}}$",
                aligned=False,
                extra=not aligned,
                removed=aligned,
            )

        ax.text(
            1.0,
            0.45,
            rf"$n={active_count}$",
            ha="center",
            va="center",
            fontsize=10.5,
            color=HLT if aligned else MUTED,
            fontweight="bold" if aligned else "normal",
        )

    axes[0].annotate(
        "",
        xy=(1.54, 0.53),
        xytext=(1.08, 0.53),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": PURPLE, "lw": 3.0, "mutation_scale": 18},
        annotation_clip=False,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALE_ORANGE,
               markeredgecolor=OFFLINE, markeredgewidth=1.8, markersize=9,
               label="offline support"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALE_ORANGE,
               markeredgecolor=HLT, markeredgewidth=2.8, markersize=9,
               label="aligned HLT slot"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=PALE_ORANGE,
               markeredgecolor=OFFLINE, markeredgewidth=1.6, markersize=8,
               label="unpaired offline"),
        Line2D([0], [0], marker="x", color=MUTED, linewidth=0, markeredgewidth=1.8,
               markersize=8, label="removed"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=4,
        frameon=False,
        fontsize=9.2,
        handletextpad=0.6,
        columnspacing=1.8,
    )
    _save(fig, output_dir, "particle_support_homotopy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/Ladder_KD/figures"),
        help="directory receiving PNG and PDF figures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_matplotlib()
    full_cardinality_matching_figure(args.output_dir)
    support_homotopy_figure(args.output_dir)
    for stem in ("fullcard_bottleneck_matching", "particle_support_homotopy"):
        print((args.output_dir / f"{stem}.png").resolve())
        print((args.output_dir / f"{stem}.pdf").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
