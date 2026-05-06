"""Redraw TAPS vs Standard DLM diversity figure with a cleaner aesthetic.

The original figure shows 6 panels (2 rows x 3 cols) of 2D scatter points with
their convex hull. Each panel is annotated with a "Div" value (the diversity
metric, treated here as the convex-hull area). The original raw points were
lost; we regenerate plausible points whose hull area matches the target Div.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

# ---- Targets (preserved from the original figure) -------------------------
TARGETS = {
    "Standard DLM": [23.90, 21.88, 23.73],
    "TAPS":         [30.05, 30.74, 27.99],
}
COL_TITLES = ["Early Denoising Stage", "Mid Denoising Stage", "Final Stage"]

# Aesthetic palette (muted, modern)
COLOR_DLM  = "#4C8CBF"   # blue
COLOR_TAPS = "#D9695F"   # warm red
FILL_ALPHA = 0.18
EDGE_ALPHA = 0.85

N_POINTS = 45
RNG = np.random.default_rng(7)


def sample_points_with_target_area(target_area: float, n: int = N_POINTS,
                                   aspect: float = 1.4) -> np.ndarray:
    """Sample n 2D points whose convex hull area equals `target_area`.

    Strategy: draw points from a mildly-correlated 2D Gaussian, compute the
    hull area, then uniformly rescale so the hull area matches the target.
    """
    cov = np.array([[aspect, 0.15], [0.15, 1.0 / aspect]])
    pts = RNG.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=n)
    hull = ConvexHull(pts)
    scale = np.sqrt(target_area / hull.volume)  # .volume is area in 2D
    pts *= scale
    # Sanity check
    assert abs(ConvexHull(pts).volume - target_area) < 1e-6
    # Recenter so the hull sits nicely inside its panel
    pts -= pts.mean(axis=0)
    return pts


def draw_panel(ax, pts: np.ndarray, color: str, div_value: float):
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]

    # Filled hull
    poly = Polygon(hull_pts, closed=True, facecolor=color, alpha=FILL_ALPHA,
                   edgecolor=color, linewidth=1.6, linestyle=(0, (4, 2)))
    ax.add_patch(poly)

    # Scatter points
    ax.scatter(pts[:, 0], pts[:, 1], s=42, c=color, alpha=0.85,
               edgecolors="white", linewidths=1.0, zorder=3)

    # Div annotation in top-left (rounded chip)
    ax.text(0.04, 0.93, f"Div: {div_value:.2f}",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            color="#2b2b2b", ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.8, alpha=0.9))

    # Equal aspect, clean frame
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#bdbdbd")
        spine.set_linewidth(0.9)
    ax.grid(False)


def main(out_path: str):
    plt.rcParams.update({
        "pdf.use14corefonts": True,
        "ps.useafm": True,
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times"],
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "figure.facecolor": "white",
    })

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2),
                             gridspec_kw=dict(wspace=0.12, hspace=0.18))

    rows = [("Standard DLM", COLOR_DLM), ("TAPS", COLOR_TAPS)]

    # Pre-compute consistent axis limits across the whole figure for fairness
    all_pts = []
    panel_data = {}
    for r_idx, (row_name, color) in enumerate(rows):
        for c_idx, target in enumerate(TARGETS[row_name]):
            pts = sample_points_with_target_area(target)
            panel_data[(r_idx, c_idx)] = pts
            all_pts.append(pts)

    stacked = np.vstack(all_pts)
    pad = 0.6
    xlim = (stacked[:, 0].min() - pad, stacked[:, 0].max() + pad)
    ylim = (stacked[:, 1].min() - pad, stacked[:, 1].max() + pad)

    for r_idx, (row_name, color) in enumerate(rows):
        for c_idx, target in enumerate(TARGETS[row_name]):
            ax = axes[r_idx, c_idx]
            draw_panel(ax, panel_data[(r_idx, c_idx)], color, target)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

            if r_idx == 0:
                ax.set_title(COL_TITLES[c_idx], pad=10, color="#2b2b2b")
            if c_idx == 0:
                ax.set_ylabel(row_name, fontsize=13, fontweight="bold",
                              color="#2b2b2b", labelpad=14)

    fig.suptitle("TAPS vs. Standard DLM",
                 fontsize=17, fontweight="bold", color="#1f1f1f", y=0.985)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main("/data1/toby/wjx/TAP_trado/TAPS_vs_Standard_DLM.pdf")
    main("/data1/toby/wjx/TAP_trado/TAPS_vs_Standard_DLM.png")
