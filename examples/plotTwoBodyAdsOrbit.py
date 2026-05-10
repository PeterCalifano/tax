#!/usr/bin/env python3
"""Plot the splitting of the two-body example projected onto the orbit.

Generates twoBodyAdsOrbit.png with two panels:

  (a) AdsIntegrator         — IC-box partition pushed forward to (x, y) at
                              t = t_max, overlaid on the reference orbit.
  (b) LowOrderAdsIntegrator — same view.

Each panel has a small inset in the top-right corner showing the IC-space
partition for the same method.  All partition cells are rendered in greys
for a print-friendly look.

Run after building & executing the C++ example so the CSV files exist:

    twoBody_te_leaf_orbit.csv
    twoBody_lo_leaf_orbit.csv
    twoBody_te_leaves.csv
    twoBody_lo_leaves.csv
    twoBody_orbit_reference.csv
    twoBody_orbit_endpoint.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

# Paper-style palette.
LEAF_FACE = "#dcdcdc"
LEAF_EDGE = "#5a5a5a"
ORBIT_COLOR = "black"
ENDPOINT_COLOR = "#222222"
INSET_FACE = "#eeeeee"
INSET_EDGE = "#888888"
INSET_OUTLINE = "#000000"


def _load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


def _draw_leaf_polygons(ax, leaves: np.ndarray, *, lw: float = 0.55) -> None:
    """Overlay every leaf's pushed-forward boundary as a light-grey polygon."""
    leaf_ids = np.unique(leaves["leaf"]).astype(int)
    for li in leaf_ids:
        m = leaves["leaf"] == li
        pts = np.column_stack([leaves["x"][m], leaves["y"][m]])
        ax.add_patch(
            Polygon(
                pts,
                closed=True,
                facecolor=LEAF_FACE,
                edgecolor=LEAF_EDGE,
                linewidth=lw,
                alpha=0.85,
                zorder=2,
            )
        )


def _add_ic_inset(ax, ic_leaves: np.ndarray) -> None:
    """Draw the IC-space partition inset in the top-right of @p ax."""
    inset = ax.inset_axes([0.71, 0.71, 0.27, 0.27])
    x_lo = float(np.min(ic_leaves["x_lo"]))
    x_hi = float(np.max(ic_leaves["x_hi"]))
    v_lo = float(np.min(ic_leaves["vy_lo"]))
    v_hi = float(np.max(ic_leaves["vy_hi"]))
    for row in ic_leaves:
        inset.add_patch(
            Rectangle(
                (row["x_lo"], row["vy_lo"]),
                row["x_hi"] - row["x_lo"],
                row["vy_hi"] - row["vy_lo"],
                facecolor=INSET_FACE,
                edgecolor=INSET_EDGE,
                linewidth=0.4,
            )
        )
    inset.add_patch(
        Rectangle(
            (x_lo, v_lo),
            x_hi - x_lo,
            v_hi - v_lo,
            facecolor="none",
            edgecolor=INSET_OUTLINE,
            linewidth=0.9,
        )
    )
    inset.set_xlim(x_lo, x_hi)
    inset.set_ylim(v_lo, v_hi)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("IC partition", fontsize=8, pad=2)
    for spine in inset.spines.values():
        spine.set_edgecolor(INSET_OUTLINE)
        spine.set_linewidth(0.8)


def _plot_panel(
    ax,
    leaves: np.ndarray,
    ic_leaves: np.ndarray,
    orbit: np.ndarray,
    endpoint: np.ndarray,
    title: str,
    extent: tuple[float, float, float, float] | None,
) -> None:
    ax.plot(orbit["x"], orbit["y"], color=ORBIT_COLOR, lw=1.0, zorder=1)
    _draw_leaf_polygons(ax, leaves)
    ax.plot(
        endpoint["x"],
        endpoint["y"],
        marker="o",
        markerfacecolor="white",
        markeredgecolor=ENDPOINT_COLOR,
        markersize=6,
        markeredgewidth=1.2,
        linestyle="",
        zorder=5,
    )
    ax.plot(
        [0.0],
        [0.0],
        marker="+",
        color=ENDPOINT_COLOR,
        markersize=10,
        markeredgewidth=1.3,
        linestyle="",
        zorder=5,
    )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}  ({len(np.unique(leaves['leaf']))} subdomains)")
    ax.grid(True, alpha=0.18, linewidth=0.5)
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    _add_ic_inset(ax, ic_leaves)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory containing the CSV files written by twoBodyAdsComparison",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <script-dir>/twoBodyAdsOrbit.png)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Add a full-orbit overview row above the zoom",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )

    out_path = (
        args.output or Path(__file__).resolve().parent / "twoBodyAdsOrbit.png"
    )

    te = _load_csv(args.data_dir / "twoBody_te_leaf_orbit.csv")
    lo = _load_csv(args.data_dir / "twoBody_lo_leaf_orbit.csv")
    te_ic = _load_csv(args.data_dir / "twoBody_te_leaves.csv")
    lo_ic = _load_csv(args.data_dir / "twoBody_lo_leaves.csv")
    orbit = _load_csv(args.data_dir / "twoBody_orbit_reference.csv")
    endpoint = _load_csv(args.data_dir / "twoBody_orbit_endpoint.csv")

    # Zoom around the pushed-forward leaf cluster (with a little padding).
    all_x = np.concatenate([te["x"], lo["x"]])
    all_y = np.concatenate([te["y"], lo["y"]])
    pad_x = 0.10 * (all_x.max() - all_x.min() + 1e-12)
    pad_y = 0.10 * (all_y.max() - all_y.min() + 1e-12)
    zoom = (
        float(all_x.min() - pad_x),
        float(all_x.max() + pad_x),
        float(all_y.min() - pad_y),
        float(all_y.max() + pad_y),
    )

    if args.full:
        fig = plt.figure(figsize=(13, 11))
        gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.20)
        ax_te_full = fig.add_subplot(gs[0, 0])
        ax_lo_full = fig.add_subplot(gs[0, 1])
        ax_te_zoom = fig.add_subplot(gs[1, 0])
        ax_lo_zoom = fig.add_subplot(gs[1, 1])
        _plot_panel(ax_te_full, te, te_ic, orbit, endpoint,
                    "(a) AdsIntegrator — full orbit", None)
        _plot_panel(ax_lo_full, lo, lo_ic, orbit, endpoint,
                    "(b) LowOrderAdsIntegrator — full orbit", None)
        _plot_panel(ax_te_zoom, te, te_ic, orbit, endpoint,
                    "(c) AdsIntegrator — apoapsis zoom", zoom)
        _plot_panel(ax_lo_zoom, lo, lo_ic, orbit, endpoint,
                    "(d) LowOrderAdsIntegrator — apoapsis zoom", zoom)
    else:
        fig, (ax_te, ax_lo) = plt.subplots(1, 2, figsize=(13, 6))
        _plot_panel(ax_te, te, te_ic, orbit, endpoint,
                    "(a) AdsIntegrator — truncation-error split", zoom)
        _plot_panel(ax_lo, lo, lo_ic, orbit, endpoint,
                    "(b) LowOrderAdsIntegrator — nonlinearity-index split", zoom)
        fig.subplots_adjust(wspace=0.22)

    fig.suptitle(
        r"Two-body problem: IC partition pushed forward to $(x, y)$ at $t = t_{\max}$",
        fontsize=12,
        y=1.0,
    )

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
