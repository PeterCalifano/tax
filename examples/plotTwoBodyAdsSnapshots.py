#!/usr/bin/env python3
"""Plot a time-evolution storyboard of the IC partition.

Generates twoBodyAdsSnapshots.png with one row per integrator and one
column per snapshot time.  Each cell shows the IC partition pushed
forward to (x, y) at that time, overlaid on the reference orbit.  An
inset in the top-right of every cell shows the IC-space partition for
the corresponding (t, method).

Run after building & executing the C++ example so the CSV files exist:

    twoBody_snapshots.csv
    twoBody_snapshots_meta.csv
    twoBody_orbit_reference.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

LEAF_FACE = "#dcdcdc"
LEAF_EDGE = "#5a5a5a"
ORBIT_COLOR = "black"
INSET_FACE = "#eeeeee"
INSET_EDGE = "#888888"
INSET_OUTLINE = "#000000"


def _load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def _draw_leaf_polygons(ax, leaves: np.ndarray, lw: float = 0.65) -> None:
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
                alpha=0.9,
                zorder=2,
            )
        )


def _add_ic_inset(ax, ic_rows: np.ndarray) -> None:
    """Render the IC partition that produced @p ic_rows as a grey inset."""
    inset = ax.inset_axes([0.71, 0.71, 0.27, 0.27])
    x_lo = float(np.min(ic_rows["x_lo"]))
    x_hi = float(np.max(ic_rows["x_hi"]))
    v_lo = float(np.min(ic_rows["vy_lo"]))
    v_hi = float(np.max(ic_rows["vy_hi"]))
    for row in ic_rows:
        inset.add_patch(
            Rectangle(
                (row["x_lo"], row["vy_lo"]),
                row["x_hi"] - row["x_lo"],
                row["vy_hi"] - row["vy_lo"],
                facecolor=INSET_FACE,
                edgecolor=INSET_EDGE,
                linewidth=0.35,
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
    inset.set_title("IC partition", fontsize=7, pad=2)
    for spine in inset.spines.values():
        spine.set_edgecolor(INSET_OUTLINE)
        spine.set_linewidth(0.7)


def _plot_snapshot(
    ax,
    leaves: np.ndarray,
    ic_rows: np.ndarray,
    orbit: np.ndarray,
    title: str,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    ax.plot(orbit["x"], orbit["y"], color=ORBIT_COLOR, lw=0.6, zorder=1, alpha=0.5)
    _draw_leaf_polygons(ax, leaves)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    if show_xlabel:
        ax.set_xlabel("x")
    if show_ylabel:
        ax.set_ylabel("y")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.18, linewidth=0.4)
    _add_ic_inset(ax, ic_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <script-dir>/twoBodyAdsSnapshots.png)",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.7,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )

    out_path = (
        args.output
        or Path(__file__).resolve().parent / "twoBodyAdsSnapshots.png"
    )

    snaps = _load_csv(args.data_dir / "twoBody_snapshots.csv")
    ic = _load_csv(args.data_dir / "twoBody_snapshots_ic.csv")
    meta = _load_csv(args.data_dir / "twoBody_snapshots_meta.csv")
    orbit = _load_csv(args.data_dir / "twoBody_orbit_reference.csv")

    snap_ids = np.unique(meta["snapshot"]).astype(int)
    n_snap = len(snap_ids)

    snap_methods = snaps["method"]
    ic_methods = ic["method"]

    # Each column has its own zoom on the local pushed-forward cluster so
    # individual leaves are visible as small rotated rectangles instead of
    # dots on the full orbit.
    fig, axes = plt.subplots(2, n_snap, figsize=(3.1 * n_snap, 6.6))
    if n_snap == 1:
        axes = axes.reshape(2, 1)

    for col, k in enumerate(snap_ids):
        meta_row = meta[meta["snapshot"] == k][0]
        t_k = float(meta_row["t"])
        te_n = int(meta_row["te_leaves"])
        lo_n = int(meta_row["lo_leaves"])

        m_te = (snaps["snapshot"] == k) & (snap_methods == "te")
        m_lo = (snaps["snapshot"] == k) & (snap_methods == "lo")
        ic_te = ic[(ic["snapshot"] == k) & (ic_methods == "te")]
        ic_lo = ic[(ic["snapshot"] == k) & (ic_methods == "lo")]

        # Zoom that contains both methods' clusters with generous padding so
        # the leaves read as boxes rather than dots.
        col_x = np.concatenate([snaps["x"][m_te], snaps["x"][m_lo]])
        col_y = np.concatenate([snaps["y"][m_te], snaps["y"][m_lo]])
        cx_lo, cx_hi = float(col_x.min()), float(col_x.max())
        cy_lo, cy_hi = float(col_y.min()), float(col_y.max())
        # Keep a square aspect inside the panel by padding the shorter axis.
        span = max(cx_hi - cx_lo, cy_hi - cy_lo)
        pad = 0.40 * max(span, 1e-3)
        cx = 0.5 * (cx_lo + cx_hi)
        cy = 0.5 * (cy_lo + cy_hi)
        zoom_x = (cx - 0.5 * span - pad, cx + 0.5 * span + pad)
        zoom_y = (cy - 0.5 * span - pad, cy + 0.5 * span + pad)

        _plot_snapshot(
            axes[0, col],
            snaps[m_te],
            ic_te,
            orbit,
            f"t = {t_k:.3f}    ({te_n} leaves)",
            show_xlabel=False,
            show_ylabel=(col == 0),
        )
        axes[0, col].set_xlim(zoom_x)
        axes[0, col].set_ylim(zoom_y)
        _plot_snapshot(
            axes[1, col],
            snaps[m_lo],
            ic_lo,
            orbit,
            f"t = {t_k:.3f}    ({lo_n} leaves)",
            show_xlabel=True,
            show_ylabel=(col == 0),
        )
        axes[1, col].set_xlim(zoom_x)
        axes[1, col].set_ylim(zoom_y)

    axes[0, 0].annotate(
        "AdsIntegrator\n(truncation error)",
        xy=(-0.20, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    axes[1, 0].annotate(
        "LowOrderAds\n(nonlinearity index)",
        xy=(-0.20, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    fig.suptitle(
        r"Two-body problem: IC partition evolution through one orbital period",
        fontsize=12,
        y=0.995,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.10,
                        wspace=0.08, hspace=0.22)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
