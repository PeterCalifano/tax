#!/usr/bin/env python3
"""Plot the splitting of the two-body example projected onto the orbit.

Generates twoBodyAdsOrbit.png with two panels:

  (a) AdsIntegrator         — IC-box partition pushed forward to (x, y) at
                              t = t_max, overlaid on the reference orbit.
  (b) LowOrderAdsIntegrator — same view.

Each leaf is drawn as the polygon obtained by walking the perimeter of its
IC box (δ ∈ [-1, 1]²) and evaluating its flow polynomial at every boundary
point.  The interior of the polygon is the image of the leaf's IC sub-box
through the propagated DA flow.

Run after building & executing the C++ example so the CSV files exist:

    twoBody_te_leaf_orbit.csv
    twoBody_lo_leaf_orbit.csv
    twoBody_orbit_reference.csv
    twoBody_orbit_endpoint.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def _load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


def _plot_split_on_orbit(
    ax,
    leaves: np.ndarray,
    orbit: np.ndarray,
    endpoint: np.ndarray,
    title: str,
    extent: tuple[float, float, float, float] | None = None,
) -> None:
    """Draw the reference orbit and overlay each leaf's pushed-forward
    boundary as a filled polygon."""
    ax.plot(orbit["x"], orbit["y"], color="black", lw=1.2,
            label="reference orbit", zorder=1, alpha=0.85)
    ax.plot([0.0], [0.0], marker="*", color="orange", ms=14,
            label="primary", zorder=5)
    ax.plot(endpoint["x"], endpoint["y"], marker="o", color="red", ms=8,
            label="$t = t_{\\max}$", zorder=5)

    cmap = plt.get_cmap("tab20")
    leaf_ids = np.unique(leaves["leaf"]).astype(int)
    for ci, li in enumerate(leaf_ids):
        m = leaves["leaf"] == li
        pts = np.column_stack([leaves["x"][m], leaves["y"][m]])
        poly = Polygon(
            pts,
            closed=True,
            facecolor="none",
            edgecolor=cmap(ci % 20),
            linewidth=0.8,
            alpha=0.85,
            zorder=2,
        )
        ax.add_patch(poly)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}\n({len(leaf_ids)} subdomains)")
    ax.grid(True, alpha=0.3)
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


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
        help="Show full orbit + zoom (2x2 layout) instead of the default 1x2 zoom",
    )
    args = parser.parse_args()

    out_path = (
        args.output or Path(__file__).resolve().parent / "twoBodyAdsOrbit.png"
    )

    te = _load_csv(args.data_dir / "twoBody_te_leaf_orbit.csv")
    lo = _load_csv(args.data_dir / "twoBody_lo_leaf_orbit.csv")
    orbit = _load_csv(args.data_dir / "twoBody_orbit_reference.csv")
    endpoint = _load_csv(args.data_dir / "twoBody_orbit_endpoint.csv")

    # Zoom extent matching the smallest bounding box around all pushed-forward
    # leaf polygons — used for the lower two panels.
    all_x = np.concatenate([te["x"], lo["x"]])
    all_y = np.concatenate([te["y"], lo["y"]])
    pad_x = 0.10 * (all_x.max() - all_x.min() + 1e-12)
    pad_y = 0.10 * (all_y.max() - all_y.min() + 1e-12)
    zoom_extent = (
        float(all_x.min() - pad_x),
        float(all_x.max() + pad_x),
        float(all_y.min() - pad_y),
        float(all_y.max() + pad_y),
    )

    if args.full:
        fig = plt.figure(figsize=(14, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22)
        ax_te_full = fig.add_subplot(gs[0, 0])
        ax_lo_full = fig.add_subplot(gs[0, 1])
        ax_te_zoom = fig.add_subplot(gs[1, 0])
        ax_lo_zoom = fig.add_subplot(gs[1, 1])
        _plot_split_on_orbit(
            ax_te_full, te, orbit, endpoint,
            "(a) AdsIntegrator — full orbit", None,
        )
        _plot_split_on_orbit(
            ax_lo_full, lo, orbit, endpoint,
            "(b) LowOrderAdsIntegrator — full orbit", None,
        )
        _plot_split_on_orbit(
            ax_te_zoom, te, orbit, endpoint,
            "(c) AdsIntegrator — zoom on apoapsis cluster", zoom_extent,
        )
        _plot_split_on_orbit(
            ax_lo_zoom, lo, orbit, endpoint,
            "(d) LowOrderAdsIntegrator — zoom on apoapsis cluster", zoom_extent,
        )
    else:
        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(1, 2, wspace=0.22)
        ax_te = fig.add_subplot(gs[0, 0])
        ax_lo = fig.add_subplot(gs[0, 1])
        _plot_split_on_orbit(
            ax_te, te, orbit, endpoint,
            "(a) AdsIntegrator — truncation-error split", zoom_extent,
        )
        _plot_split_on_orbit(
            ax_lo, lo, orbit, endpoint,
            "(b) LowOrderAdsIntegrator — nonlinearity-index split", zoom_extent,
        )

    fig.suptitle(
        "Two-body problem: IC partition pushed forward to (x, y) at $t = t_{\\max}$",
        fontsize=12,
        y=0.995,
    )

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
