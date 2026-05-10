#!/usr/bin/env python3
"""Plot the output of examples/twoBodyAdsComparison.

Generates twoBodyAdsComparison.png with four panels:

  (a) IC-space leaves produced by AdsIntegrator         (truncation-error split)
  (b) IC-space leaves produced by LowOrderAdsIntegrator (NLI split)
  (c) Endpoint error of AdsIntegrator         over (δx, δvy) ∈ [-1, 1]²
  (d) Endpoint error of LowOrderAdsIntegrator over (δx, δvy) ∈ [-1, 1]²

Run after building & executing the C++ example so the three CSV files
exist in the same directory:

    twoBody_ads_comparison.csv
    twoBody_te_leaves.csv
    twoBody_lo_leaves.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle


def _load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


def _plot_leaves(
    ax,
    leaves: np.ndarray,
    title: str,
    box_xy_extent: tuple[float, float, float, float],
) -> None:
    """Plot the IC-space leaves as colour-filled rectangles.

    Each leaf gets a deterministic colour drawn from the tab20 cycle so
    neighbouring leaves are visually distinct without relying on alpha
    blending.
    """
    x_lo, x_hi, vy_lo, vy_hi = box_xy_extent
    cmap = plt.get_cmap("tab20")
    for i, row in enumerate(leaves):
        rect = Rectangle(
            (row["x_lo"], row["vy_lo"]),
            row["x_hi"] - row["x_lo"],
            row["vy_hi"] - row["vy_lo"],
            linewidth=0.8,
            edgecolor="black",
            facecolor=cmap(i % 20),
            alpha=0.7,
        )
        ax.add_patch(rect)
    # Outer box for context.
    ax.add_patch(
        Rectangle(
            (x_lo, vy_lo),
            x_hi - x_lo,
            vy_hi - vy_lo,
            linewidth=1.6,
            edgecolor="black",
            facecolor="none",
            zorder=4,
        )
    )
    ax.set_xlim(x_lo - 0.05 * (x_hi - x_lo), x_hi + 0.05 * (x_hi - x_lo))
    ax.set_ylim(vy_lo - 0.05 * (vy_hi - vy_lo), vy_hi + 0.05 * (vy_hi - vy_lo))
    ax.set_xlabel("x(0)")
    ax.set_ylabel("$v_y(0)$")
    ax.set_title(f"{title}\n({len(leaves)} subdomains)")
    ax.grid(True, alpha=0.3)


def _plot_error(
    ax,
    samples: np.ndarray,
    column: str,
    title: str,
    vmin: float,
    vmax: float,
):
    """Plot the per-sample error on a log-scaled colour grid."""
    dx_vals = np.unique(samples["delta_x"])
    dv_vals = np.unique(samples["delta_vy"])
    nx, nv = len(dx_vals), len(dv_vals)

    grid = np.full((nv, nx), np.nan)
    dx_idx = {v: i for i, v in enumerate(dx_vals)}
    dv_idx = {v: i for i, v in enumerate(dv_vals)}
    for row in samples:
        i = dx_idx[row["delta_x"]]
        j = dv_idx[row["delta_vy"]]
        grid[j, i] = row[column]

    im = ax.pcolormesh(
        dx_vals,
        dv_vals,
        grid,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
        shading="auto",
    )
    ax.set_xlabel("δx (normalised)")
    ax.set_ylabel("δ$v_y$ (normalised)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    return im


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
        help="Output PNG path (default: <script-dir>/twoBodyAdsComparison.png)",
    )
    args = parser.parse_args()

    out_path = (
        args.output
        or Path(__file__).resolve().parent / "twoBodyAdsComparison.png"
    )

    samples = _load_csv(args.data_dir / "twoBody_ads_comparison.csv")
    te_leaves = _load_csv(args.data_dir / "twoBody_te_leaves.csv")
    lo_leaves = _load_csv(args.data_dir / "twoBody_lo_leaves.csv")

    # Common IC-space extent (use the AdsIntegrator leaves to recover the
    # outer box; both methods see the same initial domain).
    box_x_lo = float(np.min(te_leaves["x_lo"]))
    box_x_hi = float(np.max(te_leaves["x_hi"]))
    box_v_lo = float(np.min(te_leaves["vy_lo"]))
    box_v_hi = float(np.max(te_leaves["vy_hi"]))
    extent = (box_x_lo, box_x_hi, box_v_lo, box_v_hi)

    # Shared error colour-scale across both methods so they are visually
    # comparable.
    err_te = samples["err_te"]
    err_lo = samples["err_lo"]
    eps = 1e-16
    err_min = max(eps, float(np.min(np.concatenate([err_te, err_lo]))))
    err_max = float(np.max(np.concatenate([err_te, err_lo])))

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.28)
    ax_te_leaves = fig.add_subplot(gs[0, 0])
    ax_lo_leaves = fig.add_subplot(gs[0, 1])
    ax_te_err = fig.add_subplot(gs[1, 0])
    ax_lo_err = fig.add_subplot(gs[1, 1])

    _plot_leaves(
        ax_te_leaves,
        te_leaves,
        "(a) AdsIntegrator leaves\n(truncation-error criterion)",
        extent,
    )
    _plot_leaves(
        ax_lo_leaves,
        lo_leaves,
        "(b) LowOrderAdsIntegrator leaves\n(nonlinearity-index criterion)",
        extent,
    )
    im_te = _plot_error(
        ax_te_err,
        samples,
        "err_te",
        "(c) Endpoint error — AdsIntegrator",
        err_min,
        err_max,
    )
    im_lo = _plot_error(
        ax_lo_err,
        samples,
        "err_lo",
        "(d) Endpoint error — LowOrderAdsIntegrator",
        err_min,
        err_max,
    )

    cbar = fig.colorbar(
        im_lo,
        ax=[ax_te_err, ax_lo_err],
        shrink=0.85,
        pad=0.02,
        location="right",
    )
    cbar.set_label("$||x_{\\mathrm{pred}}(t_{\\max}) - x_{\\mathrm{true}}(t_{\\max})||$")

    fig.suptitle(
        "Planar two-body problem: AdsIntegrator vs LowOrderAdsIntegrator\n"
        "same tolerance (1e-3), same IC box, half-orbit propagation",
        fontsize=12,
        y=0.995,
    )

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
