#!/usr/bin/env python3
"""Plot the IC box pushed forward in time, ADS leaves vs single flow.

Reads the snapshot CSVs written by ellipticOrbit and renders a 2-panel
figure:

    left  – ADS leaves at every snapshot time, overlaid on the reference
            orbit.  At later times the box has spread enough that the
            DA truncation tolerance is exceeded and ADS subdivides the
            domain into several polynomial pieces (filled polygons).
    right – single multivariate-Taylor flow polynomial at the same times
            (one polygon per time), showing how a single polynomial would
            try (and gradually fail) to capture the same image set.

A small inset shows the ADS partition of the IC box at the latest
snapshot — the rectangular tiles in normalised δ-space.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


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
        help="Output PNG (default: <script-dir>/ellipticBoxEvolution.png)",
    )
    args = parser.parse_args()

    out_path = args.output or Path(__file__).resolve().parent / "ellipticBoxEvolution.png"

    ref = _load(args.data_dir / "orbit_reference.csv")
    ads_box = _load(args.data_dir / "ads_box_snapshots.csv")
    flow_box = _load(args.data_dir / "flow_box_snapshots.csv")
    leaves_ic = _load(args.data_dir / "ads_box_leaves.csv")

    snapshots = sorted(set(ads_box["snapshot"].astype(int).tolist()))
    n_snap = len(snapshots)

    # One snapshot time per snapshot index.
    snap_t = {s: float(ads_box["t"][ads_box["snapshot"] == s][0]) for s in snapshots}

    fig, (ax_ads, ax_flow) = plt.subplots(1, 2, figsize=(15.0, 7.5),
                                           constrained_layout=True)

    # ---------- Common axis range ----------
    all_x = np.concatenate([ref["x"], ads_box["x"], flow_box["x"]])
    all_y = np.concatenate([ref["y"], ads_box["y"], flow_box["y"]])
    pad_x = 0.05 * (all_x.max() - all_x.min())
    pad_y = 0.05 * (all_y.max() - all_y.min())
    xlim = (all_x.min() - pad_x, all_x.max() + pad_x)
    ylim = (all_y.min() - pad_y, all_y.max() + pad_y)

    # Snapshot color map (time → colour).
    snap_cmap = plt.get_cmap("plasma")
    snap_colors = [snap_cmap(0.05 + 0.85 * i / max(n_snap - 1, 1)) for i in range(n_snap)]
    snap_t_arr = np.array([snap_t[s] for s in snapshots])

    # ---------- Left: ADS ----------
    ax_ads.plot(ref["x"], ref["y"], "k-", lw=1.0, alpha=0.5, label="reference orbit")
    ax_ads.plot([0.0], [0.0], "*", color="orange", ms=14, zorder=5, label="primary")
    ax_ads.plot(ref["x"][0], ref["y"][0], "ko", ms=5,
                markerfacecolor="white", zorder=5,
                label="initial state (centre)")

    for col, snap in enumerate(snapshots):
        m = ads_box["snapshot"] == snap
        leaf_ids = sorted(set(ads_box["leaf_idx"][m].astype(int).tolist()))
        for li in leaf_ids:
            sel = m & (ads_box["leaf_idx"] == li)
            order = np.argsort(ads_box["sample_idx"][sel])
            xs = ads_box["x"][sel][order]
            ys = ads_box["y"][sel][order]
            ax_ads.fill(xs, ys, alpha=0.55, color=snap_colors[col],
                        edgecolor="black", linewidth=0.5)
        # Add a label for each snapshot time once
        ax_ads.fill([], [], color=snap_colors[col], alpha=0.55,
                    label=f"t = {snap_t[snap]:.3f}  ({len(leaf_ids)} leaves)")

    ax_ads.set_xlim(xlim)
    ax_ads.set_ylim(ylim)
    ax_ads.set_aspect("equal", adjustable="box")
    ax_ads.set_xlabel("x(t)")
    ax_ads.set_ylabel("y(t)")
    ax_ads.set_title("ADS: piecewise polynomial flow\n"
                     "(IC box partitioned, each leaf shown as a filled patch)")
    ax_ads.grid(True, alpha=0.25)
    if n_snap <= 6:
        ax_ads.legend(loc="upper right", fontsize=8, framealpha=0.95)

    # ---------- Right: single flow ----------
    ax_flow.plot(ref["x"], ref["y"], "k-", lw=1.0, alpha=0.5, label="reference orbit")
    ax_flow.plot([0.0], [0.0], "*", color="orange", ms=14, zorder=5, label="primary")
    ax_flow.plot(ref["x"][0], ref["y"][0], "ko", ms=5,
                 markerfacecolor="white", zorder=5,
                 label="initial state (centre)")

    for col, snap in enumerate(snapshots):
        m = flow_box["snapshot"] == snap
        order = np.argsort(flow_box["sample_idx"][m])
        xs = flow_box["x"][m][order]
        ys = flow_box["y"][m][order]
        ax_flow.fill(xs, ys, alpha=0.55, color=snap_colors[col],
                     edgecolor="black", linewidth=0.5,
                     label=f"t = {snap_t[snap]:.3f}")

    ax_flow.set_xlim(xlim)
    ax_flow.set_ylim(ylim)
    ax_flow.set_aspect("equal", adjustable="box")
    ax_flow.set_xlabel("x(t)")
    ax_flow.set_ylabel("y(t)")
    ax_flow.set_title("Single multivariate-Taylor flow polynomial\n"
                      "(one polygon per time; shape is the polynomial image of the IC box)")
    ax_flow.grid(True, alpha=0.25)
    if n_snap <= 6:
        ax_flow.legend(loc="upper right", fontsize=8, framealpha=0.95)

    # Colorbar showing snapshot time when too many to legend.
    if n_snap > 6:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=float(snap_t_arr.min()), vmax=float(snap_t_arr.max()))
        sm = ScalarMappable(norm=norm, cmap=snap_cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=[ax_ads, ax_flow], orientation="horizontal",
                          fraction=0.06, pad=0.08, shrink=0.6, aspect=40)
        cb.set_label("snapshot time t  (rad,  full period = 2π ≈ 6.283)",
                     fontsize=9)

    # ---------- Inset: IC-space splits at latest snapshot ----------
    last_snap = snapshots[-1]
    sel = leaves_ic["snapshot"] == last_snap
    inset_w, inset_h = 0.11, 0.22
    ax_ic = fig.add_axes([0.305, 0.18, inset_w, inset_h])
    ax_ic.add_patch(patches.Rectangle((-1, -1), 2, 2, fill=False, ec="black", lw=1.4))
    leaf_cmap = plt.get_cmap("tab20")
    for li, dy_lo, dy_hi, dvy_lo, dvy_hi in zip(
        leaves_ic["leaf_idx"][sel].astype(int),
        leaves_ic["dy_lo"][sel],
        leaves_ic["dy_hi"][sel],
        leaves_ic["dvy_lo"][sel],
        leaves_ic["dvy_hi"][sel],
    ):
        ax_ic.add_patch(patches.Rectangle(
            (dy_lo, dvy_lo), dy_hi - dy_lo, dvy_hi - dvy_lo,
            facecolor=leaf_cmap((int(li) * 3) % 20), edgecolor="black",
            linewidth=0.6, alpha=0.55))
    ax_ic.set_xlim(-1.05, 1.05)
    ax_ic.set_ylim(-1.05, 1.05)
    ax_ic.set_aspect("equal", adjustable="box")
    ax_ic.set_xlabel(r"$\delta_{y_0}$", fontsize=8)
    ax_ic.set_ylabel(r"$\delta_{v_{y0}}$", fontsize=8)
    ax_ic.set_title(f"IC partition\n(t = {snap_t[last_snap]:.3f})", fontsize=8)
    ax_ic.tick_params(labelsize=7)

    fig.suptitle(
        "IC box pushed forward in time — Kepler orbit (a=1, e=0.5)",
        fontsize=12,
    )

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
