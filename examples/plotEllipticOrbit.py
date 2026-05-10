#!/usr/bin/env python3
"""Plot the output of examples/ellipticOrbit.

Generates ellipticOrbit.png with three panels:

  (a) the reference orbit with five perturbed trajectories;
  (b) the endpoint at t = tmax for the truth, the single-DA flow, and ADS;
  (c) endpoint error vs delta_vy for the single flow vs ADS.

Run after building & executing the C++ example so the four CSV files exist
in the same directory:

    orbit_reference.csv
    orbits_perturbed.csv
    endpoint_compare.csv
    ads_leaves.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory containing the CSV files written by ellipticOrbit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <data-dir>/ellipticOrbit.png)",
    )
    args = parser.parse_args()

    out_path = args.output or args.data_dir / "ellipticOrbit.png"

    ref = _load_csv(args.data_dir / "orbit_reference.csv")
    pert = _load_csv(args.data_dir / "orbits_perturbed.csv")
    cmp_ = _load_csv(args.data_dir / "endpoint_compare.csv")
    leaves = _load_csv(args.data_dir / "ads_leaves.csv")

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.7], hspace=0.32, wspace=0.28)
    ax_orbit = fig.add_subplot(gs[0, 0])
    ax_end = fig.add_subplot(gs[0, 1])
    ax_err = fig.add_subplot(gs[1, :])

    # ---- (a) Orbits in the (x, y) plane --------------------------------------
    ax_orbit.plot(ref["x"], ref["y"], "k-", lw=2.0, label="reference (δ=0)")
    cmap = plt.get_cmap("viridis")
    deltas = np.unique(pert["delta"])
    for d in deltas:
        if abs(d) < 1e-9:
            continue
        m = pert["delta"] == d
        c = cmap(0.5 + 0.5 * d)
        ax_orbit.plot(pert["x"][m], pert["y"][m], lw=1.0, color=c, alpha=0.85,
                      label=f"δ={d:+.1f}")
    ax_orbit.plot([0.0], [0.0], marker="*", color="orange", ms=14,
                  label="primary", zorder=5)
    ax_orbit.set_xlabel("x")
    ax_orbit.set_ylabel("y")
    ax_orbit.set_title("(a) Elliptic orbit and perturbed trajectories\n"
                       "(perturbing $v_y(0)$ over $[v_p \\pm \\Delta]$)")
    ax_orbit.set_aspect("equal", adjustable="datalim")
    ax_orbit.grid(True, alpha=0.3)
    ax_orbit.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # ---- (b) Endpoint at t = tmax ---------------------------------------------
    ax_end.scatter(cmp_["truex"], cmp_["truey"], s=42, marker="o",
                   facecolor="none", edgecolor="black", lw=1.2, label="truth")
    ax_end.scatter(cmp_["flowx"], cmp_["flowy"], s=24, marker="x",
                   color="tab:blue", label="single flow expansion")
    ax_end.scatter(cmp_["adsx"], cmp_["adsy"], s=24, marker="+",
                   color="tab:green", label="ADS piecewise flow")
    ax_end.set_xlabel("x at t = $t_{\\max}$")
    ax_end.set_ylabel("y at t = $t_{\\max}$")
    ax_end.set_title("(b) Endpoint of every perturbed IC at $t_{\\max}=\\pi$\n"
                     "(single-polynomial flow drifts visibly at the box edges)")
    ax_end.grid(True, alpha=0.3)
    ax_end.legend(loc="best", fontsize=9)

    # ---- (c) Endpoint error vs δ ---------------------------------------------
    err_flow = np.hypot(cmp_["truex"] - cmp_["flowx"], cmp_["truey"] - cmp_["flowy"])
    err_ads = np.hypot(cmp_["truex"] - cmp_["adsx"], cmp_["truey"] - cmp_["adsy"])
    ax_err.semilogy(cmp_["delta"], err_flow, "o-", color="tab:blue", ms=4,
                    label="single flow expansion")
    ax_err.semilogy(cmp_["delta"], err_ads, "+-", color="tab:green", ms=6,
                    label="ADS piecewise flow")

    # Annotate ADS sub-domain boundaries on the δ-axis
    centre = 0.5 * (leaves["vy_lo"][0] + leaves["vy_hi"][-1])
    half = 0.5 * (leaves["vy_hi"][-1] - leaves["vy_lo"][0])
    boundaries = sorted(set(list(leaves["vy_lo"]) + list(leaves["vy_hi"])))
    for vy_b in boundaries:
        d_b = (vy_b - centre) / half
        if -1.0 < d_b < 1.0:
            ax_err.axvline(d_b, color="tab:green", lw=0.6, ls="--", alpha=0.5)
    ax_err.text(0.02, 0.95, f"{len(leaves):d} ADS leaves (dashed: split lines)",
                transform=ax_err.transAxes, va="top", fontsize=9, color="tab:green")
    ax_err.set_xlabel("δ (normalised perturbation in $v_y(0)$)")
    ax_err.set_ylabel("$||x_{pred}(t_{\\max}) - x_{true}(t_{\\max})||$")
    ax_err.set_title("(c) Endpoint error vs perturbation: ADS stays uniformly accurate")
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.legend(loc="best", fontsize=9)

    fig.suptitle("Planar elliptic Kepler orbit (a=1, e=0.5): plain Taylor → "
                 "flow expansion → ADS",
                 fontsize=12, y=0.995)

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
