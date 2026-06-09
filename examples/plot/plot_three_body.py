#!/usr/bin/env python3
"""Render the CR3BP example figures from cr3bp_{taylor,ads,loads}.json.

Usage:
    python3 plot_three_body.py [--data DIR] [--out DIR]

Reads the JSON files written by three_body_taylor / three_body_ads /
three_body_loads (run them first; they write to their working directory)
and produces:

    three_body_flow.png    — IC box image stretching along the L1 manifold
    three_body_ads.png     — single polynomial vs ADS leaves at breakdown
    three_body_leaves.png  — leaf counts vs time, with the e^{lambda t} slope
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
    }
)


def load(path: pathlib.Path):
    with open(path) as f:
        return json.load(f)


def draw_frame(ax, params):
    """Moon + L1 markers shared by both spatial figures."""
    mu, x_l1 = params["mu"], params["x_L1"]
    ax.scatter([1.0 - mu], [0.0], s=160, color="0.6", edgecolor="0.3", zorder=5)
    ax.annotate("Moon", (1.0 - mu, 0.0), textcoords="offset points",
                xytext=(0, 10), ha="center")
    ax.scatter([x_l1], [0.0], marker="x", s=70, color="#d1495b", zorder=5)
    ax.annotate("$L_1$", (x_l1, 0.0), textcoords="offset points",
                xytext=(0, -16), ha="center", color="#d1495b")


def plot_flow(taylor, out_dir):
    """IC box image under one flow polynomial, drifting from L1 to the Moon."""
    fig, (ax_l1, ax) = plt.subplots(1, 2, figsize=(11.8, 5.4), layout="constrained",
                                    width_ratios=[1.0, 1.5])
    params = taylor["params"]
    ref = taylor["reference_orbit"]
    snaps = taylor["snapshots"]
    norm = Normalize(0.0, params["t_final"])
    cmap = plt.get_cmap("viridis")

    # Left: the slow exponential stretch near L1 (early snapshots).
    ax_l1.plot(ref["x0"], ref["x1"], color="0.55", lw=1.0, zorder=1)
    ax_l1.scatter([params["x_L1"]], [0.0], marker="x", s=70, color="#d1495b", zorder=5)
    ax_l1.annotate("$L_1$", (params["x_L1"], 0.0), textcoords="offset points",
                   xytext=(-2, 8), ha="center", color="#d1495b")
    for snap in snaps[:8]:
        color = cmap(norm(snap["t"]))
        for leaf in snap["leaves"]:
            ax_l1.fill(leaf["x"], leaf["y"], facecolor=color, alpha=0.6,
                       edgecolor=color, lw=1.4, zorder=3)
    ax_l1.set_xlim(0.8355, 0.8475)
    ax_l1.set_ylim(-0.0042, 0.0012)
    ax_l1.set_xlabel("$x$  (rotating frame)")
    ax_l1.set_ylabel("$y$")
    ax_l1.set_title(f"near $L_1$:  $t \\leq {snaps[7]['t']:.2f}$")

    # Right: the full transit to the Moon (the t = 2 bulge running out of
    # frame is the single polynomial breaking down at the flyby).
    ax.plot(ref["x0"], ref["x1"], color="0.55", lw=1.0, zorder=1,
            label="reference trajectory")
    draw_frame(ax, params)
    for snap in snaps:
        color = cmap(norm(snap["t"]))
        for leaf in snap["leaves"]:
            ax.fill(leaf["x"], leaf["y"], facecolor=color, alpha=0.5,
                    edgecolor=color, lw=1.2, zorder=3)
    ax.set_xlim(0.80, 1.03)
    ax.set_ylim(-0.085, 0.085)
    ax.set_xlabel("$x$  (rotating frame)")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_title("transit to the Moon")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("snapshot time  $t$")

    fig.suptitle("CR3BP: IC box escaping $L_1$ along the unstable manifold")
    fig.savefig(out_dir / "three_body_flow.png", dpi=150)
    plt.close(fig)


def plot_ads(taylor, ads, out_dir):
    """Single polynomial vs ADS partition at the Moon-approach breakdown."""
    # Pick the first snapshot where ADS uses many leaves (the breakdown).
    k = next(i for i, s in enumerate(ads["snapshots"]) if len(s["leaves"]) > 32)
    st, sa = taylor["snapshots"][k], ads["snapshots"][k]

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    params = ads["params"]
    ref = ads["reference_orbit"]
    ax.plot(ref["x0"], ref["x1"], color="0.55", lw=1.0, zorder=1)
    draw_frame(ax, params)

    leaf = st["leaves"][0]
    ax.plot(leaf["x"], leaf["y"], color="#d1495b", lw=1.6, ls="--", zorder=4,
            label=f"single polynomial ($P={taylor['params']['P']}$)")

    cmap = plt.get_cmap("viridis")
    n = len(sa["leaves"])
    for i, lf in enumerate(sa["leaves"]):
        c = cmap(0.15 + 0.7 * i / max(n - 1, 1))
        ax.fill(lf["x"], lf["y"], facecolor=c, alpha=0.9, edgecolor=c, lw=1.6,
                zorder=3, label=f"ADS leaves ($P={params['P']}$, n={n})" if i == 0 else None)

    xs = np.concatenate([lf["x"] for lf in sa["leaves"]])
    ys = np.concatenate([lf["y"] for lf in sa["leaves"]])
    pad_x = 0.30 * (xs.max() - xs.min())
    pad_y = 0.30 * (ys.max() - ys.min())
    ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
    ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)

    ax.set_xlabel("$x$  (rotating frame)")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.legend(loc="best", framealpha=0.9)
    ax.set_title(f"Moon approach at $t = {st['t']:.2f}$ — "
                 "one polynomial vs the ADS partition")

    fig.tight_layout()
    fig.savefig(out_dir / "three_body_ads.png", dpi=150)
    plt.close(fig)


def plot_leaves(ads, loads, out_dir):
    """Leaf counts vs time, against the manifold's e^{lambda t} stretching."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    lam = ads["params"]["lambda_unstable"]

    for run, color, marker, label in [
        (ads, "#1f77b4", "o", f"ADS (truncation, $P={ads['params']['P']}$)"),
        (loads, "#d1495b", "s", f"LOADS (NLI, $P={loads['params']['P']}$)"),
    ]:
        t = [s["t"] for s in run["snapshots"]]
        n = [len(s["leaves"]) for s in run["snapshots"]]
        ax.plot(t, n, marker=marker, color=color, lw=1.6, label=label)

    t_ref = np.linspace(1.4, 2.6, 40)
    ax.plot(t_ref, 0.05 * np.exp(lam * t_ref), color="0.5", ls=":",
            label=f"$\\propto e^{{\\lambda t}}$,  $\\lambda = {lam:.2f}$")

    ax.set_xlabel("snapshot time  $t$")
    ax.set_ylabel("number of leaves")
    ax.set_yscale("log", base=2)
    ax.legend(framealpha=0.9, loc="upper left")
    ax.set_title("Subdivisions track the manifold stretching")
    fig.tight_layout()
    fig.savefig(out_dir / "three_body_leaves.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=".",
                    help="directory containing cr3bp_{taylor,ads,loads}.json")
    ap.add_argument("--out", type=pathlib.Path, default=".",
                    help="output directory for the PNG figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    taylor = load(args.data / "cr3bp_taylor.json")
    ads = load(args.data / "cr3bp_ads.json")
    loads = load(args.data / "cr3bp_loads.json")

    plot_flow(taylor, args.out)
    plot_ads(taylor, ads, args.out)
    plot_leaves(ads, loads, args.out)
    print(f"wrote three_body_flow.png, three_body_ads.png, three_body_leaves.png -> {args.out}")


if __name__ == "__main__":
    main()
