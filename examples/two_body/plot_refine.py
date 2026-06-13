#!/usr/bin/env python3
"""
examples/two_body/plot_refine.py

Animate the propagate-then-assess ADS refinement (refine.json):

  * Left panel  — the (x, y) plane. The grey curve is the reference orbit;
    pale dots are the Monte-Carlo reference cloud at the current time; the
    filled polygons are the box images at the current iteration, coloured by
    their depth in the refinement tree. The animation sweeps t0 -> t_final
    for iteration 0 (a single box), then iteration 1 (≈2 boxes), and so on
    until the partition has converged onto the true set.

  * Right panel — RMS error of the piecewise-polynomial prediction against
    the Monte-Carlo cloud versus the number of boxes (log-log). A marker
    tracks the iteration currently playing on the left.

Outputs (into --out, default the current directory):
  two_body_refine.gif              the animation
  two_body_refine_convergence.png  the static convergence curve

Usage:
  python3 plot_refine.py --data . --out figs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         9.0,
    "axes.titlesize":    10.5,
    "axes.linewidth":    0.7,
    "axes.grid":         True,
    "grid.linestyle":    ":",
    "grid.linewidth":    0.5,
    "grid.color":        "#9a9a9a",
    "grid.alpha":        0.55,
    "figure.dpi":        110,
})

DEPTH_CMAP = plt.cm.viridis


def load(data_dir: Path) -> dict:
    return json.loads((data_dir / "refine.json").read_text())


def cloud_limits(data: dict, pad: float = 0.12):
    xs, ys = [], []
    for snap in data["monte_carlo"]:
        xs.extend(snap["x"])
        ys.extend(snap["y"])
    ref = data.get("reference_orbit", {})
    xs.extend(ref.get("x0", []))
    ys.extend(ref.get("x1", []))
    xa, xb, ya, yb = min(xs), max(xs), min(ys), max(ys)
    dx, dy = pad * (xb - xa), pad * (yb - ya)
    return (xa - dx, xb + dx), (ya - dy, yb + dy)


CRITERIA = {
    "coefficient_match": ("coefficient match", "#1f5fbf", "o"),
    "volume_ratio":      ("volume ratio",      "#1ba36b", "s"),
}


def draw_convergence(ax, data: dict, highlight: int | None = None) -> None:
    comp = data["comparison"]
    for key, (label, color, marker) in CRITERIA.items():
        curve = comp.get(key)
        if not curve:
            continue
        nb  = [it["n_boxes"] for it in curve]
        rms = [it["rms"] for it in curve]
        ax.plot(nb, rms, "-", marker=marker, color=color, lw=1.4, ms=4.5,
                label=label, zorder=3)
    if highlight is not None:
        prim = comp["coefficient_match"]
        hi   = min(highlight, len(prim) - 1)
        ax.plot(prim[hi]["n_boxes"], prim[hi]["rms"], "o", ms=11,
                mfc="none", mec="#d1361b", mew=2.0, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of sub-boxes")
    ax.set_ylabel("RMS error vs Monte-Carlo")
    ax.set_title("convergence (tol = $10^{-6}$)", loc="left")
    ax.legend(loc="upper right", fontsize=7.5)


def make_convergence_png(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.5), constrained_layout=True)
    draw_convergence(ax, data)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"wrote {out.name}")


def make_gif(data: dict, out: Path, fps: int = 7) -> None:
    xlim, ylim = cloud_limits(data)
    iters      = data["iterations"]
    snap_times = data["snap_times"]
    t_final    = data["params"]["t_final"]
    mc         = data["monte_carlo"]
    max_depth  = max((lf["depth"] for it in iters for snap in it["snapshots"]
                      for lf in snap["leaves"]), default=1)
    norm = Normalize(vmin=0, vmax=max(max_depth, 1))

    # Frame list: for each iteration sweep all snapshot times, holding the
    # final time of each iteration for a beat so the converged set lingers.
    frames = []
    for ki in range(len(iters)):
        for si in range(len(snap_times)):
            frames.append((ki, si))
        frames.extend([(ki, len(snap_times) - 1)] * 3)

    fig = plt.figure(figsize=(9.2, 4.3), constrained_layout=True)
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0])
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    ref = data.get("reference_orbit", {})

    def render(frame):
        ki, si = frame
        it   = iters[ki]
        snap = it["snapshots"][si]

        axL.clear()
        axL.plot(ref.get("x0", []), ref.get("x1", []),
                 color="#8a8a8a", lw=0.8, alpha=0.9, zorder=1)
        axL.scatter(mc[si]["x"], mc[si]["y"], s=3.0, color="#d1361b",
                    alpha=0.45, linewidths=0, zorder=2)
        for lf in snap["leaves"]:
            axL.fill(lf["x"], lf["y"], color=DEPTH_CMAP(norm(lf["depth"])),
                     alpha=0.62, edgecolor="black", linewidth=0.3, zorder=3)
        axL.plot(0.0, 0.0, marker="o", color="black", ms=4, zorder=10)
        axL.set_xlim(xlim)
        axL.set_ylim(ylim)
        axL.set_aspect("equal", "box")
        axL.set_xlabel(r"$x$")
        axL.set_ylabel(r"$y$")
        axL.set_title(
            f"iteration {it['iter']}  ·  {it['n_boxes']} "
            f"box{'es' if it['n_boxes'] != 1 else ''}  ·  "
            rf"$t/T = {snap_times[si] / t_final:.2f}$",
            loc="left")

        axR.clear()
        draw_convergence(axR, data, highlight=ki)
        axR.text(0.04, 0.06,
                 f"coeff-match RMS = {it['rms']:.2e}",
                 transform=axR.transAxes, ha="left", va="bottom", fontsize=8.5,
                 color="#d1361b")
        return []

    anim = FuncAnimation(fig, render, frames=frames, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {out.name}  ({len(frames)} frames)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".", help="directory containing refine.json")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--fps", type=int, default=7)
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load(data_dir)
    make_convergence_png(data, out_dir / "two_body_refine_convergence.png")
    make_gif(data, out_dir / "two_body_refine.gif", fps=args.fps)

    print()
    print(f"  {'iter':>4}  {'boxes':>6}  {'area':>14}  {'rms':>12}")
    print(f"  {'-'*4:>4}  {'-'*6:>6}  {'-'*14:>14}  {'-'*12:>12}")
    for it in data["iterations"]:
        print(f"  {it['iter']:>4}  {it['n_boxes']:>6}  {it['area']:>14.6g}  {it['rms']:>12.4e}")


if __name__ == "__main__":
    main()
