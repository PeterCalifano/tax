#!/usr/bin/env python3
"""
examples/two_body/plot.py

Render the three example JSONs (taylor / ads / loads) into a minimal,
Nature-style 3-panel figure: thin axes, no top/right spines, no grid,
sans-serif labels, single shared colour bar at the bottom.

Output: two_body_box_evolution.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

HERE = Path.cwd()
METHODS = ("taylor", "ads", "loads")
LABELS  = {"taylor": "taylor", "ads": "ads", "loads": "loads"}

# ---- Minimal Nature-style rcParams ----------------------------------------
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":            8.0,
    "axes.labelsize":       8.0,
    "axes.titlesize":       9.0,
    "axes.titlepad":        4.0,
    "axes.linewidth":       0.55,
    "xtick.major.size":     2.5,
    "ytick.major.size":     2.5,
    "xtick.minor.size":     1.5,
    "ytick.minor.size":     1.5,
    "xtick.major.width":    0.5,
    "ytick.major.width":    0.5,
    "xtick.labelsize":      7.0,
    "ytick.labelsize":      7.0,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "legend.frameon":       False,
    "legend.fontsize":      7.0,
    "axes.grid":            False,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "figure.dpi":           120,
    "savefig.dpi":          400,
    "savefig.bbox":         "tight",
})


def load(method: str) -> dict | None:
    p = HERE / f"{method}.json"
    return json.loads(p.read_text()) if p.exists() else None


def panel_xy_limits(*datasets: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    xs, ys = [], []
    for d in datasets:
        ref = d.get("reference_orbit", {})
        xs.extend(ref.get("x0", []))
        ys.extend(ref.get("x1", []))
        for poly in d.get("polygons", []):
            if "leaves" in poly:
                for lf in poly["leaves"]:
                    xs.extend(lf["x"]); ys.extend(lf["y"])
            else:
                xs.extend(poly["x"]); ys.extend(poly["y"])
    if not xs:
        return (-2.0, 2.0), (-2.0, 2.0)
    xa, xb = min(xs), max(xs)
    ya, yb = min(ys), max(ys)
    px = 0.05 * max(xb - xa, 1e-12)
    py = 0.05 * max(yb - ya, 1e-12)
    return (xa - px, xb + px), (ya - py, yb + py)


def draw_panel(ax: plt.Axes, data: dict, *,
               panel_letter: str, panel_label: str,
               xlim, ylim, cmap, norm) -> None:
    polygons = data["polygons"]

    # ---- Reference orbit (very subtle backdrop) ----
    ref = data.get("reference_orbit")
    if ref is not None:
        ax.plot(ref["x0"], ref["x1"], color="#9a9a9a",
                lw=0.45, alpha=0.85, zorder=1)

    # ---- Polygon snapshots ----
    for snap in polygons:
        color = cmap(norm(snap["t"]))
        if "leaves" in snap:
            for lf in snap["leaves"]:
                ax.fill(lf["x"], lf["y"], color=color, alpha=0.70,
                        edgecolor="black", linewidth=0.25, zorder=2)
        else:
            ax.fill(snap["x"], snap["y"], color=color, alpha=0.70,
                    edgecolor="black", linewidth=0.25, zorder=2)

    # ---- Primary at origin (single black dot) ----
    ax.plot(0.0, 0.0, marker="o", color="black",
            markersize=3.0, zorder=10)

    # ---- Frame, axes ----
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", "box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.tick_params(length=2.5, width=0.5)

    # ---- Panel letter (a, b, c) — bold, upper-left of axes box ----
    ax.text(-0.18, 1.04, panel_letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")
    # ---- Panel label (method name) ----
    ax.text(0.0, 1.04, panel_label, transform=ax.transAxes,
            fontsize=9, va="bottom", ha="left")


def main() -> None:
    loaded = [(m, load(m)) for m in METHODS]
    loaded = [(m, d) for m, d in loaded if d is not None]
    if not loaded:
        print("No JSON outputs found. Run two_body_taylor / two_body_ads / two_body_loads first.")
        return

    t_final = max(d["config"]["t_final"] for _, d in loaded)
    cmap    = plt.cm.viridis
    norm    = Normalize(vmin=0.0, vmax=t_final)
    xlim, ylim = panel_xy_limits(*(d for _, d in loaded))

    n        = len(loaded)
    fig      = plt.figure(figsize=(2.6 * n, 3.1), constrained_layout=True)
    gs       = fig.add_gridspec(2, n, height_ratios=[1.0, 0.03])

    letters = "abcdef"
    for col, (m, d) in enumerate(loaded):
        ax = fig.add_subplot(gs[0, col])
        draw_panel(ax, d,
                   panel_letter=letters[col],
                   panel_label=LABELS[m],
                   xlim=xlim, ylim=ylim,
                   cmap=cmap, norm=norm)

    # ---- Shared horizontal colour bar ----
    cax  = fig.add_subplot(gs[1, :])
    sm   = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(r"$t$", labelpad=1.5)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(length=2.0, width=0.4, labelsize=6.5)

    out_path = HERE / "two_body_box_evolution.png"
    fig.savefig(out_path)
    print(f"wrote {out_path.name}")

    # ---- Terminal summary ----
    print()
    print(f"  {'method':<8}  {'elapsed':>9}   {'snaps':>5}   leaves (per snap)")
    print(f"  {'-'*8:<8}  {'-'*9:>9}   {'-'*5:>5}   {'-'*32}")
    for m, d in loaded:
        elapsed = d.get("timing", {}).get("elapsed_ms", float("nan")) / 1e3
        polys   = d["polygons"]
        if "leaves" in polys[0]:
            leaves = [len(s["leaves"]) for s in polys]
        else:
            leaves = [1] * len(polys)
        print(f"  {m:<8}  {elapsed:>7.2f} s   {len(polys):>5}   {leaves}")


if __name__ == "__main__":
    main()
