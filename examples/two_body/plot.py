#!/usr/bin/env python3
"""
examples/two_body/plot.py

Render the JSON output of the three two-body examples (taylor / ads /
loads) as a single comparison figure.

Usage:

    cd /tmp/two_body_run                      # or any working dir
    /path/to/build/examples/two_body_taylor
    /path/to/build/examples/two_body_ads
    /path/to/build/examples/two_body_loads
    python3 /path/to/tax/examples/two_body/plot.py

Output: two_body_box_evolution.png — a 3-panel figure (taylor / ads /
loads) showing the IC box pushed forward in time, colour-coded by the
snapshot time. Each panel overlays the reference orbit and (for ADS /
LOADS) annotates the leaf count per snapshot.

Robust to missing files: any method whose JSON isn't present is silently
skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path.cwd()

METHODS = ("taylor", "ads", "loads")
TITLES  = {
    "taylor": "Single Taylor flow polynomial",
    "ads":    "ADS (truncation criterion)",
    "loads":  "LOADS (NLI criterion)",
}


def load(method: str) -> dict | None:
    p = HERE / f"{method}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def panel_xy_limits(*datasets: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Union x/y bounds across reference orbits and polygons of all panels."""
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
        return (-2, 2), (-2, 2)
    xa, xb = min(xs), max(xs)
    ya, yb = min(ys), max(ys)
    px, py = 0.05 * (xa - xb), 0.05 * (ya - yb)
    return (xa + px, xb - px), (ya + py, yb - py)


def draw_panel(ax: plt.Axes, data: dict, *, xlim, ylim, title: str) -> None:
    polygons = data["polygons"]
    n        = len(polygons)
    cmap     = plt.cm.plasma
    norm     = plt.Normalize(vmin=0, vmax=max(n - 1, 1))

    # Reference orbit.
    ref = data.get("reference_orbit")
    if ref is not None:
        ax.plot(ref["x0"], ref["x1"], "k-", lw=0.8, alpha=0.4, zorder=0)

    # Polygon snapshots.
    for i, snap in enumerate(polygons):
        color = cmap(norm(i))
        label = f"t = {snap['t']:.2f}"
        if "leaves" in snap:
            label += f"  ({len(snap['leaves'])} leaves)"
            for lf in snap["leaves"]:
                ax.fill(lf["x"], lf["y"], color=color, alpha=0.55,
                        edgecolor="black", linewidth=0.4)
        else:
            ax.fill(snap["x"], snap["y"], color=color, alpha=0.55,
                    edgecolor="black", linewidth=0.4)
        # Marker on the legend (one per snapshot).
        ax.fill([], [], color=color, alpha=0.55, label=label)

    ax.plot(0, 0, "k*", markersize=12, zorder=5)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)


def main() -> None:
    loaded = [(m, load(m)) for m in METHODS]
    loaded = [(m, d) for m, d in loaded if d is not None]
    if not loaded:
        print("No JSON outputs found. Run two_body_taylor / two_body_ads / two_body_loads first.")
        return

    xlim, ylim = panel_xy_limits(*(d for _, d in loaded))

    fig, axes = plt.subplots(1, len(loaded), figsize=(6.5 * len(loaded), 7),
                             squeeze=False)
    for ax, (m, d) in zip(axes[0], loaded):
        draw_panel(ax, d, xlim=xlim, ylim=ylim, title=TITLES[m])

    fig.suptitle("IC box pushed forward in time — Kepler orbit (a=1, e=0.5, 1 orbit)",
                 y=0.99)
    fig.tight_layout()
    out_path = HERE / "two_body_box_evolution.png"
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path.name}")

    # Print a small timing summary.
    print()
    print(f"{'method':<10} {'elapsed':>10} {'snaps':>8} {'leaves (per snap)':>30}")
    for m, d in loaded:
        elapsed_ms = d.get("timing", {}).get("elapsed_ms", float("nan"))
        polys      = d["polygons"]
        if "leaves" in polys[0]:
            leaves = [len(s["leaves"]) for s in polys]
        else:
            leaves = [1] * len(polys)
        print(f"{m:<10} {elapsed_ms/1e3:>8.2f} s {len(polys):>8} "
              f"{str(leaves):>30}")


if __name__ == "__main__":
    main()
