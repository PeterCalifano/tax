#!/usr/bin/env python3
"""
examples/three_body/plot.py

Render manifolds.json — the L1 unstable manifold trajectories of the
planar Earth-Moon CR3BP — in the synodic rotating frame.

Run:
    cd /tmp/your_run_dir
    /path/to/build/examples/three_body_manifolds
    python3 /path/to/tax/examples/three_body/plot.py

Output: three_body_manifolds.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path.cwd()


# ---- Style (match the two-body figures) ------------------------------------
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":            8.5,
    "axes.labelsize":       8.5,
    "axes.titlesize":       10.0,
    "axes.titlepad":        4.0,
    "axes.linewidth":       0.6,
    "xtick.major.size":     2.5,
    "ytick.major.size":     2.5,
    "xtick.major.width":    0.55,
    "ytick.major.width":    0.55,
    "xtick.labelsize":      7.0,
    "ytick.labelsize":      7.0,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "legend.frameon":       False,
    "legend.fontsize":      7.0,
    "axes.grid":            True,
    "grid.linestyle":       ":",
    "grid.linewidth":       0.45,
    "grid.color":           "#9a9a9a",
    "grid.alpha":           0.6,
    "figure.dpi":           120,
    "savefig.dpi":          400,
    "savefig.bbox":         "tight",
})


def main() -> None:
    p = HERE / "manifolds.json"
    if not p.exists():
        print("manifolds.json not found. Run three_body_manifolds first.")
        return
    data = json.loads(p.read_text())

    cfg          = data["config"]
    lin          = data["linearization"]
    trajectories = data["trajectories"]
    mu           = cfg["mu"]
    x_L1         = cfg["x_L1"]
    earth_x      = cfg["earth_x"]
    moon_x       = cfg["moon_x"]
    lam          = lin["lambda_unstable"]

    moon_cmap  = plt.cm.Reds
    earth_cmap = plt.cm.Blues

    moon_traj  = [t for t in trajectories if t["branch"] == "moon"]
    earth_traj = [t for t in trajectories if t["branch"] == "earth"]

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)

    # ---- Manifold trajectories (each branch in a sequential colour map) ----
    for i, tr in enumerate(moon_traj):
        c = moon_cmap(0.4 + 0.55 * i / max(len(moon_traj) - 1, 1))
        ax.plot(tr["x"], tr["y"], color=c, lw=0.8, alpha=0.95, zorder=2)
    for i, tr in enumerate(earth_traj):
        c = earth_cmap(0.4 + 0.55 * i / max(len(earth_traj) - 1, 1))
        ax.plot(tr["x"], tr["y"], color=c, lw=0.8, alpha=0.95, zorder=2)

    # ---- Primaries + L1 ----
    ax.plot(earth_x, 0.0, marker="o", color="#1f77b4", markersize=11,
            markeredgecolor="black", markeredgewidth=0.5, zorder=10)
    ax.text(earth_x, -0.04, "Earth", ha="center", va="top", fontsize=7.5)

    ax.plot(moon_x, 0.0, marker="o", color="#aeaeae", markersize=6,
            markeredgecolor="black", markeredgewidth=0.5, zorder=10)
    ax.text(moon_x, -0.04, "Moon", ha="center", va="top", fontsize=7.5)

    ax.plot(x_L1, 0.0, marker="x", color="black", markersize=7,
            markeredgewidth=1.2, zorder=10)
    ax.text(x_L1, 0.04, r"$L_1$", ha="center", va="bottom", fontsize=8.5)

    ax.set_xlabel(r"$x$  (synodic, rotating)")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal", "box")
    ax.set_title(
        rf"L1 unstable manifolds — Earth-Moon CR3BP "
        rf"($\mu = {mu:.5f}$, $\lambda = {lam:.3f}$)",
        loc="left",
    )

    # ---- Legend ----
    legend_entries = [
        Line2D([], [], color=moon_cmap(0.7), lw=1.2,
               label="unstable manifold (Moon branch, $\\epsilon > 0$)"),
        Line2D([], [], color=earth_cmap(0.7), lw=1.2,
               label="unstable manifold (Earth branch, $\\epsilon < 0$)"),
        Line2D([], [], marker="o", linestyle="none", color="#1f77b4",
               markersize=7, markeredgecolor="black", markeredgewidth=0.5,
               label="Earth"),
        Line2D([], [], marker="o", linestyle="none", color="#aeaeae",
               markersize=5, markeredgecolor="black", markeredgewidth=0.5,
               label="Moon"),
        Line2D([], [], marker="x", linestyle="none", color="black",
               markersize=7, markeredgewidth=1.2, label="$L_1$"),
    ]
    ax.legend(handles=legend_entries, loc="upper right", fontsize=6.8)

    out_path = HERE / "three_body_manifolds.png"
    fig.savefig(out_path)
    print(f"wrote {out_path.name}")


if __name__ == "__main__":
    main()
