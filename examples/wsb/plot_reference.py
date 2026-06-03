#!/usr/bin/env python3
"""
examples/wsb/plot_reference.py

Quick-look plot for the WSB-reference C++ output. Two panels in the
Sun-Earth synodic frame, both Earth-centred:

  (a) Wide view — entire trajectory plus L1, L2, and the Moon's mean
      orbit radius as a reference circle. Distances in km.
  (b) Zoom on Earth vicinity (~3 Moon orbits across) — same
      trajectory, lets you eyeball the parking-orbit segment.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

HERE = Path.cwd()


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
    p = HERE / "wsb_reference.json"
    if not p.exists():
        print("wsb_reference.json not found. Run wsb_reference first.")
        return
    data = json.loads(p.read_text())

    cfg   = data["config"]
    traj  = data["trajectory"]
    ic    = data["ic"]
    AU_km = cfg["AU_km"]

    # Convert to km, Earth-centred.
    x = np.asarray(traj["x_earth"]) * AU_km
    y = np.asarray(traj["y_earth"]) * AU_km
    t = np.asarray(traj["t"])
    t_days = t * cfg["time_unit_days"]

    L1_dx   = (cfg["L1_x"] - cfg["earth_x"]) * AU_km
    L2_dx   = (cfg["L2_x"] - cfg["earth_x"]) * AU_km
    moon_r  = cfg["moon_orbit_AU"] * AU_km
    hill_r  = cfg["earth_hill_radius"] * AU_km

    # Colour the trajectory by time.
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0.0, vmax=t_days.max())

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), constrained_layout=True)

    # ---- (a) Wide view ----
    ax = axes[0]
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], color=cmap(norm(t_days[i])),
                lw=0.55, zorder=2)
    # Earth, L1, L2, Hill sphere, Moon orbit circle.
    ax.add_patch(Circle((0, 0), hill_r, fill=False, edgecolor="#1f77b4",
                        linewidth=0.6, linestyle="--", alpha=0.6, zorder=1))
    ax.add_patch(Circle((0, 0), moon_r, fill=False, edgecolor="#aeaeae",
                        linewidth=0.6, linestyle=":", alpha=0.7, zorder=1))
    ax.plot(0, 0, marker="o", color="#1f77b4", markersize=8,
            markeredgecolor="black", markeredgewidth=0.4, zorder=10)
    ax.plot(L1_dx, 0, marker="x", color="black", markersize=5,
            markeredgewidth=1.0, zorder=10)
    ax.plot(L2_dx, 0, marker="x", color="black", markersize=5,
            markeredgewidth=1.0, zorder=10)
    ax.text(0, 1e5, "Earth", ha="center", va="bottom", fontsize=7.5)
    ax.text(L2_dx, 1e5, r"$L_2$", ha="center", va="bottom", fontsize=8.0)
    ax.text(L1_dx, 1e5, r"$L_1$", ha="center", va="bottom", fontsize=8.0)
    ax.text(moon_r * 1.05, 0, "Moon orbit", ha="left", va="center",
            fontsize=6.5, color="#666666")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Earth-centred $x$  (km)")
    ax.set_ylabel("Earth-centred $y$  (km)")
    ax.set_title("Wide view (Sun-Earth synodic, Earth-centred)", loc="left")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("days since launch")

    # ---- (b) Earth-vicinity zoom ----
    ax = axes[1]
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], color=cmap(norm(t_days[i])),
                lw=0.55, zorder=2)
    ax.add_patch(Circle((0, 0), moon_r, fill=False, edgecolor="#aeaeae",
                        linewidth=0.6, linestyle=":", alpha=0.7, zorder=1))
    ax.plot(0, 0, marker="o", color="#1f77b4", markersize=8,
            markeredgecolor="black", markeredgewidth=0.4, zorder=10)
    zoom = 3.0 * moon_r
    ax.set_xlim(-zoom, zoom)
    ax.set_ylim(-zoom, zoom)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Earth-centred $x$  (km)")
    ax.set_ylabel("Earth-centred $y$  (km)")
    ax.set_title("Zoom (~3 Moon orbits across)", loc="left")
    ax.text(moon_r * 1.05, 0, "Moon orbit", ha="left", va="center",
            fontsize=6.5, color="#666666")

    fig.suptitle(
        rf"WSB reference draft — v0 = {ic['v_inertial_kms']:.2f} km/s "
        rf"(excess = {ic['v_excess_kms']:.2f} km/s), "
        rf"r_park = {ic['r_park_km']:.0f} km, "
        rf"t_final = {cfg['t_final_days']:.0f} days",
        y=1.02,
    )

    out_path = HERE / "wsb_reference.png"
    fig.savefig(out_path)
    print(f"wrote {out_path.name}")


if __name__ == "__main__":
    main()
