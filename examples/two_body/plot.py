#!/usr/bin/env python3
"""
examples/two_body/plot.py

Render the JSON output of the three two-body examples (taylor / ads /
loads) as a single comparison figure intended for publication-style
presentation.

Usage:

    cd /tmp/two_body_run                  # or any working dir
    /path/to/build/examples/two_body_taylor
    /path/to/build/examples/two_body_ads
    /path/to/build/examples/two_body_loads
    python3 /path/to/tax/examples/two_body/plot.py

Output: two_body_box_evolution.png — three side-by-side panels showing
the IC box pushed forward in time, colour-coded by snapshot time via a
shared horizontal colour bar.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

HERE = Path.cwd()

METHODS = ("taylor", "ads", "loads")
TITLES  = {
    "taylor": "Single Taylor flow polynomial",
    "ads":    "ADS — truncation criterion",
    "loads":  "LOADS — NLI criterion",
}

# ---- Publication-style matplotlib defaults ---------------------------------
plt.rcParams.update({
    "font.family":          "serif",
    "font.size":            10.0,
    "axes.titlesize":       11.0,
    "axes.labelsize":       10.0,
    "axes.titleweight":     "regular",
    "axes.linewidth":       0.8,
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "legend.fontsize":      8.5,
    "legend.frameon":       False,
    "figure.titlesize":     12.5,
    "figure.dpi":           120,
    "savefig.dpi":          240,
    "savefig.bbox":         "tight",
    "axes.grid":            True,
    "grid.linewidth":       0.4,
    "grid.alpha":           0.25,
    "grid.linestyle":       ":",
})


# ---- IO --------------------------------------------------------------------
def load(method: str) -> dict | None:
    p = HERE / f"{method}.json"
    return json.loads(p.read_text()) if p.exists() else None


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
        return (-2.0, 2.0), (-2.0, 2.0)
    xa, xb = min(xs), max(xs)
    ya, yb = min(ys), max(ys)
    px = 0.05 * max(xb - xa, 1e-12)
    py = 0.05 * max(yb - ya, 1e-12)
    return (xa - px, xb + px), (ya - py, yb + py)


# ---- Drawing ---------------------------------------------------------------
def _criterion_subtitle(data: dict) -> str:
    """Compact criterion summary for the panel subtitle."""
    crit = data.get("criterion")
    if not crit:
        # The taylor example has no criterion → single polynomial path.
        return r"single multivariate Taylor polynomial"
    label = {"truncation": "trunc.", "nli": "NLI"}.get(crit["type"], crit["type"])
    return rf"{label}: tol = {crit['tol']:.0e},  depth $\leq$ {crit['maxDepth']}"


def _leaves_summary(polygons: list[dict]) -> str:
    """Like '1 → 4 leaves' showing first vs last snapshot leaf count."""
    if "leaves" not in polygons[0]:
        return ""
    first = len(polygons[0]["leaves"])
    last  = len(polygons[-1]["leaves"])
    if first == last:
        return f"{last} leaves throughout"
    return rf"leaves: {first} $\rightarrow$ {last}"


def draw_panel(ax: plt.Axes, data: dict, *,
               xlim, ylim, title: str, cmap, norm) -> None:
    polygons = data["polygons"]

    # ---- Reference orbit (thin grey backdrop) ----
    ref = data.get("reference_orbit")
    if ref is not None:
        ax.plot(ref["x0"], ref["x1"], color="#404040",
                lw=0.6, alpha=0.65, zorder=1)

    # ---- Polygon snapshots ----
    for snap in polygons:
        color = cmap(norm(snap["t"]))
        if "leaves" in snap:
            for lf in snap["leaves"]:
                ax.fill(lf["x"], lf["y"], color=color, alpha=0.65,
                        edgecolor="black", linewidth=0.35, zorder=2)
        else:
            ax.fill(snap["x"], snap["y"], color=color, alpha=0.65,
                    edgecolor="black", linewidth=0.35, zorder=2)

    # ---- Primary (Sun-like star at origin) ----
    ax.plot(0.0, 0.0, marker="*", color="#f1c40f",
            markersize=15, markeredgecolor="black", markeredgewidth=0.6,
            zorder=10)

    # ---- Frame, axes, title ----
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", "box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    subtitle = _criterion_subtitle(data)
    ax.set_title(f"{title}\n{subtitle}", linespacing=1.25)

    # ---- Inset annotation: elapsed + leaf summary ----
    timing = data.get("timing", {}).get("elapsed_ms")
    leaves = _leaves_summary(polygons)
    lines = []
    if timing is not None:
        lines.append(f"elapsed: {timing/1e3:.2f} s")
    if leaves:
        lines.append(leaves)
    if lines:
        ax.text(0.025, 0.975, "\n".join(lines), transform=ax.transAxes,
                ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor="white", edgecolor="#d0d0d0",
                          linewidth=0.6, alpha=0.92))


# ---- Main ------------------------------------------------------------------
def main() -> None:
    loaded = [(m, load(m)) for m in METHODS]
    loaded = [(m, d) for m, d in loaded if d is not None]
    if not loaded:
        print("No JSON outputs found. Run two_body_taylor / two_body_ads / two_body_loads first.")
        return

    # Shared colour normalisation across panels (t in [0, t_final]).
    t_final = max(d["config"]["t_final"] for _, d in loaded)
    cmap    = plt.cm.plasma
    norm    = Normalize(vmin=0.0, vmax=t_final)

    xlim, ylim = panel_xy_limits(*(d for _, d in loaded))

    n_panels = len(loaded)
    fig      = plt.figure(figsize=(5.6 * n_panels, 6.2),
                          constrained_layout=True)
    gs       = fig.add_gridspec(2, n_panels, height_ratios=[1.0, 0.05])

    for col, (m, d) in enumerate(loaded):
        ax = fig.add_subplot(gs[0, col])
        draw_panel(ax, d, xlim=xlim, ylim=ylim, title=TITLES[m],
                   cmap=cmap, norm=norm)

    # ---- Shared horizontal colour bar ----
    cbar_ax = fig.add_subplot(gs[1, :])
    sm      = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar    = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"snapshot time $t$  (one orbital period at $t = 2\pi$)")
    cbar.outline.set_linewidth(0.6)

    fig.suptitle(
        r"IC box pushed forward in time — Kepler orbit ($a = 1$, $e = 0.5$)",
        y=1.02,
    )

    out_path = HERE / "two_body_box_evolution.png"
    fig.savefig(out_path)
    print(f"wrote {out_path.name}")

    # ---- Terminal summary ----
    print()
    print(f"  {'method':<8}  {'elapsed':>9}   {'snaps':>5}   {'leaves (per snap)':<32}")
    print(f"  {'-'*8:<8}  {'-'*9:>9}   {'-'*5:>5}   {'-'*32:<32}")
    for m, d in loaded:
        elapsed = d.get("timing", {}).get("elapsed_ms", float("nan")) / 1e3
        polys   = d["polygons"]
        if "leaves" in polys[0]:
            leaves = [len(s["leaves"]) for s in polys]
        else:
            leaves = [1] * len(polys)
        print(f"  {m:<8}  {elapsed:>7.2f} s   {len(polys):>5}   {str(leaves):<32}")


if __name__ == "__main__":
    main()
