#!/usr/bin/env python3
"""
examples/two_body/plot.py

Plot the output of the three two-body examples. Run from the directory
where the example executables wrote their CSV files:

    cd /tmp/two_body_run
    /path/to/build/examples/two_body_simple_taylor
    /path/to/build/examples/two_body_ads
    /path/to/build/examples/two_body_loads
    python3 /path/to/tax/examples/two_body/plot.py

Produces three figures:

  Figure 1 — orbits in the x-y plane for simple Taylor, ADS, LOADS.
  Figure 2 — number of active boxes vs time for ADS and LOADS.
  Figure 3 — IC-space subdivisions (projected to dim 0 vs dim 1) for
             both ADS and LOADS, color-coded by depth.

Skips any panel whose CSVs are missing, so you can run any subset.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

HERE = Path.cwd()


def read_timing(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def load_traj(prefix: str) -> pd.DataFrame | None:
    p = HERE / f"{prefix}_traj.csv"
    return pd.read_csv(p) if p.exists() else None


def load_tree(prefix: str) -> pd.DataFrame | None:
    p = HERE / f"{prefix}_tree.csv"
    return pd.read_csv(p) if p.exists() else None


def load_boxcount(prefix: str) -> pd.DataFrame | None:
    p = HERE / f"{prefix}_boxcount.csv"
    return pd.read_csv(p) if p.exists() else None


def load_distribution() -> pd.DataFrame | None:
    p = HERE / "two_body_distribution.csv"
    return pd.read_csv(p) if p.exists() else None


# ----- Figure 1: orbit comparison -----------------------------------------
def plot_orbits():
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plotted = False
    for prefix, color, label in [
        ("simple_taylor", "tab:blue",   "simple Taylor"),
        ("ads",           "tab:orange", "ADS centerpoint"),
        ("loads",         "tab:green",  "LOADS centerpoint"),
    ]:
        df = load_traj(prefix)
        if df is None:
            continue
        ax.plot(df["x0"], df["x1"], color=color, label=label, lw=1.2)
        plotted = True
    if not plotted:
        print("No trajectory CSVs found; skipping orbit figure.")
        plt.close(fig)
        return
    ax.plot(0, 0, "k*", markersize=12, label="primary")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", "box")
    ax.legend(loc="upper right")
    ax.set_title("Two-body orbit (snapshots)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("two_body_orbits.png", dpi=140)
    print("wrote two_body_orbits.png")


# ----- Figure 2: box count vs time ----------------------------------------
def plot_box_count():
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plotted = False
    for prefix, color, label in [
        ("ads",   "tab:orange", "ADS (truncation)"),
        ("loads", "tab:green",  "LOADS (NLI)"),
    ]:
        df = load_boxcount(prefix)
        if df is None:
            continue
        ax.step(df["t"], df["n_alive"], where="post", color=color, label=label, lw=1.3)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("t")
    ax.set_ylabel("number of active boxes")
    ax.set_title("Boxes alive over time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("two_body_box_count.png", dpi=140)
    print("wrote two_body_box_count.png")


# ----- Figure 3: IC-space subdivisions (projection cx0 vs cx1) ------------
def plot_subdivisions():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=True)
    plotted = False
    for ax, prefix, title in [
        (axes[0], "ads",   "ADS (truncation)"),
        (axes[1], "loads", "LOADS (NLI)"),
    ]:
        df = load_tree(prefix)
        if df is None:
            ax.set_visible(False)
            continue
        # Only show "done" leaves — they form the final partition.
        done = df[df["done"] == 1]
        if done.empty:
            ax.text(0.5, 0.5, "no done leaves", ha="center", va="center", transform=ax.transAxes)
            continue
        depth_max = int(done["depth"].max())
        cmap = plt.cm.viridis
        norm = plt.Normalize(vmin=0, vmax=max(depth_max, 1))
        for _, r in done.iterrows():
            cx, cy = r["cx0"], r["cx1"]
            hx, hy = r["hw0"], r["hw1"]
            rect = patches.Rectangle(
                (cx - hx, cy - hy), 2 * hx, 2 * hy,
                fill=True, alpha=0.4,
                facecolor=cmap(norm(r["depth"])),
                edgecolor="black", linewidth=0.4,
            )
            ax.add_patch(rect)
        ax.relim()
        ax.autoscale_view()
        ax.set_xlabel("IC component 0 (x)")
        ax.set_ylabel("IC component 1 (y)")
        ax.set_title(f"{title} — {len(done)} done leaves, max depth {depth_max}")
        ax.set_aspect("equal", "box")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="depth")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    fig.tight_layout()
    fig.savefig("two_body_subdivisions.png", dpi=140)
    print("wrote two_body_subdivisions.png")


# ----- Summary -----------------------------------------------------------
def print_timing_summary():
    print()
    print("=" * 60)
    print(f"{'method':<18} {'elapsed_ms':>12} {'extra':>20}")
    print("-" * 60)
    for prefix, label in [
        ("simple_taylor", "simple Taylor"),
        ("ads",           "ADS"),
        ("loads",         "LOADS"),
    ]:
        t = read_timing(HERE / f"{prefix}_timing.txt")
        if not t:
            continue
        elapsed = t.get("elapsed_ms", "—")
        if "n_done" in t:
            extra = f"done={t['n_done']} depth={t.get('max_depth','?')}"
        else:
            extra = f"steps={t.get('n_steps','?')}"
        print(f"{label:<18} {elapsed:>12} {extra:>20}")


# ----- Figure 4: distribution snapshots in the (x, y) plane ----------------
def plot_distribution_snapshots():
    df = load_distribution()
    if df is None:
        return
    # Overlay the reference orbit (from simple_taylor or ads).
    ref = None
    for prefix in ("simple_taylor", "ads", "loads"):
        df_ref = load_traj(prefix)
        if df_ref is not None:
            ref = df_ref
            break

    times = sorted(df["t"].unique())
    n = len(times)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for i, t in enumerate(times):
        ax = axes[i // cols, i % cols]
        snap = df[df["t"] == t]
        if ref is not None:
            ax.plot(ref["x0"], ref["x1"], color="gray", lw=0.6, alpha=0.5, zorder=0)
        ax.scatter(snap["x0"], snap["x1"], s=10, c="tab:blue", alpha=0.6, edgecolor="none")
        ax.plot(0, 0, "k*", markersize=10, zorder=3)
        ax.set_aspect("equal", "box")
        ax.set_title(f"t = {t:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].set_visible(False)
    fig.suptitle(f"State distribution snapshots ({len(df['sample'].unique())} samples in IC box)")
    fig.tight_layout()
    fig.savefig("two_body_distribution.png", dpi=140)
    print("wrote two_body_distribution.png")


if __name__ == "__main__":
    plot_orbits()
    plot_box_count()
    plot_subdivisions()
    plot_distribution_snapshots()
    print_timing_summary()
