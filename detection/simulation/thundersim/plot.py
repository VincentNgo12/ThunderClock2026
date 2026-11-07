# File: plot.py
# Purpose: Debug figure for one run: spine curve snapshots (Hermite interpolation through the
#          four anchors) and per-anchor signals with ground-truth and model events.
# Contributors: Unassigned

from __future__ import annotations

from pathlib import Path

import numpy as np

from .model_api import GRAVITY
from .scenarios import build_run
from .spine import PHASES, random_mounts, spine_curve, strip_kinematics, virtual_imu


def plot_run(scenario_id: str, seed: int, out: Path, events: list | None = None, obs_s: float = 60.0) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = build_run(scenario_id, seed, obs_s=obs_s)
    kin = strip_kinematics(run.track, run.worker, random_mounts(run.worker, np.random.default_rng(seed + 3)))
    acc, gyro = virtual_imu(kin, run.track.fs)
    t = run.track.t
    fig = plt.figure(figsize=(13, 8.5))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.3, 1, 1])
    # Spine snapshots at ground-truth moments (side view: horizontal distance vs height).
    moments = [("calibration", 2.0)] + [(n, tt) for tt, n, _ in run.events][:2] + [("end", t[-1] - 1)]
    for col, (label, tt) in enumerate(moments[:4]):
        ax = fig.add_subplot(grid[0, col])
        i = int(np.clip(tt * run.track.fs, 0, len(t) - 1))
        anchors = kin.anchors[i]
        axes = np.stack([kin.seg_rot[s][i].apply([0, 1, 0]) for s in range(4)])
        curve = spine_curve(anchors, axes)
        horiz = np.linalg.norm((curve - anchors[0])[:, [0, 2]], axis=1) * np.sign((curve - anchors[0])[:, 2] + 1e-9)
        ax.plot(horiz, curve[:, 1], "-", color="#555")
        ah = np.linalg.norm((anchors - anchors[0])[:, [0, 2]], axis=1) * np.sign((anchors - anchors[0])[:, 2] + 1e-9)
        ax.scatter(ah, anchors[:, 1], c=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"], zorder=3)
        ax.axhline(0, color="#aaa", lw=0.8)
        ax.set_title(f"{label} @ {tt:.1f} s", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=7)
    names = ["pelvis", "lumbar", "thoracic", "upper thoracic"]
    ax1 = fig.add_subplot(grid[1, :])
    for s in range(4):
        ax1.plot(t, np.linalg.norm(acc[:, s], axis=1) / GRAVITY, lw=0.6, label=names[s])
    ax1.set_ylabel("|a| (g), true, before device")
    ax1.legend(fontsize=7, ncol=4, loc="upper right")
    ax2 = fig.add_subplot(grid[2, :], sharex=ax1)
    for s in range(4):
        ax2.plot(t, np.linalg.norm(gyro[:, s], axis=1), lw=0.6)
    ax2.set_ylabel("|gyro| (rad/s)")
    ax2.set_xlabel("time (s)")
    for ax in (ax1, ax2):
        for tt, n, _ in run.events:
            ax.axvline(tt, color="k", ls=":", lw=0.8)
        for e in events or []:
            if e["kind"] != "suspected_fall":
                ax.axvline(e["t"], color="r", ls="--", lw=0.9)
    ax1.set_title(f"{scenario_id} seed {seed} - expected {run.expected}; dotted = ground truth, red = model events "
                  f"({', '.join(PHASES[p] for p in np.unique(run.track.phase))})", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
