# File: scenes_index.py
# Purpose: Pick each scenario's default "test scene" seed for the 3D viewer: the first dev
#          seed (0-19) whose run fits one consistent physical scene.
# Contributors: Unassigned
"""Some seeds draw a lead-in walk that crosses the drop edge before a backward fall (the
simulator draws the walk and the fall direction independently). Those still play, with a
note, but they make a confusing first impression, so the viewer opens on the first dev seed
without such a note. Dev seeds only: test seeds (1,000,000+) stay held out.
"""

from __future__ import annotations

import threading
from functools import partial

import numpy as np

from ..scenarios import build_run
from ..spine import random_mounts, strip_kinematics
from .recorder import RecordingBuilder
from .scene import Ctx, build_scene

DEV_SEEDS = range(0, 20)
_cache: dict[str, int] = {}
_lock = threading.Lock()


def scene_is_consistent(scenario: str, seed: int, obs_s: float = 20.0) -> bool:
    sink: list = []
    run = build_run(scenario, seed, obs_s=obs_s, builder=partial(RecordingBuilder, sink=sink))
    kin = strip_kinematics(run.track, run.worker, random_mounts(run.worker, np.random.default_rng(seed + 3)))
    scene = build_scene(Ctx(run, sink[0], kin), scenario)
    return not scene.notes


def default_seed(scenario: str) -> int:
    with _lock:
        if scenario in _cache:
            return _cache[scenario]
    chosen = 0
    for seed in DEV_SEEDS:
        if scene_is_consistent(scenario, seed):
            chosen = seed
            break
    with _lock:
        _cache[scenario] = chosen
    return chosen
