# File: bodies/__init__.py
# Purpose: Worker body presets measured from parametric human models (currently Anny).
"""Body presets for ThunderSim's Worker. See README.md and NOTICE in this folder."""

from .anny import BODIES_DIR, available_bodies, worker_from_anny

__all__ = ["BODIES_DIR", "available_bodies", "worker_from_anny"]
