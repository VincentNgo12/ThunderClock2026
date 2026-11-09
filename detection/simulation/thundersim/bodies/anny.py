# File: bodies/anny.py
# Purpose: Turn an Anny body JSON (written offline by tools/anny_body.py) into Worker(**kwargs).
"""Pure standard library; does not import anny. Usage:

    from thundersim.bodies import worker_from_anny
    from thundersim.spine import Worker
    worker = Worker(**worker_from_anny("avg_male", load_kg=8.0))

`name_or_path` is a preset name ("avg_male", "female_160") or a path to a JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path

BODIES_DIR = Path(__file__).resolve().parent
WORKER_KEYS = ("height_m", "mass_kg", "load_kg", "spacing_m", "neutral_curvature_deg",
               "mount_error_deg", "sensor_offset_m")
# Optional Worker fields (None = the sim's old formulas); filled only with fall_geometry=True.
OPTIONAL_KEYS = ("pelvis_height_m", "com_to_pelvis_m")


def available_bodies() -> list[str]:
    """Preset names shipped in this folder (anny_<name>.json)."""
    return sorted(p.stem[len("anny_"):] for p in BODIES_DIR.glob("anny_*.json"))


def _resolve(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    if path.suffix == ".json":
        return path
    return BODIES_DIR / f"anny_{name_or_path}.json"


def worker_from_anny(name_or_path: str | Path, fall_geometry: bool = False, **overrides) -> dict:
    """Worker keyword arguments from an Anny body JSON (format anny-worker-body/1).

    spacing_m: Anny spinal-midline chords S1-L3, L3-T10, T10-T2 (m).
    neutral_curvature_deg: sensor pitch at S1/L3/T10/T2 from Anny's skin normals.
    height_m, mass_kg: measured on the Anny mesh (mass uses Anny's 980 kg/m3).
    load_kg, mount_error_deg, sensor_offset_m: sim defaults, not body properties.
    fall_geometry=True also sets the optional Worker fields from Anny's rest pose:
      pelvis_height_m = S1 spinal-midline height (the sim's pelvis anchor, root_pos);
      com_to_pelvis_m = whole-body centre of mass -> that anchor (+X right, +Y up, +Z posterior).
    It is off by default: the realism scorecard found these do not close the fall-descent gap.
    Overrides win, e.g. load_kg=12.
    """
    with open(_resolve(name_or_path)) as f:
        doc = json.load(f)
    if doc.get("format") != "anny-worker-body/1":
        raise ValueError(f"unexpected format {doc.get('format')!r}")
    fields = dict(doc["worker_fields"])
    if fall_geometry:
        props = doc["worker_properties_not_settable"]
        fields["pelvis_height_m"] = props["pelvis_height"]["anny_value_S1_midline_y"]
        fields["com_to_pelvis_m"] = props["falls_r_p_com_to_pelvis_anchor"]["anny_com_to_S1_midline"]
    fields.update(overrides)
    unknown = set(fields) - set(WORKER_KEYS) - set(OPTIONAL_KEYS)
    if unknown:
        raise KeyError(f"not Worker fields: {sorted(unknown)}")
    kwargs = {key: float(fields[key]) for key in ("height_m", "mass_kg", "load_kg", "mount_error_deg", "sensor_offset_m")}
    spacing = fields["spacing_m"]
    kwargs["spacing_m"] = None if spacing is None else tuple(float(x) for x in spacing)
    kwargs["neutral_curvature_deg"] = tuple(float(x) for x in fields["neutral_curvature_deg"])
    if kwargs["spacing_m"] is not None and len(kwargs["spacing_m"]) != 3:
        raise ValueError("spacing_m needs 3 values")
    if len(kwargs["neutral_curvature_deg"]) != 4:
        raise ValueError("neutral_curvature_deg needs 4 values")
    if fields.get("pelvis_height_m") is not None:
        kwargs["pelvis_height_m"] = float(fields["pelvis_height_m"])
    if fields.get("com_to_pelvis_m") is not None:
        kwargs["com_to_pelvis_m"] = tuple(float(x) for x in fields["com_to_pelvis_m"])
    return kwargs
