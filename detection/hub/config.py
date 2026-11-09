# File: config.py
# Purpose: Load the hub thresholds from config.json, apply overrides, and hash the result
#          (the hash goes into the seed ledger and every event's data block).
# Contributors: Unassigned

import copy
import hashlib
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None, overrides=None):
    """Return the config dict. overrides: nested dict, e.g. {"man_down": {"enabled": False}}."""
    with open(path or CONFIG_PATH) as handle:
        config = json.load(handle)
    if overrides:
        config = _merge(copy.deepcopy(config), overrides)
    return config


def config_hash(config):
    """Short sha256 of the thresholds (comments excluded), stable across key order."""
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    text = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:12]
