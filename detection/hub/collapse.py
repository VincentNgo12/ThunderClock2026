# File: collapse.py
# Purpose: C3 collapse confirmer (DETECTION_ARCHITECTURE.md 3.5). After a T3 topple trigger it
#          decides fall_confirmed or recovered from transparent stillness/posture rules that
#          tolerate brief movement. Thresholds: config.json "collapse".
# Contributors: Unassigned
"""Window [T3 + start_s, T3 + end_s], in one-second PostureTrack bins:

  fast   the first fast_still_s after T3 + start_s: still fraction >= fast_still_frac and
         "down" fraction >= down_frac                         -> fall_confirmed, topple_still_down
  slow   any 10 s window ending by T3 + end_s: still >= slow_still_frac and down >= down_frac
                                                              -> fall_confirmed, topple_mostly_still
  else   back upright (>= 3 anchors < upright_deg for upright_hold_s), or T3 + end_s reached
                                                              -> recovered, moving_or_up
"Not worn" (no motion at all) never rejects: it counts as still and only adds the tag no_motion.
Windows with less than min_coverage of their bins observed are skipped (tag gap_in_window).
"""
import math


class CollapseConfirmer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pending = None

    def reset(self):
        self.pending = None

    def open(self, t3, evidence=None):
        """Start a candidate at trigger time t3 (ignored while one is pending)."""
        if self.pending is None:
            self.pending = {"t3": t3, "evidence": dict(evidence or {}), "fast_done": False, "tags": set()}

    def _decide(self, t, kind, rule, stats, confidence):
        p = self.pending
        self.pending = None
        tags = set(p["tags"])
        evidence = dict(p["evidence"])
        if stats is not None:
            evidence.update(still_frac=round(stats["still_frac"], 3), down_frac=round(stats["down_frac"], 3),
                            incl_median_deg=stats["incl_median_deg"])
            if stats["no_motion"]:
                tags.add("no_motion")
        return {"t": t, "kind": kind, "rule": rule, "trigger_t": p["t3"], "tags": tags,
                "evidence": evidence, "confidence": confidence}

    def on_sample(self, t, track):
        """Per sample: return to upright ends the candidate early."""
        p = self.pending
        if p is None:
            return None
        hold = self.cfg["upright_hold_s"]
        if track.upright_since is not None and track.upright_since >= p["t3"] and track.upright_for(t) >= hold:
            p["tags"].add("ended_upright")
            return self._decide(t, "recovered", "moving_or_up", None, 0.6)
        return None

    def on_bin(self, closed_bin, track, t_now):
        """After each closed one-second bin."""
        p = self.pending
        if p is None:
            return None
        cfg = self.cfg
        start = math.ceil(p["t3"] + cfg["start_s"] - 1e-9)
        done_s = closed_bin.end - start
        if done_s <= 0:
            return None
        n_fast = int(round(cfg["fast_still_s"]))
        n_slow = int(round(cfg["slow_window_s"]))
        if not p["fast_done"] and done_s >= n_fast:
            p["fast_done"] = True
            stats = track.window(n_fast, since=start)
            if stats["coverage"] < 1.0:
                p["tags"].add("gap_in_window")
            if stats["coverage"] >= cfg["min_coverage"] and stats["still_frac"] >= cfg["fast_still_frac"] \
                    and stats["down_frac"] >= cfg["down_frac"]:
                return self._decide(t_now, "fall_confirmed", "topple_still_down", stats, 0.8)
        if done_s >= n_slow:
            stats = track.window(n_slow, since=start)
            if stats["coverage"] < 1.0:
                p["tags"].add("gap_in_window")
            if stats["coverage"] >= cfg["min_coverage"] and stats["still_frac"] >= cfg["slow_still_frac"] \
                    and stats["down_frac"] >= cfg["down_frac"]:
                return self._decide(t_now, "fall_confirmed", "topple_mostly_still", stats, 0.7)
        if closed_bin.end >= p["t3"] + cfg["end_s"]:
            return self._decide(t_now, "recovered", "moving_or_up", track.window(n_slow), 0.6)
        return None
