# File: arbiter.py
# Purpose: Arbiter (DETECTION_ARCHITECTURE.md 3.7): merges the impact, perturbation, topple and
#          man-down paths so one physical event gives one episode and at most one alert per kind.
# Contributors: Unassigned
"""Rules
  * Triggers within suspect_merge_s give at most one suspected_fall.
  * The first fall_confirmed (C1 falldet or C3 topple) opens an episode; later confirmations of
    the same event merge into it (merged_from) and emit nothing.
  * man_down_warning / man_down_critical: each at most once per episode; with no open episode
    they open one.
  * A dismissal (supervisor button) covers only the alert it answered: a later man_down_* of
    the same episode still alerts, and a new T1/T3 trigger opens a new episode. A dismissed
    episode closes close_after_dismiss_s after the last activity.
  * cleared: upright for clear_upright_s AND gait, no earlier than clear_after_s after the
    episode's trigger (upright alone would clear a harness-suspended worker).
  * recovered is emitted only when no episode is open (timeline information).

Times are the hub's internal seconds; MainModel converts them to device time when emitting.
Output: proto-events {t, kind, confidence, episode_id, trigger_t, path, rule, merged_from,
tags, evidence}.
"""


class Arbiter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.episode = None
        self.next_id = 1
        self.last_suspect_t = -1e18
        self.closed = []          # ids of closed episodes (bounded)

    # ------------------------------------------------------------------ helpers
    def _open(self, t, path, trigger_t):
        self.episode = {"id": self.next_id, "opened_t": t, "trigger_t": trigger_t, "paths": [path],
                        "alerts": set(), "dismissed_t": None, "last_t": t}
        self.next_id += 1
        return self.episode

    def _close(self):
        if self.episode is not None:
            self.closed = (self.closed + [self.episode["id"]])[-20:]
        self.episode = None

    @staticmethod
    def _proto(t, kind, confidence, episode, path, rule, trigger_t, tags=(), evidence=None):
        return {"t": t, "kind": kind, "confidence": confidence,
                "episode_id": None if episode is None else episode["id"], "trigger_t": trigger_t,
                "path": path, "rule": rule, "merged_from": list(episode["paths"]) if episode else [path],
                "tags": set(tags), "evidence": evidence or {}}

    def _same_event(self, trigger_t):
        """Does a trigger at trigger_t belong to the open episode?"""
        ep = self.episode
        if ep is None:
            return False
        if ep["dismissed_t"] is not None and trigger_t is not None and trigger_t > ep["dismissed_t"]:
            return False
        return True

    # ------------------------------------------------------------------ inputs
    def suspected(self, t, path, trigger_t, evidence=None):
        ep = self.episode
        if ep is not None and ep["dismissed_t"] is not None and trigger_t > ep["dismissed_t"]:
            self._close()                         # a new trigger after a dismissal: new episode
        out = []
        if t - self.last_suspect_t > self.cfg["suspect_merge_s"]:
            out.append(self._proto(t, "suspected_fall", 0.5, self.episode, path,
                                   "stage1" if path in ("impact", "perturbation") else "topple", trigger_t,
                                   evidence=evidence))
        self.last_suspect_t = t
        return out

    def confirmed(self, t, path, rule, trigger_t, confidence, tags=(), evidence=None):
        if self._same_event(trigger_t):
            ep = self.episode
            if path not in ep["paths"]:
                ep["paths"].append(path)
            ep["last_t"] = t
            if "fall_confirmed" in ep["alerts"]:
                return []
            ep["alerts"].add("fall_confirmed")
            return [self._proto(t, "fall_confirmed", confidence, ep, path, rule, trigger_t, tags, evidence)]
        self._close()
        ep = self._open(t, path, trigger_t)
        ep["alerts"].add("fall_confirmed")
        return [self._proto(t, "fall_confirmed", confidence, ep, path, rule, trigger_t, tags, evidence)]

    def recovered(self, t, path, rule, trigger_t, confidence, tags=(), evidence=None):
        if self.episode is not None:
            return []
        return [self._proto(t, "recovered", confidence, None, path, rule, trigger_t, tags, evidence)]

    def near_fall(self, t, path, rule, trigger_t, confidence, evidence=None):
        return [self._proto(t, "near_fall", confidence, self.episode, path, rule, trigger_t, evidence=evidence)]

    def man_down(self, t, kind, rule, trigger_t, tags=(), evidence=None):
        ep = self.episode
        if ep is None:
            ep = self._open(t, "man_down", trigger_t)
        if "man_down" not in ep["paths"]:
            ep["paths"].append("man_down")
        ep["last_t"] = t
        if kind in ep["alerts"]:
            return []
        ep["alerts"].add(kind)
        confidence = 0.9 if kind == "man_down_critical" else 0.8
        return [self._proto(t, kind, confidence, ep, "man_down", rule, trigger_t, tags, evidence)]

    def dismiss(self, t, episode_id=None):
        """Supervisor dismissed the open episode's alert. Returns True if an episode was dismissed."""
        ep = self.episode
        if ep is None or (episode_id is not None and ep["id"] != episode_id):
            return False
        ep["dismissed_t"] = t
        ep["last_t"] = t
        return True

    def on_sample(self, t, track):
        ep = self.episode
        if ep is None:
            return []
        cfg = self.cfg
        if ep["dismissed_t"] is not None and t - ep["last_t"] >= cfg["close_after_dismiss_s"]:
            self._close()
            return []
        start = ep["trigger_t"] if ep["trigger_t"] is not None else ep["opened_t"]
        if ep["alerts"] and t >= start + cfg["clear_after_s"] and track.upright_for(t) >= cfg["clear_upright_s"] \
                and track.gait(t):
            out = [self._proto(t, "cleared", 0.8, ep, ep["paths"][0], "upright_and_gait", ep["trigger_t"])]
            self._close()
            return out
        return []
