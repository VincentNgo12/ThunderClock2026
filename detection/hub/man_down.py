# File: man_down.py
# Purpose: T4 gated man-down timer (DETECTION_ARCHITECTURE.md 3.6). Armed by a T1/T3 trigger,
#          an open episode or a restart near a trigger; escalates a worker who stays down and
#          mostly still. Also the lying_work timer. Thresholds: config.json "man_down".
# Contributors: Unassigned
"""Gated timer
    armed by the trigger; disarmed once the worker has been upright (>= 3 anchors < upright_deg)
    for upright_hold_s (disarm_on_upright, the default since collapse_v1). With disarm_on_upright
    off: armed for >= arm_s, then until upright. A worker who never stands up stays armed.
    Each closed one-second bin counts as "down and mostly still" when its down fraction is
    >= down_frac and the still fraction of the last window_s (bins since arming) is >= still_frac.
    The run of such bins starts the clock: man_down_warning at warning_s, man_down_critical at
    critical_s. A short movement inside a mostly-still window does not reset it; standing up does,
    and so does a bin that is clearly not down. Empty bins (data gap) neither start nor reset it.
    "Not worn" after a trigger escalates (rule no_motion_after_trigger): apnea or a supine body
    damping the strip looks the same as a strip on a table.

lying_work (worker-declared, expires after lying_work_expiry_s)
    no trigger needed: down and fully still (bin fraction of breathing-only "quiet" samples, or of
    "still" samples with lying_work_quiet off, >= lying_work_still_frac) for
    lying_work_warning_s -> man_down_warning, lying_work_critical_s -> man_down_critical.
"""


class ManDownTimer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.armed = False
        self.armed_until = -1e18
        self.arm_t = None
        self.arm_reason = None
        self.trigger_t = None
        self.recovered = False
        self.down_start = None
        self.emitted = set()
        self.lw_start = None
        self.lw_emitted = set()
        self.lying_work_since = None

    def arm(self, t, reason, trigger_t=None):
        if not self.armed:
            self.down_start = None
            self.emitted = set()
            self.arm_t = t
        self.armed = True
        self.recovered = False
        self.arm_reason = reason
        self.trigger_t = t if trigger_t is None else trigger_t
        self.armed_until = max(self.armed_until, t + self.cfg["arm_s"])

    def on_sample(self, t, track):
        if not self.armed:
            return
        if track.upright_for(t) >= self.cfg["upright_hold_s"]:
            self.recovered = True
            self.down_start = None
        # disarm_on_upright: back upright (upright_hold_s) cancels arming at once; off = keep >= arm_s.
        if self.recovered and (t >= self.armed_until or self.cfg["disarm_on_upright"]):
            self.armed = False

    def lying_work_active(self, t, work_mode):
        if work_mode != "lying_work":
            self.lying_work_since = None
            return False
        if self.lying_work_since is None:
            self.lying_work_since = t
        return t - self.lying_work_since <= self.cfg["lying_work_expiry_s"]

    def on_bin(self, closed_bin, track, t_now, work_mode="normal"):
        """Returns a list of decisions {kind, rule, t, trigger_t, tags, evidence}."""
        cfg = self.cfg
        out = []
        b = closed_bin
        bin_down = (b.n_down / b.n) if b.n else None
        if self.armed and b.n:
            n_window = int(min(cfg["window_s"], max(1.0, b.end - self.arm_t)))
            stats = track.window(n_window)
            if bin_down >= cfg["down_frac"] and stats["still_frac"] >= cfg["still_frac"]:
                if self.down_start is None:
                    self.down_start = b.start
                elapsed = b.end - self.down_start
                for kind, limit in (("man_down_warning", cfg["warning_s"]), ("man_down_critical", cfg["critical_s"])):
                    if elapsed >= limit and kind not in self.emitted:
                        self.emitted.add(kind)
                        rule = "no_motion_after_trigger" if stats["no_motion"] else "down_mostly_still"
                        out.append({"kind": kind, "rule": rule, "t": t_now, "trigger_t": self.trigger_t,
                                    "tags": {"no_motion"} if stats["no_motion"] else set(),
                                    "evidence": {"down_s": round(elapsed, 1), "still_frac": round(stats["still_frac"], 3),
                                                 "incl_median_deg": stats["incl_median_deg"],
                                                 "armed_by": self.arm_reason}})
            elif bin_down < cfg["down_frac"]:
                self.down_start = None
        if self.lying_work_active(t_now, work_mode) and b.n:
            calm = b.n_quiet if cfg["lying_work_quiet"] else b.n_still   # quiet = breathing-only stillness
            if bin_down >= cfg["down_frac"] and calm / b.n >= cfg["lying_work_still_frac"]:
                if self.lw_start is None:
                    self.lw_start = b.start
                elapsed = b.end - self.lw_start
                for kind, limit in (("man_down_warning", cfg["lying_work_warning_s"]),
                                    ("man_down_critical", cfg["lying_work_critical_s"])):
                    if elapsed >= limit and kind not in self.lw_emitted:
                        self.lw_emitted.add(kind)
                        out.append({"kind": kind, "rule": "lying_work_still", "t": t_now, "trigger_t": self.lw_start,
                                    "tags": {"lying_work"}, "evidence": {"still_s": round(elapsed, 1)}})
            else:
                self.lw_start = None
                self.lw_emitted = set()
        elif work_mode != "lying_work":
            self.lw_start = None
            self.lw_emitted = set()
        return out
