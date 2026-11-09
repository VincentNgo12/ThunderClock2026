// File: charts.js
// Purpose: Timeline (phases, activities, ground-truth and model events, playhead) and the
//          live sensor charts (|a| in g and |omega| in deg/s per anchor, as delivered).
import { SLOT_COLORS, PHASE_COLORS } from "./util.js";

const MODEL_COLORS = {
  suspected_fall: "#ffb020", near_fall: "#ffd23f", recovered: "#3ccf7a", cleared: "#3ccf7a",
  fall_confirmed: "#ff4d4d", man_down_warning: "#ff4dd2", man_down_critical: "#ff00aa",
};

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight || canvas.height;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const g = canvas.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, w, h };
}

export class Timeline {
  constructor(canvas, onSeek) {
    this.canvas = canvas;
    this.onSeek = onSeek;
    this.hover = null;
    let dragging = false;
    const seek = (e) => {
      const r = canvas.getBoundingClientRect();
      const x = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
      if (this.payload) this.onSeek(x * this.payload.meta.duration);
    };
    canvas.addEventListener("pointerdown", (e) => { dragging = true; canvas.setPointerCapture(e.pointerId); seek(e); });
    canvas.addEventListener("pointermove", (e) => { if (dragging) seek(e); });
    canvas.addEventListener("pointerup", () => { dragging = false; });
  }

  set(payload) {
    this.payload = payload;
    this.static = null;
  }

  drawStatic(w, h) {
    const P = this.payload;
    const off = document.createElement("canvas");
    const dpr = window.devicePixelRatio || 1;
    off.width = w * dpr; off.height = h * dpr;
    const g = off.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const T = P.meta.duration;
    const X = (t) => (t / T) * w;
    g.fillStyle = "#0b0e12"; g.fillRect(0, 0, w, h);
    // phases band
    const names = P.meta.phase_names;
    for (const [a, b, code] of P.phases) {
      g.fillStyle = PHASE_COLORS[names[code]] || "#555";
      g.fillRect(X(a), 0, Math.max(1, X(b) - X(a)), 12);
    }
    // activities band
    g.font = "10.5px ui-sans-serif, -apple-system, sans-serif";
    g.textBaseline = "middle";
    P.activities.forEach(([a, b, label], i) => {
      const x0 = X(a), x1 = X(b);
      g.fillStyle = i % 2 ? "#1d2530" : "#18202a";
      g.fillRect(x0, 14, Math.max(1, x1 - x0), 16);
      if (x1 - x0 > 40) {
        g.fillStyle = "#aab4c0";
        g.save(); g.beginPath(); g.rect(x0 + 2, 14, x1 - x0 - 4, 16); g.clip();
        g.fillText(label, x0 + 4, 22.5); g.restore();
      }
    });
    // ground truth (white diamonds) and model events (coloured triangles)
    for (const e of P.truth) {
      const x = X(e.t);
      g.fillStyle = "#e6e9ee";
      g.beginPath(); g.moveTo(x, 33); g.lineTo(x + 4, 37); g.lineTo(x, 41); g.lineTo(x - 4, 37); g.fill();
    }
    for (const e of P.model_events) {
      const x = X(e.t);
      g.fillStyle = MODEL_COLORS[e.kind] || "#aaa";
      g.beginPath(); g.moveTo(x, 44); g.lineTo(x + 5, 54); g.lineTo(x - 5, 54); g.fill();
    }
    if (P.meta.ref_time !== null && P.meta.ref_time !== undefined) {
      g.strokeStyle = "rgba(255,255,255,.35)"; g.setLineDash([2, 3]);
      g.beginPath(); g.moveTo(X(P.meta.ref_time), 0); g.lineTo(X(P.meta.ref_time), h); g.stroke(); g.setLineDash([]);
    }
    // seconds ticks
    g.fillStyle = "rgba(255,255,255,.25)";
    const step = T > 120 ? 20 : T > 60 ? 10 : 5;
    for (let t = 0; t <= T; t += step) g.fillRect(X(t), h - 3, 1, 3);
    this.static = off;
    this.staticSize = [w, h];
  }

  draw(t) {
    if (!this.payload) return;
    const { g, w, h } = fitCanvas(this.canvas);
    if (!this.static || this.staticSize[0] !== w || this.staticSize[1] !== h) this.drawStatic(w, h);
    g.drawImage(this.static, 0, 0, w, h);
    const x = (t / this.payload.meta.duration) * w;
    g.fillStyle = "#ff8a1f";
    g.fillRect(x - 1, 0, 2, h);
    g.beginPath(); g.moveTo(x - 5, 0); g.lineTo(x + 5, 0); g.lineTo(x, 6); g.fill();
  }
}

export class SignalChart {
  constructor(canvas, kind) {
    this.canvas = canvas;
    this.kind = kind; // "acc" | "gyro"
  }

  set(payload) {
    this.payload = payload;
    const S = payload.signals;
    const vals = this.kind === "acc" ? S.acc_g : S.gyro_dps;
    let max = 0;
    for (const v of vals) if (v > max) max = v;
    const prof = payload.meta.profile_info;
    if (this.kind === "acc") this.ymax = Math.max(2.5, Math.min(max, prof.accel_range_g * 1.75) * 1.1);
    else this.ymax = Math.max(120, Math.min(max, prof.gyro_range_dps * 1.75) * 1.1);
    this.limit = this.kind === "acc" ? prof.accel_range_g : prof.gyro_range_dps;
  }

  draw(t, windowBefore = 8, windowAfter = 4) {
    if (!this.payload) return;
    const { g, w, h } = fitCanvas(this.canvas);
    const P = this.payload, S = P.signals;
    const vals = this.kind === "acc" ? S.acc_g : S.gyro_dps;
    const pad = { l: 34, r: 6, t: 16, b: 14 };
    const t0 = t - windowBefore, t1 = t + windowAfter;
    const X = (tt) => pad.l + ((tt - t0) / (t1 - t0)) * (w - pad.l - pad.r);
    const Y = (v) => h - pad.b - (Math.min(v, this.ymax) / this.ymax) * (h - pad.t - pad.b);
    g.clearRect(0, 0, w, h);
    g.font = "10px ui-monospace, Menlo, monospace";
    g.fillStyle = "#8b95a3";
    g.fillText(this.kind === "acc" ? "|a| (g) as delivered" : "|ω| (°/s) as delivered", pad.l, 11);
    // grid
    const ticks = this.kind === "acc" ? niceTicks(this.ymax) : niceTicks(this.ymax);
    g.strokeStyle = "rgba(255,255,255,.07)";
    for (const v of ticks) { const y = Y(v); g.beginPath(); g.moveTo(pad.l, y); g.lineTo(w - pad.r, y); g.stroke(); g.fillText(String(v), 2, y + 3); }
    if (this.kind === "acc") {
      g.strokeStyle = "rgba(255,255,255,.3)"; g.setLineDash([4, 4]);
      g.beginPath(); g.moveTo(pad.l, Y(1)); g.lineTo(w - pad.r, Y(1)); g.stroke(); g.setLineDash([]);
    }
    if (this.limit < this.ymax) {
      g.strokeStyle = "rgba(255,77,77,.6)"; g.setLineDash([6, 3]);
      g.beginPath(); g.moveTo(pad.l, Y(this.limit)); g.lineTo(w - pad.r, Y(this.limit)); g.stroke(); g.setLineDash([]);
      g.fillStyle = "rgba(255,77,77,.8)";
      g.fillText(`clip ±${this.limit}${this.kind === "acc" ? " g/axis" : " °/s/axis"}`, w - pad.r - 110, Y(this.limit) - 3);
    }
    // events
    for (const e of P.truth) if (e.t >= t0 && e.t <= t1) {
      g.strokeStyle = "rgba(230,233,238,.45)"; g.setLineDash([2, 3]);
      g.beginPath(); g.moveTo(X(e.t), pad.t); g.lineTo(X(e.t), h - pad.b); g.stroke(); g.setLineDash([]);
    }
    for (const e of P.model_events) if (e.t >= t0 && e.t <= t1) {
      g.strokeStyle = MODEL_COLORS[e.kind] || "#aaa";
      g.beginPath(); g.moveTo(X(e.t), pad.t); g.lineTo(X(e.t), h - pad.b); g.stroke();
    }
    // series
    const T = S.t;
    let i0 = lowerBound(T, t0) - 1, i1 = lowerBound(T, t1) + 1;
    i0 = Math.max(0, i0); i1 = Math.min(T.length, i1);
    const sparse = (i1 - i0) < 40;
    for (let s = 0; s < 4; s++) {
      if (P.meta.dead_slot === s) continue;
      g.strokeStyle = SLOT_COLORS[s];
      g.fillStyle = SLOT_COLORS[s];
      g.lineWidth = 1.3;
      g.beginPath();
      let pen = false;
      for (let i = i0; i < i1; i++) {
        if (!S.ok[i]) { pen = false; continue; }
        const x = X(T[i]), y = Y(vals[i * 4 + s]);
        if (sparse) { g.fillRect(x - 2, y - 2, 4, 4); }
        if (!pen) { g.moveTo(x, y); pen = true; } else g.lineTo(x, y);
      }
      g.stroke();
      // saturated samples
      g.fillStyle = "#ff3030";
      for (let i = i0; i < i1; i++) if (S.sat[i * 4 + s] && S.ok[i]) g.fillRect(X(T[i]) - 2, Y(vals[i * 4 + s]) - 2, 4, 4);
    }
    g.lineWidth = 1;
    // lost packets: tick marks along the bottom
    g.fillStyle = "rgba(255,176,32,.7)";
    for (let i = i0; i < i1; i++) if (!S.ok[i]) g.fillRect(X(T[i]), h - pad.b + 2, 1.5, 4);
    // playhead
    g.fillStyle = "#ff8a1f";
    g.fillRect(X(t) - 0.5, pad.t - 4, 1.5, h - pad.t - pad.b + 4);
  }
}

function niceTicks(max) {
  const raw = max / 4;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = 0; v <= max + 1e-9; v += step) out.push(Math.round(v * 100) / 100);
  return out;
}

export function lowerBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < x) lo = mid + 1; else hi = mid; }
  return lo;
}
