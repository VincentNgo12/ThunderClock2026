// File: main.js
// Purpose: ThunderSim 3D viewer app: scenario picker, run options, playback, cameras,
//          sensor table, event list and charts. Data comes from the local Python server.
// Contributors: Unassigned
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { World } from "./world.js";
import { Body } from "./body.js";
import { Timeline, SignalChart, lowerBound } from "./charts.js";
import { SLOT_NAMES, SLOT_COLORS, PHASE_COLORS, fmt, lerpFrame } from "./util.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

// slowmo: opt-in slow motion around falls (off: falls play in real time).
// slowRate: current slow-mo factor (1 = real time); it eases toward its goal, never steps.
// physics: ask the server for physics-baked limbs (query param physics=1).
// mocap: ask for motion-capture limbs (query param mocap=1/0); null = the server's default (off).
// contact: ask for the contact and balance pass (query param contact=1/0); null = the server's default (off).
const state = {
  catalogue: null, payload: null, scenario: null, t: 0, playing: false, speed: 1, follow: true,
  slowmo: false, slowRate: 1, physics: true, labels: true, cam: "chase", lastUi: 0, lastCharts: 0,
  slowWindows: [], camAnim: null, loadSeq: 0, snapFollow: false, rateKey: "", mocap: null, contact: null,
};

// --- playback and camera tuning ------------------------------------------------------------------
const SLOW_FACTOR = 0.25;            // slow-mo rate inside fall windows
const SLOW_WINDOW_S = [-0.5, 1.3];   // slow-mo window around each fall event, simulated s
const SLOW_EASE_S = 0.28;            // wall-clock time constant of the slow-mo rate ramp, s
const GRAVITY = 9.80665;             // m/s^2, only for the on-screen gravity readout
const FOLLOW_Y_SMOOTH_S = 0.8;       // vertical follow spring, simulated s: ~60 % of a height step after 0.8 s, ~90 % after 1.6 s
const LOOK_MAX_SCREEN = 0.55;        // worker may drop at most ~this fraction of the half-frame below centre (soft cap)
const FOLLOW_JUMP_M = 8;             // a larger per-frame target jump during playback is a teleport: hold the camera

// --- three.js setup ------------------------------------------------------------------------------
const viewport = $("viewport");
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: params.has("shot") });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
viewport.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xa9c4dd);
scene.fog = new THREE.Fog(0xb9cde0, 60, 260);
const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 900);
camera.position.set(4, 3, 6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.maxPolarAngle = Math.PI * 0.495;
controls.minDistance = 0.4;
controls.maxDistance = 120;

const sky = new THREE.Mesh(new THREE.SphereGeometry(600, 32, 16), new THREE.ShaderMaterial({
  side: THREE.BackSide, depthWrite: false, fog: false,
  uniforms: { top: { value: new THREE.Color(0x5b8fc9) }, bottom: { value: new THREE.Color(0xdfe8ef) } },
  vertexShader: "varying vec3 vP; void main(){ vP = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
  fragmentShader: "uniform vec3 top; uniform vec3 bottom; varying vec3 vP; void main(){ float h = clamp(vP.y*1.6+0.15,0.0,1.0); gl_FragColor = vec4(mix(bottom, top, h),1.0); }",
}));
scene.add(sky);
const hemi = new THREE.HemisphereLight(0xdbe8f5, 0x6b5a46, 1.1);
scene.add(hemi);
const sun = new THREE.DirectionalLight(0xfff1dc, 2.4);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -9; sun.shadow.camera.right = 9; sun.shadow.camera.top = 9; sun.shadow.camera.bottom = -9;
sun.shadow.camera.near = 1; sun.shadow.camera.far = 80;
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.02;
scene.add(sun, sun.target);

const world = new World(scene);
const body = new Body(scene);
const timeline = new Timeline($("timeline-canvas"), (t) => { seek(t); });
const chartAcc = new SignalChart($("chart-acc"), "acc");
const chartGyro = new SignalChart($("chart-gyro"), "gyro");

function resize() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = w + "px";
  renderer.domElement.style.height = h + "px";
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
  state.rateKey = "";  // re-place the slow-motion badge: the HUD may wrap differently
}
new ResizeObserver(resize).observe(viewport);
resize();

// --- catalogue and options -------------------------------------------------------------------------
async function loadCatalogue() {
  const res = await fetch("/api/catalogue");
  state.catalogue = await res.json();
  const list = $("scenario-list");
  const groups = state.catalogue.groups;
  $("scenario-count").textContent = `(${state.catalogue.scenarios.length})`;
  for (const [gid, gname] of Object.entries(groups)) {
    const items = state.catalogue.scenarios.filter((s) => s.group === gid);
    if (!items.length) continue;
    const h = document.createElement("div");
    h.className = "group-title"; h.textContent = gname; h.dataset.group = gid;
    list.appendChild(h);
    for (const s of items) {
      const b = document.createElement("button");
      b.className = "scn"; b.dataset.id = s.id;
      b.innerHTML = `<span class="id">${s.id}</span><span class="exp">${s.report_only ? "report only" : s.expected.replace("_", " ")}</span><span class="desc">${s.description}</span>`;
      b.addEventListener("click", () => { $("opt-seed").value = ""; selectScenario(s.id, true); });
      list.appendChild(b);
    }
  }
  for (const m of state.catalogue.models) $("opt-model").add(new Option(m, m));
  for (const p of state.catalogue.profiles) $("opt-profile").add(new Option(`${p.name} (${p.odr_hz} Hz)`, p.name));
  $("opt-model").value = params.get("model") || (state.catalogue.models.includes("falldet") ? "falldet" : "baseline");
  $("opt-profile").value = params.get("profile") || "target";
  if (params.has("obs")) $("opt-obs").value = params.get("obs");
  if (params.has("dead")) $("opt-dead").value = params.get("dead");
  if (params.has("seed")) $("opt-seed").value = params.get("seed");
}

$("scenario-search").addEventListener("input", (e) => {
  const q = e.target.value.toLowerCase();
  for (const b of document.querySelectorAll(".scn")) b.classList.toggle("hidden", q && !b.textContent.toLowerCase().includes(q));
});
$("btn-run").addEventListener("click", () => state.scenario && selectScenario(state.scenario, true));
$("btn-test-seed").addEventListener("click", () => { $("opt-seed").value = ""; if (state.scenario) selectScenario(state.scenario, true); });

function selectScenario(id, run) {
  state.scenario = id;
  for (const b of document.querySelectorAll(".scn")) b.classList.toggle("active", b.dataset.id === id);
  if (run) loadRun();
}

// opts: t (start time, s), play, cam (preset name), keepView (keep the current camera, e.g. on a physics toggle).
async function loadRun(opts = {}) {
  const seq = ++state.loadSeq;  // only the newest request may replace the run (quick re-toggles)
  const q = new URLSearchParams({
    scenario: state.scenario, seed: $("opt-seed").value || "auto", model: $("opt-model").value,
    profile: $("opt-profile").value, obs: $("opt-obs").value, dead: $("opt-dead").value,
    physics: state.physics ? "1" : "0",
  });
  if (state.mocap !== null) q.set("mocap", state.mocap ? "1" : "0");
  if (state.contact !== null) q.set("contact", state.contact ? "1" : "0");
  $("loading").classList.remove("hidden");
  $("error").classList.add("hidden");
  $("loading-text").textContent = `Simulating ${state.scenario}: body → MPU-6050s → BLE → ${$("opt-model").value} model`
    + `${state.physics ? " · physics limbs" : ""}${state.mocap ? " · mocap limbs" : ""}${state.contact ? " · contact pass" : ""}…`;
  try {
    const res = await fetch(`/api/run?${q}`);
    const payload = await res.json();
    if (seq !== state.loadSeq) return;
    if (payload.error) throw new Error(payload.error);
    setPayload(payload, opts);
  } catch (err) {
    if (seq !== state.loadSeq) return;
    $("error").textContent = `Could not run ${state.scenario}:\n${err.message}`;
    $("error").classList.remove("hidden");
  } finally {
    if (seq === state.loadSeq) $("loading").classList.add("hidden");
  }
}

function setPayload(P, opts = {}) {
  state.payload = P;
  $("opt-seed").value = P.meta.seed;
  document.querySelector('[data-toggle="mocap"]').classList.toggle("on", Boolean(P.meta.mocap?.enabled));
  document.querySelector('[data-toggle="contact"]').classList.toggle("on", Boolean(P.meta.contact?.enabled));
  world.build(P);
  world.buildDynamics(P);
  body.build(P);
  body.trail.visible = document.querySelector('[data-toggle="trail"]').classList.contains("on");
  timeline.set(P);
  chartAcc.set(P);
  chartGyro.set(P);
  state.slowWindows = slowWindows(P);
  buildInfo(P);
  buildEvents(P);
  const start = opts.t ?? defaultStart(P);
  state.t = Math.max(0, Math.min(start, P.meta.duration));
  state.slowRate = slowTarget(state.t);
  if (opts.keepView) {
    state.camAnim = null;
    resetFollow(camTarget(state.t * P.meta.fps));
  } else {
    placeCamera(opts.cam || state.cam, true);
  }
  updateUrl();
  state.playing = opts.play ?? true;
  $("btn-play").textContent = state.playing ? "⏸" : "▶";
  frame(performance.now(), true);
}

function defaultStart(P) {
  // Start a few seconds before the first interesting moment, after the calibration stand.
  const firsts = P.truth.filter((e) => ["fall_onset", "near_fall_onset", "struck", "hat_off", "strap_slip", "strip_removed", "free_fall_start"].includes(e.name));
  if (firsts.length) return Math.max(5, firsts[0].t - 6);
  const acts = P.activities.filter((a) => !["Calibration stand", "Standing", "Walking"].includes(a[2]));
  return acts.length ? Math.max(5, acts[0][0] - 3) : 5;
}

// Simulated-time windows [t0, t1] around fall events where opt-in slow-mo applies (overlaps merged).
function slowWindows(P) {
  const names = ["fall_onset", "near_fall_onset", "struck", "free_fall_start", "impact", "arrest_start", "hat_off", "slide_start"];
  const raw = P.truth.filter((e) => names.includes(e.name)).map((e) => [e.t + SLOW_WINDOW_S[0], e.t + SLOW_WINDOW_S[1]]);
  raw.sort((a, b) => a[0] - b[0]);
  const out = [];
  for (const w of raw) { if (out.length && w[0] <= out.at(-1)[1]) out.at(-1)[1] = Math.max(out.at(-1)[1], w[1]); else out.push(w); }
  return out;
}

function updateUrl() {
  const P = state.payload;
  if (!P) return;
  const q = new URLSearchParams({ s: P.meta.scenario, seed: P.meta.seed, model: P.meta.model, profile: P.meta.profile, obs: P.meta.obs_s });
  if (P.meta.dead_slot !== null) q.set("dead", P.meta.dead_slot);
  if (state.slowmo) q.set("slowmo", "1");     // default off
  if (!state.physics) q.set("physics", "0");  // default on
  if (P.meta.mocap?.enabled) q.set("mocap", "1");  // default off
  if (P.meta.contact?.enabled) q.set("contact", "1");  // default off
  history.replaceState(null, "", `?${q}`);
  document.title = `${P.meta.scenario} · ThunderSim 3D`;
}

// --- info panels ------------------------------------------------------------------------------------
function buildInfo(P) {
  const M = P.meta, R = P.result;
  $("run-title").innerHTML = `<b>${M.scenario}</b>${M.description} <span class="dim">· seed ${M.seed}</span>`;
  const chips = [];
  if (R) {
    const cls = R.passed ? "pass" : (R.partial ? "partial" : "fail");
    chips.push(`<span class="chip ${M.report_only ? "" : cls}">${M.report_only ? "report only" : (R.passed ? "PASS" : (R.partial ? "PARTIAL" : "FAIL"))}</span>`);
  }
  chips.push(`<span class="chip">${M.model}</span><span class="chip">${M.profile}</span>`);
  $("result-chips").innerHTML = chips.join("");
  let verdict = "";
  if (R) {
    const cls = M.report_only ? "" : (R.passed ? "pass" : (R.partial ? "partial" : "fail"));
    verdict = `<div class="verdict ${cls}"><span class="big">${M.report_only ? "REPORT ONLY" : (R.passed ? "PASS" : (R.partial ? "PARTIAL" : "FAIL"))}</span>
      <span>expected <b>${M.expected}</b>, model said <b>${R.output}</b></span></div>
      <dl class="kv"><dt>Alert latency</dt><dd>${R.latency_s === null ? "—" : fmt(R.latency_s, 2) + " s"}</dd>
      <dt>Supervisor alert after</dt><dd>${R.supervisor_latency_s === null ? "—" : fmt(R.supervisor_latency_s, 1) + " s"}</dd>
      <dt>Supervisor alerts / cancelled</dt><dd>${R.supervisor_alerts} / ${R.cancelled_alerts}</dd>
      <dt>Stage-1 triggers</dt><dd>${R.stage1_triggers}</dd>
      <dt>Packets delivered</dt><dd>${R.link.delivered} of ${R.link.sent} (${fmt(R.link.loss_pct, 1)} % lost)</dd>
      <dt>Radio latency p50 / p95</dt><dd>${fmt(R.link.latency_ms_p50, 0)} / ${fmt(R.link.latency_ms_p95, 0)} ms</dd>
      <dt>Clipped frames</dt><dd>${R.saturated_frames} of ${R.frames}</dd></dl>`;
  } else {
    verdict = `<div class="verdict"><span>No model selected: expected <b>${M.expected}</b>.</span></div>`;
  }
  $("info-result").innerHTML = `<h2>Result</h2><div class="result-head">${M.scenario}</div><div class="result-desc">${M.description}</div>${verdict}`;
  const W = M.worker, S = M.site;
  const params = Object.entries(M.params).filter(([k]) => !["surface", "condition", "clothing", "radio", "height_m", "mass_kg"].includes(k))
    .map(([k, v]) => `<dt>${k.replace(/_/g, " ")}</dt><dd>${typeof v === "number" ? fmt(v, 2) : v}</dd>`).join("");
  $("worker-info").innerHTML = `<dl class="kv">
    <dt>Worker</dt><dd>${fmt(W.height_m, 2)} m, ${fmt(W.mass_kg, 0)} kg + ${fmt(W.load_kg, 0)} kg PPE</dd>
    <dt>Clothing</dt><dd>${W.clothing}</dd><dt>Surface</dt><dd>${S.surface.replace("_", " ")}</dd><dt>Condition</dt><dd>${S.condition}</dd>
    <dt>Radio</dt><dd>${S.radio.replace("_", " ")}</dd><dt>Ambient</dt><dd>${S.ambient_c[0]} → ${S.ambient_c[1]} °C</dd>
    <dt>Device</dt><dd>${M.profile_info.odr_hz} Hz, ±${M.profile_info.accel_range_g} g, ±${M.profile_info.gyro_range_dps} °/s</dd>
    ${params}</dl>`;
  $("sensor-profile").textContent = `${M.profile} · ${M.profile_info.odr_hz} Hz`;
  $("notes").innerHTML = `<h2>Notes</h2>` + physicsInfo(M) + mocapInfo(M) + contactInfo(M)
    + M.notes.map((n, i) => `<p class="${i < M.notes.length - 1 ? "warn" : ""}">${n}</p>`).join("")
    + `<p>Timeline: phase band (ground truth), activity band, ◆ ground-truth events, ▲ model events. The orange line is now.</p>`;
  $("legend").innerHTML = SLOT_NAMES.map((n, i) => `<span><i style="background:${SLOT_COLORS[i]}"></i>${n}</span>`).join("");
}

// One physics window as text; accepts [t0, t1] or {t0, t1} / {start, end} (simulated s).
function windowText(w) {
  const a = Array.isArray(w) ? w[0] : (w?.t0 ?? w?.start);
  const b = Array.isArray(w) ? w[1] : (w?.t1 ?? w?.end);
  return typeof a === "number" && typeof b === "number" ? `${fmt(a, 2)}–${fmt(b, 2)} s` : JSON.stringify(w);
}

// meta.physics = {engine, version, substep_hz, enabled, windows, rest_lift_max_m?, sim_frames?} from the
// server's limb bake, as a small table. Windows are simulated-time spans where the limbs are a ragdoll.
function physicsInfo(M) {
  const ph = M.physics;
  if (!ph) {
    return state.physics ? `<p class="warn">Physics limbs were requested, but this run has no physics info: limbs are procedural.</p>` : "";
  }
  const windows = Array.isArray(ph.windows) ? ph.windows : [];
  const shown = windows.slice(0, 4).map(windowText).join(", ") + (windows.length > 4 ? ` (+${windows.length - 4} more)` : "");
  const requested = state.physics ? "on" : "off";
  const used = ph.enabled ? "on" : "off";
  const extra = [];
  if (typeof ph.rest_lift_max_m === "number") extra.push(`<dt>Rest lift (max)</dt><dd>${fmt(ph.rest_lift_max_m * 100, 1)} cm</dd>`);
  if (typeof ph.sim_frames === "number") extra.push(`<dt>Ragdoll frames</dt><dd>${ph.sim_frames} of ${M.frames}</dd>`);
  return `<dl class="kv physics-kv">
    <dt>Physics limbs</dt><dd>${used}${used !== requested ? ` (requested ${requested})` : ""}</dd>
    <dt>Engine</dt><dd>${ph.engine ?? "—"} ${ph.version ?? ""}</dd>
    <dt>Substep</dt><dd>${ph.substep_hz ?? "—"} Hz</dd>
    <dt>Windows</dt><dd>${windows.length ? shown : "none"}</dd>${extra.join("")}</dl>`;
}

// meta.mocap = {enabled, manifest_sha256, uses: [{role, limbs, clip, t0, t1, licence_class}], credits, problems}:
// which motion-capture clips drove which limbs and when (mocap.py). Credits are in the notes below.
function mocapInfo(M) {
  const mc = M.mocap;
  if (!mc || !mc.enabled) return "";
  const uses = mc.uses.map((u) => `${u.role.replace("_", " ")} ${u.limbs}: ${u.clip} (${fmt(u.t0, 1)}–${fmt(u.t1, 1)} s)`);
  const shown = uses.slice(0, 6).join("<br>") + (uses.length > 6 ? `<br>(+${uses.length - 6} more)` : "");
  return `<dl class="kv physics-kv">
    <dt>Mocap limbs</dt><dd>on${uses.length ? "" : " (no clip applies to this run: limbs are procedural)"}</dd>
    ${uses.length ? `<dt>Clips</dt><dd>${shown}</dd>` : ""}
    ${mc.problems.length ? `<dt>Problems</dt><dd>${mc.problems.join("<br>")}</dd>` : ""}</dl>`;
}

// meta.contact = {enabled, version, early_lifts, settle_steps, lift_set_steps, rail_regrips, unreachable_frames, ...}
// from the contact and balance pass (contact.py): what it changed and which handholds were out of reach.
function contactInfo(M) {
  const c = M.contact;
  if (!c || !c.enabled) return "";
  const steps = (c.early_lifts || 0) + (c.catch_steps || 0) + (c.settle_steps || 0) + (c.lift_set_steps || 0);
  return `<dl class="kv physics-kv">
    <dt>Contact pass</dt><dd>on (v${c.version})</dd>
    <dt>Feet</dt><dd>${steps} steps added or moved (early lifts ${c.early_lifts || 0}, settle ${c.settle_steps || 0}, before lifts ${c.lift_set_steps || 0})</dd>
    ${c.sacks ? `<dt>Sacks</dt><dd>${c.sacks} held at fixed handholds</dd>` : ""}
    ${c.rail_regrips ? `<dt>Ladder rails</dt><dd>hand over hand, ${c.rail_regrips} re-grips</dd>` : ""}
    <dt>Out of reach</dt><dd>${c.unreachable_frames ? `${c.unreachable_frames} hand frames (handhold out of the arm's reach)` : "none"}</dd></dl>`;
}

function buildEvents(P) {
  const rows = [];
  for (const e of P.truth) {
    const info = Object.entries(e.info).map(([k, v]) => `${k} ${typeof v === "number" ? fmt(v, 2) : v}`).join(", ");
    rows.push({ t: e.t, cls: "truth", html: `<span class="k">${e.name.replace(/_/g, " ")}</span>${info ? `<small>${info}</small>` : ""}` });
  }
  for (const e of P.model_events) {
    const cls = e.alert ? "alert" : (e.kind === "near_fall" ? "near" : (["recovered", "cleared"].includes(e.kind) ? "good" : "model"));
    rows.push({ t: e.t, cls: `model ${cls}`, html: `<span class="k">model: ${e.kind.replace(/_/g, " ")}</span><small>conf ${fmt(e.confidence, 2)} · at laptop ${fmt(e.t_laptop, 2)} s</small>` });
  }
  rows.sort((a, b) => a.t - b.t);
  const list = $("event-list");
  list.innerHTML = "";
  if (!rows.length) list.innerHTML = `<div class="dim">No events: a hard-negative run should stay quiet.</div>`;
  state.eventRows = rows.map((r) => {
    const div = document.createElement("div");
    div.className = `ev ${r.cls}`;
    div.innerHTML = `<span class="t">${fmt(r.t, 2)}</span><span>${r.html}</span>`;
    div.addEventListener("click", () => seek(Math.max(0, r.t - 1.5)));
    list.appendChild(div);
    return { t: r.t, el: div };
  });
}

function sampleAt(t) {
  const S = state.payload.signals;
  const i = lowerBound(S.t, t + 1e-9) - 1;
  if (i < 0) return null;
  return {
    i, ok: S.ok[i], fw: S.fw.slice(i * 12, i * 12 + 12), ww: S.ww.slice(i * 12, i * 12 + 12),
    acc: S.acc_g.slice(i * 4, i * 4 + 4), gyro: S.gyro_dps.slice(i * 4, i * 4 + 4), sat: S.sat.slice(i * 4, i * 4 + 4), age: t - S.t[i],
  };
}

function updatePanels(t, sample) {
  const P = state.payload;
  // activity and phase
  const act = [...P.activities].reverse().find((a) => a[0] <= t && t < a[1]) || [...P.activities].reverse().find((a) => a[0] <= t);
  const ph = P.phases.find((p) => p[0] <= t && t < p[1]) || P.phases.at(-1);
  const phaseName = P.meta.phase_names[ph[2]];
  const fallLike = ["fall_descent", "impact", "lying", "suspended", "near_fall", "free_fall", "collapse", "device_off", "recovery", "lying_work"].includes(phaseName);
  $("now-activity").textContent = fallLike ? phaseName.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase()) : (act ? act[2] : "—");
  $("now-phase").textContent = `phase: ${phaseName}`;
  $("now-phase-swatch").style.background = PHASE_COLORS[phaseName] || "#555";
  // sensor table
  const prof = P.meta.profile_info;
  let html = `<table class="sensors"><tr><th>anchor</th><th>|a| g</th><th>|ω| °/s</th><th>state</th></tr>`;
  for (let s = 0; s < 4; s++) {
    const dead = P.meta.dead_slot === s;
    const a = sample ? sample.acc[s] : NaN, w = sample ? sample.gyro[s] : NaN;
    let st = "ok", cls = "";
    if (dead) { st = "dead"; cls = "st-dead"; } else if (!sample) { st = "waiting"; cls = "st-lost"; } else if (sample.sat[s]) { st = "clipped"; cls = "st-clip"; } else if (!sample.ok) { st = "lost"; cls = "st-lost"; } else if (sample.age > 0.6) { st = `held ${fmt(sample.age, 1)} s`; cls = "st-lost"; }
    const pct = Math.min(100, (a / Math.max(prof.accel_range_g, 2)) * 100);
    html += `<tr><td><i style="background:${SLOT_COLORS[s]}"></i>${SLOT_NAMES[s]}<div class="bar"><div style="width:${dead ? 0 : pct}%;background:${SLOT_COLORS[s]}"></div></div></td>
      <td>${dead ? "—" : fmt(a, 2)}</td><td>${dead ? "—" : fmt(w, 0)}</td><td class="${cls}">${st}</td></tr>`;
  }
  $("sensor-table").innerHTML = html + "</table>";
  for (const row of state.eventRows || []) {
    row.el.classList.toggle("past", row.t < t - 1.5);
    row.el.classList.toggle("current", row.t <= t + 0.05 && row.t > t - 1.5);
  }
}

// --- labels projected over the canvas --------------------------------------------------------------
const labelEls = SLOT_NAMES.map((n, i) => {
  const d = document.createElement("div");
  d.className = "slabel"; d.style.borderColor = SLOT_COLORS[i];
  $("labels-layer").appendChild(d);
  return d;
});

function updateLabels(sample) {
  const P = state.payload;
  const w = viewport.clientWidth, h = viewport.clientHeight;
  for (let i = 0; i < 4; i++) {
    const el = labelEls[i];
    if (!state.labels || !body.pods) { el.style.display = "none"; continue; }
    const p = body.pods[i].group.position.clone().project(camera);
    if (p.z > 1) { el.style.display = "none"; continue; }
    el.style.display = "block";
    const off = (i - 1.5) * 0;
    el.style.left = `${(p.x * 0.5 + 0.5) * w}px`;
    el.style.top = `${(-p.y * 0.5 + 0.5) * h + off}px`;
    const dead = P.meta.dead_slot === i;
    el.classList.toggle("dead", dead);
    el.classList.toggle("clip", !!(sample && sample.sat[i]));
    el.textContent = dead ? `${SLOT_NAMES[i]} DEAD` : `${SLOT_NAMES[i]} ${sample ? fmt(sample.acc[i], 2) : "—"} g`;
  }
}

// --- cameras ---------------------------------------------------------------------------------------
function headingAt(f) {
  const q = new THREE.Quaternion();
  const sq = state.payload.frames.segq;
  const i = Math.max(0, Math.min(Math.round(f), sq.length / 16 - 1)) * 16;
  q.set(sq[i], sq[i + 1], sq[i + 2], sq[i + 3]);
  const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(q);
  fwd.y = 0;
  if (fwd.lengthSq() < 1e-4) fwd.set(0, 0, -1);
  return fwd.normalize();
}

function camTarget(f) {
  const p = body.pelvis(f);
  return p.add(new THREE.Vector3(0, 0.25, 0));
}

// Follow state. cam: point the camera body tracks (x, z = worker; y = spring height).
// look: point the orbit target tracks. worker: last worker point. y, vy: spring height (m) and rate (m/s).
const camFollow = { cam: null, look: null, worker: null, y: 0, vy: 0 };

function resetFollow(target) {
  camFollow.cam = target.clone();
  camFollow.look = target.clone();
  camFollow.worker = target.clone();
  camFollow.y = target.y;
  camFollow.vy = 0;
}

// Exact step of a critically damped spring toward a fixed goal over dt (stable for any dt).
function springStep(y, vy, goal, dt, omega) {
  const offset = y - goal;
  const temp = (vy + omega * offset) * dt;
  const decay = Math.exp(-omega * dt);
  return [goal + (offset + temp) * decay, (vy - omega * temp) * decay];
}

// Move the camera with the worker: horizontally 1:1, vertically through a critically damped spring
// stepped in simulated time (same look at every playback rate), so a fall is seen against the site
// instead of the site rising past a fixed worker. The camera and its orbit target both ride the spring
// height; the orbit target is softly held within LOOK_MAX_SCREEN of the worker so a long drop stays
// in frame. simDt: simulated s since the last frame.
function followTarget(target, simDt) {
  const snap = state.snapFollow;  // after a seek: jump with the worker, no spring lag
  if (!snap && target.distanceTo(camFollow.worker) > FOLLOW_JUMP_M) { resetFollow(target); return; }
  if (snap) {
    camFollow.y = target.y;
    camFollow.vy = 0;
  } else {
    [camFollow.y, camFollow.vy] = springStep(camFollow.y, camFollow.vy, target.y, simDt, 2 / FOLLOW_Y_SMOOTH_S);
  }
  const lag = target.y - camFollow.y;  // < 0 while the worker drops faster than the camera
  const halfFrame = camera.position.distanceTo(controls.target) * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const maxOffset = Math.max(LOOK_MAX_SCREEN * halfFrame, 1e-3);
  const lookY = target.y - maxOffset * Math.tanh(lag / maxOffset);  // = spring height for small lags
  const camPoint = new THREE.Vector3(target.x, camFollow.y, target.z);
  const lookPoint = new THREE.Vector3(target.x, lookY, target.z);
  camera.position.add(camPoint.clone().sub(camFollow.cam));
  controls.target.add(lookPoint.clone().sub(camFollow.look));
  camFollow.cam.copy(camPoint);
  camFollow.look.copy(lookPoint);
  camFollow.worker.copy(target);
}

function placeCamera(kind, instant = false) {
  if (!state.payload) return;
  state.cam = kind;
  for (const b of document.querySelectorAll("[data-cam]")) b.classList.toggle("on", b.dataset.cam === kind);
  const f = state.t * state.payload.meta.fps;
  const target = camTarget(f);
  const fwd = headingAt(f);
  const right = new THREE.Vector3(-fwd.z, 0, fwd.x);
  const up = new THREE.Vector3(0, 1, 0);
  const H = state.payload.meta.body.H;
  let pos;
  if (kind === "side") pos = target.clone().addScaledVector(right, 4.2).addScaledVector(up, 0.5);
  else if (kind === "front") pos = target.clone().addScaledVector(fwd, 4.2).addScaledVector(up, 0.7);
  else if (kind === "back") {
    const sp = state.payload.frames.spine;
    const i = Math.round(f) * 12;
    target.set((sp[i + 3] + sp[i + 6]) / 2, (sp[i + 4] + sp[i + 7]) / 2, (sp[i + 5] + sp[i + 8]) / 2);
    pos = target.clone().addScaledVector(fwd, -1.25).addScaledVector(right, 0.45).addScaledVector(up, 0.35);
  } else if (kind === "top") pos = target.clone().addScaledVector(up, 9).addScaledVector(fwd, -0.5);
  else if (kind === "wide") pos = target.clone().addScaledVector(fwd, -9).addScaledVector(right, 7).addScaledVector(up, 6 + H);
  else pos = target.clone().addScaledVector(fwd, -3.6).addScaledVector(right, 1.6).addScaledVector(up, 1.5);
  if (instant) {
    controls.target.copy(target);
    camera.position.copy(pos);
    state.camAnim = null;
  } else {
    state.camAnim = { from: camera.position.clone(), fromT: controls.target.clone(), to: pos, toT: target, start: performance.now() };
  }
  resetFollow(camTarget(f));  // follow tracks the worker point even when the preset aims elsewhere (Strip)
}

for (const b of document.querySelectorAll("[data-cam]")) b.addEventListener("click", () => placeCamera(b.dataset.cam));

// --- toggles ---------------------------------------------------------------------------------------
// Sets a toggle's state (flips it when on is undefined) and returns the new state.
function setToggle(name, on) {
  const b = document.querySelector(`[data-toggle="${name}"]`);
  if (on === undefined) on = !b.classList.contains("on");
  b.classList.toggle("on", on);
  if (name === "follow") state.follow = on;
  if (name === "xray") body.setXray(on);
  if (name === "forces") body.showForces = on;
  if (name === "gyro") body.showGyro = on;
  if (name === "labels") state.labels = on;
  if (name === "trail" && body.trail) body.trail.visible = on;
  if (name === "slowmo") state.slowmo = on;
  if (name === "physics") state.physics = on;
  if (name === "mocap") state.mocap = on;
  if (name === "contact") state.contact = on;
  return on;
}

// A toggle flipped by the user (button or key): slow-mo eases in or out; physics reloads the run in place.
function userToggle(name) {
  setToggle(name);
  if ((name === "physics" || name === "mocap" || name === "contact") && state.payload) loadRun({ t: state.t, play: state.playing, keepView: true });
  if (name === "slowmo") {
    if (!state.playing) state.slowRate = slowTarget(state.t);  // paused: show the new rate at once
    updateRateDisplay();
    updateUrl();
  }
}
for (const b of document.querySelectorAll("[data-toggle]")) b.addEventListener("click", () => userToggle(b.dataset.toggle));

// --- playback --------------------------------------------------------------------------------------
function seek(t) {
  if (!state.payload) return;
  state.t = Math.max(0, Math.min(t, state.payload.meta.duration));
  state.slowRate = slowTarget(state.t);  // a jump has no motion to ease across
  state.snapFollow = true;
  frame(performance.now(), true);
}

// Slow-mo factor the playback eases toward at simulated time t (1 = real time).
function slowTarget(t) {
  if (!state.slowmo) return 1;
  return state.slowWindows.some(([a, b]) => t >= a && t <= b) ? SLOW_FACTOR : 1;
}

// Advance the playback clock by one frame. The slow-mo factor approaches its goal exponentially
// (time constant SLOW_EASE_S of wall time), so the rate never steps. Returns simulated s advanced.
function advanceClock(wallDt) {
  const goal = slowTarget(state.t);
  state.slowRate = goal + (state.slowRate - goal) * Math.exp(-wallDt / SLOW_EASE_S);
  const before = state.t;
  state.t = Math.min(state.t + wallDt * state.speed * state.slowRate, state.payload.meta.duration);
  return state.t - before;
}

// Rate as short text: 2 significant digits (0.25, 0.025, 2).
function rateText(x) {
  return String(Number(x.toPrecision(2)));
}

// Effective playback rate (speed x slow-mo) on the badge, shown only while slower than real time,
// and on the clock, shown whenever it is not 1.
function updateRateDisplay() {
  const rate = state.speed * state.slowRate;
  const slowmoActive = state.slowRate < 0.995;
  const key = `${rateText(rate)}|${slowmoActive}`;
  if (key === state.rateKey) return;
  state.rateKey = key;
  const slowed = rate < 0.995;
  $("slowmo-badge").classList.toggle("hidden", !slowed);
  if (slowed) {
    // just below the HUD (activity box, toggles), which wraps to more rows on a narrow stage
    const hudBottom = Math.max($("toggles").getBoundingClientRect().bottom, $("now-activity").parentElement.getBoundingClientRect().bottom);
    $("slowmo-badge").style.top = `${Math.round(hudBottom - viewport.parentElement.getBoundingClientRect().top + 10)}px`;
    $("slowmo-rate").textContent = `×${rateText(rate)}`;
    const cause = slowmoActive ? "Slow-mo around the fall" : "Playback speed";
    $("slowmo-note").textContent = `${cause} · gravity on screen ${rateText(GRAVITY * rate * rate)} m/s² (real ${GRAVITY.toFixed(2)})`;
  }
  $("clock-rate").textContent = `×${rateText(rate)}`;
  $("clock-rate").classList.toggle("hidden", Math.abs(rate - 1) < 0.005);
}

function jumpEvent(dir) {
  const P = state.payload;
  if (!P) return;
  const times = [...P.truth.map((e) => e.t), ...P.model_events.map((e) => e.t)].sort((a, b) => a - b);
  const pre = 1.5;
  if (dir > 0) { const n = times.find((x) => x - pre > state.t + 0.05); if (n !== undefined) seek(n - pre); }
  else { const p = [...times].reverse().find((x) => x - pre < state.t - 0.3); seek(p !== undefined ? p - pre : 0); }
}

$("btn-play").addEventListener("click", togglePlay);
$("btn-start").addEventListener("click", () => seek(0));
$("btn-prev-event").addEventListener("click", () => jumpEvent(-1));
$("btn-next-event").addEventListener("click", () => jumpEvent(1));
$("opt-speed").addEventListener("change", (e) => { state.speed = parseFloat(e.target.value); });

function togglePlay() {
  if (!state.payload) return;
  if (!state.playing && state.t >= state.payload.meta.duration - 0.01) seek(0);  // replay from the start
  state.playing = !state.playing;
  $("btn-play").textContent = state.playing ? "⏸" : "▶";
}

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const k = e.key;
  if (k === " ") { e.preventDefault(); togglePlay(); }
  else if (k === "ArrowRight") seek(state.t + (e.shiftKey ? 1 : 0.1));
  else if (k === "ArrowLeft") seek(state.t - (e.shiftKey ? 1 : 0.1));
  else if (k === "Home") seek(0);
  else if (k === "j" || k === "J") jumpEvent(-1);
  else if (k === "k" || k === "K") jumpEvent(1);
  else if (k === "[" || k === "]") {
    const sel = $("opt-speed"); const i = sel.selectedIndex + (k === "]" ? 1 : -1);
    if (i >= 0 && i < sel.options.length) { sel.selectedIndex = i; state.speed = parseFloat(sel.value); }
  } else {
    const map = { f: "follow", x: "xray", v: "forces", g: "gyro", l: "labels", t: "trail", s: "slowmo", p: "physics", m: "mocap", c: "contact" };
    if (map[k.toLowerCase()]) userToggle(map[k.toLowerCase()]);
    const cams = { 1: "chase", 2: "side", 3: "front", 4: "back", 5: "top", 6: "wide" };
    if (cams[k]) placeCamera(cams[k]);
  }
});

let last = performance.now();
function frame(now, force = false) {
  // wall s since the last frame, in [0, 0.1]: a rAF timestamp can precede the performance.now() of a
  // forced frame (seek, load), and a negative step would run the clock backwards
  const wallDt = Math.max(0, Math.min(0.1, (now - last) / 1000));
  last = now;
  const P = state.payload;
  if (P) {
    let simDt = 0;
    if (state.playing && !force) {
      simDt = advanceClock(wallDt);
      if (state.t >= P.meta.duration) { state.playing = false; $("btn-play").textContent = "▶"; }
    } else if (!state.playing) {
      state.slowRate = slowTarget(state.t);  // paused: nothing moves, so the rate may settle at once
    }
    const f = state.t * P.meta.fps;
    const sample = sampleAt(state.t);
    body.update(f, state.t, sample);
    world.update(f, P, body);
    // camera: preset tween, else follow (horizontal 1:1, vertical spring), else just track the worker point
    const target = camTarget(f);
    if (state.camAnim) {
      const u = Math.min(1, (now - state.camAnim.start) / 600);
      const e = u * u * (3 - 2 * u);
      camera.position.lerpVectors(state.camAnim.from, state.camAnim.to, e);
      controls.target.lerpVectors(state.camAnim.fromT, state.camAnim.toT, e);
      if (u >= 1) state.camAnim = null;
      resetFollow(target);
    } else if (state.follow && camFollow.cam) {
      followTarget(target, simDt);
    } else {
      resetFollow(target);
    }
    state.snapFollow = false;
    const floor = world.solidTop(camera.position.x, camera.position.z) + 0.35;
    if (camera.position.y < floor) camera.position.y = floor;
    sun.position.copy(controls.target).add(new THREE.Vector3(6, 14, 4));
    sun.target.position.copy(controls.target);
    if (world.grid) world.grid.position.set(Math.round(target.x / 4) * 4, world.grid.position.y, Math.round(target.z / 4) * 4);
    $("clock-t").textContent = state.t.toFixed(2);
    $("clock-total").textContent = P.meta.duration.toFixed(2);
    updateRateDisplay();
    if (force || now - state.lastCharts > 33) {
      timeline.draw(state.t);
      chartAcc.draw(state.t);
      chartGyro.draw(state.t);
      state.lastCharts = now;
    }
    if (force || now - state.lastUi > 120) { updatePanels(state.t, sample); state.lastUi = now; }
    controls.update();
    renderer.render(scene, camera);
    updateLabels(sample);
  } else {
    controls.update();
    renderer.render(scene, camera);
  }
  if (!force) requestAnimationFrame(frame);
}

// --- boot ------------------------------------------------------------------------------------------
(async function boot() {
  await loadCatalogue();
  if (params.has("xray")) setToggle("xray", params.get("xray") === "1");
  if (params.has("gyro")) setToggle("gyro", params.get("gyro") === "1");
  if (params.has("follow")) setToggle("follow", params.get("follow") === "1");
  if (params.has("slowmo")) setToggle("slowmo", params.get("slowmo") === "1");    // default off
  if (params.has("physics")) setToggle("physics", params.get("physics") !== "0"); // default on; read before the first run
  if (params.has("mocap")) setToggle("mocap", params.get("mocap") === "1");       // default off (server default)
  if (params.has("contact")) setToggle("contact", params.get("contact") === "1"); // default off (server default)
  if (params.has("speed")) { state.speed = parseFloat(params.get("speed")); $("opt-speed").value = params.get("speed"); }
  const first = params.get("s") || "H-LAD";
  selectScenario(first, false);
  await loadRun({ t: params.has("t") ? parseFloat(params.get("t")) : undefined, play: params.has("play") ? params.get("play") === "1" : undefined,
    cam: params.get("cam") || undefined });
  if (params.has("at") && state.payload) {
    // deep link to an event: ?at=impact&dt=-0.5 (ground-truth event name or model event kind)
    const name = params.get("at");
    const ev = state.payload.truth.find((e) => e.name === name) || state.payload.model_events.find((e) => e.kind === name);
    if (ev) { seek(ev.t + parseFloat(params.get("dt") || "0")); placeCamera(params.get("cam") || state.cam, true); }
  }
  requestAnimationFrame(frame);
  if (params.has("shot")) {
    // headless screenshot mode: render a few frames, then mark ready
    for (let i = 0; i < 3; i++) frame(performance.now(), true);
    document.title = "READY";
  }
})();
