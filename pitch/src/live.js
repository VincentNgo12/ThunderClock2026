/* Page 4 (#live): the team's SpineView ported to vanilla three.js r147.
   Two sources: (1) LIVE, the backend telemetry WebSocket (WorkerTelemetry 2.0, docs/TELEMETRY_PROTOCOL.md),
   connected like frontend/src/hooks/useTelemetry.ts (default ws://127.0.0.1:8000/ws/telemetry, ?ws= override),
   with reconnect backoff; (2) REPLAY, a simulated ThunderSim run plus synthetic streams for the other
   firmware sensors on the same clock. Auto mode shows live whenever the backend streams.
   Spine math follows frontend/src/lib/posture3d/spine.ts (cubic Hermite in rotation-vector space).
   Frame: +X wearer-right, +Y up, +Z posterior. Run quaternions are (x, y, z, w). */
(function () {
  'use strict';
  const SEC = document.getElementById('live');
  if (!SEC || !window.THREE) return;
  const T = window.THREE;
  const DEG = 180 / Math.PI;
  const $ = (id) => document.getElementById(id);
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');

  // ---------- tokens ----------
  let C = {};
  function readTokens() {
    const cs = getComputedStyle(document.documentElement);
    const g = (n) => cs.getPropertyValue(n).trim();
    C = { bg: g('--bg'), ink: g('--ink'), ink2: g('--ink-2'), ink3: g('--ink-3'), rule: g('--rule'), surface: g('--surface'), alert: g('--alert'), accent: g('--accent') };
  }
  readTokens();

  // ---------- run decoding ----------
  const PHASE_LABEL = { 1: 'Calibration stand', 2: 'Falling', 3: 'Impact', 4: 'Lying still', 5: 'Hanging in harness', 6: 'Recovering',
    7: 'Near fall', 8: 'Working lying down', 9: 'Strip taken off', 10: 'Free fall', 11: 'Collapse' };
  const PHASE_KIND = { 2: 'fall', 3: 'fall', 10: 'fall', 11: 'fall', 4: 'lying', 5: 'lying', 6: 'near', 7: 'near' };
  const LOOP_START = { 'H-LAD': 12, 'NF-STUMBLE': 8, 'A-CARRY': 8, 'A-REBAR': 8 };
  const ACT_LEVEL = { 'Calibration stand': 0, Standing: 0.05, Walking: 0.35, 'Climbing up': 0.8, 'Carrying a load': 0.75, 'Tying rebar': 0.45 };

  function b64buf(s) { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u.buffer; }
  // smooth deterministic noise: sum of incommensurate sines, roughly in [-1, 1]
  const sn = (t, s) => (Math.sin(t * 0.731 + s * 1.3) + 0.6 * Math.sin(t * 1.913 + s * 2.1) + 0.35 * Math.sin(t * 4.37 + s * 0.7)) / 1.95;
  // hash noise for the noise-level samples
  const hn = (n, s) => { const x = Math.sin(n * 12.9898 + s * 78.233) * 43758.5453; return (x - Math.floor(x)) * 2 - 1; };

  function decodeRun(p) {
    const F = p.frames, fps = p.fps;
    const sq = new Int16Array(b64buf(p.segq)), a16 = new Uint16Array(b64buf(p.acc));
    const r = { id: p.id, seed: p.seed, F, fps, duration: (F - 1) / fps, site: p.site || {}, phases: p.phases || [], activities: p.activities || [],
      events: (p.events || []).slice().sort((a, b) => a.t - b.t), truth: p.truth || [], result: p.result || {},
      phase: new Uint8Array(b64buf(p.phase)), lost: new Uint8Array(b64buf(p.lost)), acc: new Float32Array(F * 4) };
    for (let i = 0; i < a16.length; i++) r.acc[i] = a16[i] / 100;
    const cal = r.phases.find((x) => x[2] === 1);
    const iRef = clamp(Math.round((cal ? (cal[0] + cal[1]) / 2 : 2.5) * fps), 0, F - 1);
    r.loopStart = Math.min(LOOP_START[r.id] || (cal ? cal[1] : 5), r.duration - 5);

    // heading (yaw of the pelvis forward axis), unwrapped, held while the pelvis is not upright
    const q = [0, 1, 2, 3].map(() => new T.Quaternion());
    const load = (f) => { for (let k = 0; k < 4; k++) { const o = f * 16 + k * 4; q[k].set(sq[o] / 1e4, sq[o + 1] / 1e4, sq[o + 2] / 1e4, sq[o + 3] / 1e4).normalize(); } };
    const fwd = new T.Vector3(), pu = new T.Vector3(), up = new T.Vector3();
    const heading = new Float32Array(F); r.trunk = new Float32Array(F);
    let lastH = 0, prev = null, accH = 0;
    for (let f = 0; f < F; f++) {
      load(f);
      fwd.set(0, 0, -1).applyQuaternion(q[0]); pu.set(0, 1, 0).applyQuaternion(q[0]);
      if (pu.y > 0.45 && Math.hypot(fwd.x, fwd.z) > 0.2) lastH = Math.atan2(-fwd.x, -fwd.z);
      if (prev === null) accH = lastH; else { let d = lastH - prev; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; accH += d; }
      prev = lastH; heading[f] = accH;
      up.set(0, 1, 0).applyQuaternion(q[3]);
      const fl = Math.hypot(fwd.x, fwd.z) || 1;
      const sgn = (up.x * fwd.x + up.z * fwd.z) / fl >= 0 ? 1 : -1;
      r.trunk[f] = sgn * Math.acos(clamp(up.y, -1, 1)) * DEG;
    }
    // neutral-corrected segment rotations: D_k = (Ry(-h_f) q_k(f)) (Ry(-h_ref) q_k(ref))^-1
    const Y = new T.Vector3(0, 1, 0), yaw = new T.Quaternion();
    load(iRef);
    yaw.setFromAxisAngle(Y, -heading[iRef]);
    const refInv = q.map((x) => yaw.clone().multiply(x).invert());
    r.D = new Float32Array(F * 16); r.tilt = new Float32Array(F * 4);
    const tmp = new T.Quaternion();
    for (let f = 0; f < F; f++) {
      load(f); yaw.setFromAxisAngle(Y, -heading[f]);
      for (let k = 0; k < 4; k++) {
        tmp.copy(yaw).multiply(q[k]).multiply(refInv[k]).normalize();
        const o = f * 16 + k * 4; r.D[o] = tmp.x; r.D[o + 1] = tmp.y; r.D[o + 2] = tmp.z; r.D[o + 3] = tmp.w;
        up.set(0, 1, 0).applyQuaternion(tmp); r.tilt[f * 4 + k] = Math.acos(clamp(up.y, -1, 1)) * DEG;
      }
    }
    // data age per frame (ms) from lost packet bins, for the freshness state
    r.age = new Float32Array(F);
    for (let f = 0; f < F; f++) r.age[f] = r.lost[f] ? (f ? r.age[f - 1] : 0) + 1000 / fps : 0;
    synthesise(r);
    return r;
  }

  function activityAt(r, t) {
    for (const a of r.activities) if (t >= a[0] && t < a[1]) return a[2];
    let prevA = null; for (const a of r.activities) if (a[1] <= t) prevA = a;
    return prevA && /Climbing/.test(prevA[2]) ? 'On the ladder' : 'Holding position';
  }

  // ---------- synthetic streams (deterministic, react to the run) ----------
  function synthesise(r) {
    const F = r.F, dt = 1 / r.fps, s = r.seed + r.id.length;
    const impacts = r.truth.filter((e) => e.name === 'impact').map((e) => e.t);
    const nearT = r.truth.filter((e) => e.name === 'near_fall_onset').map((e) => e.t);
    const lifts = r.truth.filter((e) => e.name === 'lift_off' || e.name === 'put_down_touch').map((e) => e.t);
    const amb = r.site.ambient_c || [16, 26];
    const heavy = r.site.radio === 'heavy_equipment';
    const S = (n) => new Float32Array(F);
    Object.assign(r, { hr: S(), temp: S(), hum: S(), eco2: S(), tvoc: S(), noise: S() });
    let hr = 74, act = 0, exSlow = 0;
    for (let f = 0; f < F; f++) {
      const t = f * dt, ph = r.phase[f], kind = PHASE_KIND[ph];
      const lying = kind === 'fall' || kind === 'lying';
      const lvl = lying ? 0.05 : ph === 1 ? 0 : (ACT_LEVEL[activityAt(r, t)] ?? 0.3);
      act += (lvl - act) * (1 - Math.exp(-dt / 6));
      exSlow += (act - exSlow) * (1 - Math.exp(-dt / 90));
      let stress = 0;
      for (const ti of impacts) if (t >= ti) stress = Math.max(stress, Math.exp(-(t - ti) / 70));
      for (const ti of nearT) if (t >= ti) stress = Math.max(stress, 0.55 * Math.exp(-(t - ti) / 18));
      for (const ti of lifts) if (t >= ti && t < ti + 4) stress = Math.max(stress, 0.15);
      const target = 72 + 40 * act + 46 * stress + 2.2 * sn(t * 0.35, s);
      hr += (target - hr) * (1 - Math.exp(-dt / (target > hr ? 5 : 16)));
      r.hr[f] = hr + 0.8 * sn(t * 2.3, s + 3);
      r.temp[f] = amb[0] + 0.42 * (amb[1] - amb[0]) + 1.4 * exSlow + 0.12 * sn(t * 0.05, s + 5) + 0.03 * sn(t * 0.9, s);
      r.hum[f] = 44 + 9 * exSlow + 1.5 * sn(t * 0.04, s + 7) + 0.3 * sn(t * 0.7, s + 1);
      const burst = heavy ? Math.max(0, sn(t * 0.09, s + 11)) : 0;
      r.eco2[f] = 470 + 70 * (sn(t * 0.03, s + 13) + 1) + 260 * burst + 6 * sn(t * 0.8, s);
      r.tvoc[f] = 90 + 0.32 * (r.eco2[f] - 450) + 12 * sn(t * 0.11, s + 17);
      let nz = 57 + 3 * sn(t * 0.21, s + 19) + 11 * Math.max(0, sn(t * 0.13, s + 23)) * (heavy ? 1 : 0.5);
      if (/Walking|Carrying|Climbing/.test(activityAt(r, t)) && !lying) nz += 2.5 + 1.5 * Math.abs(Math.sin(t * Math.PI * 1.8));
      for (const ti of impacts) if (t >= ti && t < ti + 3) nz = Math.max(nz, 60 + 30 * Math.exp(-(t - ti) / 0.5));
      for (const ti of lifts) if (t >= ti && t < ti + 1.5) nz = Math.max(nz, 64 + 10 * Math.exp(-(t - ti) / 0.4));
      r.noise[f] = nz + 1.2 * hn(f, s);
    }
  }

  function aqiOf(eco2) { return eco2 < 600 ? 1 : eco2 < 800 ? 2 : eco2 < 1000 ? 3 : eco2 < 1500 ? 4 : 5; }

  function detectorState(r, t) {
    let st = { level: 'ok', head: 'No fall detected', ev: null };
    for (const e of r.events) {
      if (e.t > t) break;
      if (e.kind === 'suspected_fall') st = { level: 'watch', head: 'Suspected fall, confirming', ev: e };
      else if (e.kind === 'fall_confirmed') st = { level: 'alert', head: 'Fall confirmed. Supervisor alerted', ev: e };
      else if (e.kind === 'near_fall') st = { level: 'watch', head: 'Near fall logged, no alert', ev: e };
    }
    return st;
  }

  // ---------- spine pose (port of spine.ts) ----------
  // SENSOR_ANCHORS: pods on sacrum, L3, T8, T2; the upper thoracic bend takes effect from T4 (bendIndex 14).
  const ANCHORS = [0, 3, 10, 16];
  const KNOTS = [0, 3, 10, 14];
  const REGION = ['Pelvis', 'Lumbar', 'Thoracic', 'Upper thoracic'];
  let VERT = null, REST = null, INTERP = 'spline';
  const UP = new T.Vector3(0, 1, 0), XA = new T.Vector3(1, 0, 0);
  function quatLog(q) {
    const sgn = q.w < 0 ? -1 : 1, v = new T.Vector3(q.x * sgn, q.y * sgn, q.z * sgn), sh = v.length();
    if (sh < 1e-9) return v.multiplyScalar(2);
    return v.multiplyScalar((2 * Math.atan2(sh, q.w * sgn)) / sh);
  }
  function quatExp(v) {
    const a = v.length();
    if (a < 1e-9) return new T.Quaternion(v.x / 2, v.y / 2, v.z / 2, 1).normalize();
    return new T.Quaternion().setFromAxisAngle(v.clone().divideScalar(a), a);
  }
  function segmentRotations(anchors, mode) {
    const knots = KNOTS, spans = knots.length - 1, deltas = [], widths = [];
    for (let k = 0; k < spans; k++) { deltas.push(quatLog(anchors[k + 1].clone().multiply(anchors[k].clone().invert()))); widths.push(knots[k + 1] - knots[k]); }
    const slopes = deltas.map((d, k) => d.clone().divideScalar(widths[k]));
    const rates = new Array(knots.length);
    for (let k = 1; k < spans; k++) { const b = widths[k - 1], a = widths[k]; rates[k] = slopes[k - 1].clone().multiplyScalar(a).addScaledVector(slopes[k], b).divideScalar(a + b); }
    rates[0] = slopes[0].clone().multiplyScalar(3).sub(rates[1]).multiplyScalar(0.5);
    rates[spans] = slopes[spans - 1].clone().multiplyScalar(3).sub(rates[spans - 1]).multiplyScalar(0.5);
    return VERT.map((_, i) => {
      if (i <= knots[0]) return anchors[0].clone();
      if (i >= knots[spans]) return anchors[spans].clone();
      const k = knots.findIndex((kn, j) => j < spans && i >= kn && i <= knots[j + 1]);
      const w = widths[k], t = (i - knots[k]) / w;
      if (mode === 'linear') return anchors[k].clone().slerp(anchors[k + 1], t);
      const t2 = t * t, t3 = t2 * t;
      const v = rates[k].clone().multiplyScalar((t3 - 2 * t2 + t) * w).addScaledVector(deltas[k], -2 * t3 + 3 * t2).addScaledVector(rates[k + 1], (t3 - t2) * w);
      return quatExp(v).multiply(anchors[k]);
    });
  }
  function computeSpinePose(anchors, mode) {
    const out = [], cur = new T.Vector3(), seg = segmentRotations(anchors, mode);
    VERT.forEach((spec, i) => {
      const rot = seg[i].multiply(REST[i]);
      out.push({ position: cur.clone(), rotation: rot });
      cur.add(UP.clone().applyQuaternion(rot).multiplyScalar(spec.height));
    });
    return out;
  }

  // ---------- 3D view: frontend/src/components/spine/SpineView.tsx in three r147 ----------
  // Same scene, colours, lights, shadows, pods, curve, labels and camera presets. r147 is told to use
  // physical lights and sRGB output, with hex colours converted to linear, so it lights like the frontend's r170.
  const STATUS_COLORS = { valid: 0x16a37a, stale: 0xe0911b, unavailable: 0xa3a9b0, invalid: 0xd14343 };
  const CAMERA_POSITIONS = { oblique: new T.Vector3(-0.95, 0.3, 0.95), side: new T.Vector3(-1.4, 0.05, 0), back: new T.Vector3(0, 0.05, 1.4) };
  const CURVE_COLOR = 0x2563eb, CURVE_RADIUS = 0.0032, CURVE_SEGMENTS = 160;
  const lin = (hex) => new T.Color(hex).convertSRGBToLinear();
  const hexCss = (h) => '#' + h.toString(16).padStart(6, '0');
  let view = null;
  function buildView(model) {
    VERT = model.vertebrae;
    REST = VERT.map((v) => new T.Quaternion().setFromAxisAngle(XA, v.restTiltDeg / DEG));
    const well = $('spine-well');
    const scene = new T.Scene();
    scene.background = lin(0xffffff);
    const camera = new T.PerspectiveCamera(35, 1, 0.01, 20);
    let preset = 'oblique';
    camera.position.copy(CAMERA_POSITIONS.oblique);
    const renderer = new T.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    renderer.physicallyCorrectLights = true;
    renderer.outputEncoding = T.sRGBEncoding;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = T.PCFSoftShadowMap;
    well.prepend(renderer.domElement);
    const layer = document.createElement('div'); layer.className = 'label-layer'; renderer.domElement.after(layer);
    const controls = new T.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.enablePan = false;
    controls.enableZoom = false; // the deck scrolls with the wheel; the frontend allows zoom 0.35 to 3 m
    controls.minDistance = 0.35; controls.maxDistance = 3;

    scene.add(new T.HemisphereLight(lin(0xffffff), lin(0xd8dde3), 1.6));
    const key = new T.DirectionalLight(lin(0xffffff), 1.8);
    key.position.set(-0.8, 1.6, 1.0);
    key.castShadow = true; key.shadow.mapSize.set(1024, 1024);
    Object.assign(key.shadow.camera, { left: -0.6, right: 0.6, top: 0.6, bottom: -0.6 });
    key.shadow.radius = 6;
    scene.add(key);

    const group = new T.Group(); scene.add(group);
    const floor = new T.Mesh(new T.PlaneGeometry(3, 3), new T.ShadowMaterial({ opacity: 0.08 }));
    floor.rotation.x = -Math.PI / 2; floor.position.y = -0.06; floor.receiveShadow = true;
    group.add(floor);

    const boneMat = new T.MeshStandardMaterial({ color: lin(0xe6dccb), roughness: 0.7, metalness: 0 });
    const vg = VERT.map(() => { const g = new T.Group(); group.add(g); return g; });
    const boxes = VERT.map((s) => new T.Box3(new T.Vector3(-s.width / 2, 0, -s.depth / 2), new T.Vector3(s.width / 2, s.height, s.depth / 2)));
    // vertebrae.glb (BodyParts3D, CC BY 4.0), shipped base64 in JSON; each mesh is local to its pivot
    try {
      new T.GLTFLoader().parse(b64buf(model.glb), '', (gltf) => {
        VERT.forEach((spec, i) => {
          const src = gltf.scene.getObjectByName(spec.name); let geo = null;
          if (src) src.traverse((c) => { if (!geo && c.isMesh) geo = c.geometry; });
          if (!geo) return;
          if (!geo.getAttribute('normal')) geo.computeVertexNormals();
          geo.computeBoundingBox(); boxes[i] = geo.boundingBox.clone();
          const bone = new T.Mesh(geo, boneMat); bone.castShadow = true; bone.receiveShadow = true;
          vg[i].add(bone);
        });
      }, () => {});
    } catch (e) { /* bones missing: pods and curve still show */ }

    const podGeo = new T.RoundedBoxGeometry(0.032, 0.021, 0.01, 3, 0.003);
    const axesList = [];
    const pods = ANCHORS.map((idx, k) => {
      const mat = new T.MeshStandardMaterial({ color: lin(STATUS_COLORS.unavailable), roughness: 0.4 });
      const m = new T.Mesh(podGeo, mat); m.position.fromArray(VERT[idx].pod); m.castShadow = true; vg[idx].add(m);
      const axes = new T.AxesHelper(0.05); axes.visible = false; m.add(axes); axesList.push(axes);
      // label: a CSS2D-style pill anchored at (-0.035, 0, 0.01) in pod space, centre (1, 0.5)
      const el = document.createElement('div'); el.className = 'sensor-label';
      const dot = document.createElement('span'); dot.className = 'sensor-dot';
      const tx = document.createElement('span'); tx.textContent = REGION[k];
      el.append(dot, tx); layer.appendChild(el);
      const anchor = new T.Object3D(); anchor.position.set(-0.035, 0, 0.01); m.add(anchor);
      return { mat, el, dot, anchor, status: '', px: new T.Vector3() };
    });
    const curveMat = new T.MeshStandardMaterial({ color: lin(CURVE_COLOR), emissive: lin(CURVE_COLOR), emissiveIntensity: 0.25, roughness: 0.35 });
    const curve = new T.Mesh(new T.BufferGeometry(), curveMat); curve.castShadow = true; group.add(curve);
    const offs = VERT.map((v) => new T.Vector3().fromArray(v.pod));

    const resize = () => {
      const w = well.clientWidth, h = well.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h);
      camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
    };
    new ResizeObserver(resize).observe(well); resize();

    const shown = ANCHORS.map(() => new T.Quaternion());
    const fb = new T.Box3(), pb = new T.Box3(), tgt = new T.Vector3(), sph = new T.Sphere();
    let first = true, dist = CAMERA_POSITIONS.oblique.length();
    function pose(anchors, dt, status) {
      const blend = first ? 1 : 1 - Math.exp(-dt / 0.015);
      shown.forEach((q, k) => q.slerp(anchors[k], blend));
      const P = computeSpinePose(shown, INTERP);
      P.forEach((p, i) => { vg[i].position.copy(p.position); vg[i].quaternion.copy(p.rotation); vg[i].updateMatrix(); });
      const pts = [];
      for (let i = ANCHORS[0]; i <= ANCHORS[3]; i++) pts.push(offs[i].clone().applyMatrix4(vg[i].matrix));
      curve.geometry.dispose();
      curve.geometry = new T.TubeGeometry(new T.CatmullRomCurve3(pts, false, 'centripetal'), CURVE_SEGMENTS, CURVE_RADIUS, 10, false);
      pods.forEach((p, k) => {
        if (p.status === status[k]) return;
        p.status = status[k]; const c = STATUS_COLORS[status[k]];
        p.mat.color.copy(lin(c)); p.dot.style.background = hexCss(c);
      });
      // re-centre the posed column and rest the shadow just under its lowest bone (as the frontend)
      fb.makeEmpty();
      vg.forEach((g, i) => fb.union(pb.copy(boxes[i]).applyMatrix4(g.matrix)));
      const e = first ? 1 : 1 - Math.exp(-dt / 0.35);
      if (!fb.isEmpty()) {
        fb.getCenter(tgt).multiplyScalar(-1);
        group.position.lerp(tgt, e);
        floor.position.y = fb.min.y - 0.01;
      }
      // the frontend's camera stays at the preset distance; back off only if this narrower panel would crop the column
      fb.getBoundingSphere(sph);
      const fov = (camera.fov / 2) / DEG, hfov = Math.atan(Math.tan(fov) * camera.aspect);
      const need = Math.max(CAMERA_POSITIONS[preset].length(), (sph.radius * (camera.aspect < 0.72 ? 1.3 : 1.05)) / Math.sin(Math.min(fov, hfov))); // narrow: room for the pill labels
      dist += (need - dist) * e;
      camera.position.setLength(dist);
      first = false;
    }
    function render() {
      controls.update(); renderer.render(scene, camera);
      const w = well.clientWidth, h = well.clientHeight;
      group.updateMatrixWorld();
      pods.forEach((p) => {
        p.anchor.getWorldPosition(p.px).project(camera);
        const st = p.el.style, x = (p.px.x + 1) * w / 2, y = (1 - p.px.y) * h / 2;
        st.transform = `translate(-100%, -50%) translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
        st.display = p.px.z > 1 ? 'none' : '';
        st.visibility = '';
      });
      // when the column lies down the pills can stack; the higher sensor's label wins
      const placed = [];
      for (let k = pods.length - 1; k >= 0; k--) {
        const b = pods[k].el.getBoundingClientRect();
        if (placed.some((o) => b.left < o.right && b.right > o.left && b.top < o.bottom && b.bottom > o.top)) pods[k].el.style.visibility = 'hidden'; else placed.push(b);
      }
    }
    function setPreset(name) {
      preset = name; dist = CAMERA_POSITIONS[name].length();
      camera.position.copy(CAMERA_POSITIONS[name]); controls.target.set(0, 0, 0); controls.update();
    }
    function setAxes(on) { axesList.forEach((a) => { a.visible = on; }); }
    return { pose, render, setPreset, setAxes };
  }

  // ---------- 2D traces ----------
  const canv = {};
  function fitCanvas(c) {
    const d = Math.min(devicePixelRatio || 1, 2), w = Math.round(c.clientWidth * d), h = Math.round(c.clientHeight * d);
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const ctx = c.getContext('2d'); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, w, h);
    return { ctx, w, h, d };
  }
  const WIN = 45; // seconds of history in each trace
  function trace(id, r, f, series, opts) {
    const c = canv[id] || (canv[id] = $(id)); if (!c || !c.clientWidth) return;
    const { ctx, w, h, d } = fitCanvas(c);
    const n = Math.round(WIN * r.fps), f0 = f - n;
    const ok = (i) => Number.isFinite(s0(i));
    let s0 = () => NaN;
    ctx.strokeStyle = C.rule; ctx.lineWidth = 1 * d; ctx.beginPath(); ctx.moveTo(0, h - 0.5 * d); ctx.lineTo(w, h - 0.5 * d); ctx.stroke();
    if (f < 0) return;
    series.forEach((s) => {
      s0 = (i) => s.a[i];
      const y = (v) => h - 3 * d - ((v - s.lo) / (s.hi - s.lo)) * (h - 8 * d);
      ctx.strokeStyle = s.color; ctx.lineWidth = (s.w || 1.5) * d; ctx.setLineDash(s.dash ? s.dash.map((x) => x * d) : []);
      if (s.bars) {
        ctx.fillStyle = s.color; const step = 4, bw = Math.max(1, (w / n) * step - 1 * d);
        for (let i = Math.max(0, f0); i <= f; i += step) { if (!ok(i)) continue; const x = ((i - f0) / n) * w, v = clamp(y(s.a[i]), 0, h); ctx.fillRect(x, v, bw, h - v); }
        return;
      }
      ctx.beginPath(); let started = false;
      for (let i = Math.max(0, f0); i <= f; i += 2) { if (!ok(i)) { started = false; continue; } const x = ((i - f0) / n) * w, v = clamp(y(s.a[i]), d, h - d); if (!started) { ctx.moveTo(x, v); started = true; } else ctx.lineTo(x, v); }
      ctx.stroke(); ctx.setLineDash([]);
      if (!ok(f)) return;
      if (s.hi2) { ctx.fillStyle = C.accent; ctx.strokeStyle = C.ink; ctx.lineWidth = 1 * d; ctx.beginPath(); ctx.arc(w - 4 * d, clamp(y(s.a[f]), 4 * d, h - 4 * d), 3.5 * d, 0, 7); ctx.fill(); ctx.stroke(); }
      else { ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(w - 3 * d, clamp(y(s.a[f]), d, h - d), 2.5 * d, 0, 7); ctx.fill(); }
    });
  }

  // ======================= LIVE: backend telemetry WebSocket =======================
  // Connection as frontend/src/hooks/useTelemetry.ts: ?ws= overrides the default URL; the browser sends nothing;
  // every text message is one WorkerTelemetry snapshot. Reconnect with backoff (1 s doubling to 5 s).
  const WS_URL = (() => { try { const q = new URLSearchParams(location.search).get('ws'); if (q) return q; } catch (e) { /* default */ } return 'ws://127.0.0.1:8000/ws/telemetry'; })();
  const REGIONS = ['pelvis', 'lumbar', 'thoracic', 'upper_thoracic'];
  const STALE_MS = 1000;      // no message for this long: valid sensors show as stale (frontend SILENT_AFTER_MS)
  const FALLBACK_MS = 2000;   // Auto mode falls back to the replay after this long without a message
  const LV = { sock: null, state: 'connecting', latest: null, lastAt: -1e9, count: 0, rate: 0, delay: 1000, timer: 0, retryAt: 0, ever: false };
  const liveQ = REGIONS.map(() => new T.Quaternion()), rawQ = REGIONS.map(() => null), neutral = REGIONS.map(() => null);
  const have = REGIONS.map(() => false);
  let usingSeg = false;
  const qTmp = new T.Quaternion(), upV = new T.Vector3();
  const numOr = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : NaN);

  function connect() {
    clearTimeout(LV.timer);
    LV.state = 'connecting';
    let sock;
    try { sock = new WebSocket(WS_URL); } catch (e) { LV.state = 'closed'; retry(); return; }
    LV.sock = sock;
    sock.onopen = () => { LV.state = 'open'; LV.delay = 1000; LV.ever = true; neutral.fill(null); /* reconnect: fresh neutral, fresh sequence */ };
    sock.onmessage = (ev) => {
      let m; try { m = JSON.parse(ev.data); } catch (e) { return; } // malformed: the next snapshot replaces it
      if (!m || typeof m !== 'object' || !Array.isArray(m.imus)) return;
      LV.latest = m; LV.lastAt = performance.now(); LV.count++;
      onSnapshot(m);
      hrSimUpdate(m);
    };
    sock.onerror = () => { /* onclose follows */ };
    sock.onclose = () => { if (LV.sock === sock) { LV.sock = null; LV.state = 'closed'; retry(); } };
  }
  function retry() { LV.retryAt = performance.now() + LV.delay; LV.timer = setTimeout(connect, LV.delay); LV.delay = Math.min(LV.delay * 2, 5000); }
  setInterval(() => { LV.rate = LV.count; LV.count = 0; }, 1000);

  // Segment rotations: the backend's heading-free gravity-model poses when present (frontend App.tsx), else the raw
  // segment-to-world quaternion corrected by a neutral: D = q * inverse(q_neutral) (first valid sample per sensor,
  // re-taken with "Set upright"), the same correction the replay applies to the simulated runs.
  function onSnapshot(m) {
    let seg = false;
    REGIONS.forEach((reg, k) => {
      const slot = m.imus.find((i) => i && i.region === reg);
      const sp = m.segments && m.segments[reg];
      const o = slot && slot.orientation;
      const valid = slot && slot.freshness && slot.freshness.status === 'valid';
      if (o && valid) { rawQ[k] = (rawQ[k] || new T.Quaternion()).set(o.x, o.y, o.z, o.w).normalize(); if (!neutral[k]) neutral[k] = rawQ[k].clone(); }
      if (sp && sp.rotation) {
        liveQ[k].set(sp.rotation.x, sp.rotation.y, sp.rotation.z, sp.rotation.w).normalize(); have[k] = true; seg = true;
      } else if (o && valid) {
        liveQ[k].copy(rawQ[k]).multiply(qTmp.copy(neutral[k]).invert()).normalize(); have[k] = true;
      } // otherwise hold the last pose; the pod colour shows the freshness
    });
    usingSeg = seg;
    sample(m);
  }

  function liveFresh(now) { return LV.state === 'open' && now - LV.lastAt <= STALE_MS; }
  // the stream carries a wearer: the device link is up (or simulated), or at least one IMU is valid
  function wearerStreaming(m) {
    if (!m) return false;
    if (m.connection && m.connection.esp32 && m.connection.esp32 !== 'disconnected') return true;
    return m.imus.some((i) => i && i.freshness && i.freshness.status === 'valid');
  }
  let mode = 'auto', shownSrc = '';
  function effective() {
    if (mode === 'replay') return 'replay';
    if (mode === 'live') return 'live';
    return LV.state === 'open' && performance.now() - LV.lastAt <= FALLBACK_MS && wearerStreaming(LV.latest) ? 'live' : 'replay';
  }

  // availability (protocol "Availability rules"): a group counts only when its freshness says valid
  // ---- Laptop microphone as the noise sensor (Web Audio). localhost is a secure context, so the browser asks
  // for permission once. Level: RMS of the raw input in dBFS, shown as an approximate sound level (dBFS + 100,
  // a typical laptop-mic offset; not calibrated). Overrides the vest/replay noise everywhere while on.
  const MIC = { on: false, db: NaN, ctx: null, an: null, buf: null, stream: null, timer: 0 };
  const MB = { fps: 10, n: 0, noise: new Float32Array(6000) };
  function micPush(v) { if (MB.n === MB.noise.length) { MB.noise.copyWithin(0, MB.n / 2); MB.n /= 2; } MB.noise[MB.n++] = v; }
  async function micStart() {
    try {
      MIC.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
      MIC.ctx = new (window.AudioContext || window.webkitAudioContext)();
      const src = MIC.ctx.createMediaStreamSource(MIC.stream);
      MIC.an = MIC.ctx.createAnalyser(); MIC.an.fftSize = 2048; MIC.buf = new Float32Array(MIC.an.fftSize);
      src.connect(MIC.an);
      if (MIC.ctx.state === 'suspended') MIC.ctx.resume().catch(() => {});
      MIC.on = true; MB.n = 0;
      MIC.timer = setInterval(() => {
        MIC.an.getFloatTimeDomainData(MIC.buf);
        let sum = 0, mean = 0;
        for (let i = 0; i < MIC.buf.length; i++) mean += MIC.buf[i];
        mean /= MIC.buf.length;
        for (let i = 0; i < MIC.buf.length; i++) { const x = MIC.buf[i] - mean; sum += x * x; }
        const dbfs = 10 * Math.log10(Math.max(sum / MIC.buf.length, 1e-12));
        const lvl = Math.max(30, Math.min(110, dbfs + 100));
        MIC.db = Number.isFinite(MIC.db) ? (lvl > MIC.db ? lvl : MIC.db + (lvl - MIC.db) * 0.3) : lvl;   // fast attack, slower release
        micPush(MIC.db);
      }, 100);
    } catch (e) {
      MIC.on = false;   // blocked or no microphone: keep the vest/replay noise
    }
    micUi();
  }
  function micLive() { return MIC.on && MIC.ctx && MIC.ctx.state === 'running' && Number.isFinite(MIC.db); }
  function micUi() { $('u-noise').textContent = MIC.on ? 'dB' : 'dB rel.'; }
  // Default on: ask for the microphone when the page loads. Browsers keep an AudioContext suspended until the
  // first click or key press, so the level starts with the presenter's first interaction (e.g. an arrow key).
  // If the microphone is blocked or missing, the noise row keeps the vest/replay value.
  const micResume = () => { if (MIC.ctx && MIC.ctx.state === 'suspended') MIC.ctx.resume(); };
  document.addEventListener('pointerdown', micResume, true); document.addEventListener('keydown', micResume, true);
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) micStart();

  // Simulated heart rate for live mode: the backend does not send heart_rate_bpm yet, so while the vest
  // streams, a synthetic value follows the wearer's measured trunk motion (resting ~72 bpm, rising with
  // movement, ~8 s lag). Shown and labelled as simulated; a real backend value always wins.
  const HRS = { v: 74, act: 0, q: null, t: 0 };
  function hrSimUpdate(m) {
    const now = performance.now() / 1000, k = 2;   // thoracic sensor
    const q = rawQ[k];
    if (!q || !wearerStreaming(m)) { HRS.q = null; return; }
    if (HRS.q && HRS.t) {
      const dt = Math.max(1e-3, now - HRS.t);
      const dot = Math.min(1, Math.abs(HRS.q.x * q.x + HRS.q.y * q.y + HRS.q.z * q.z + HRS.q.w * q.w));
      const rate = (2 * Math.acos(dot)) / dt;                       // rad/s
      HRS.act += (Math.min(rate, 4) - HRS.act) * Math.min(1, dt / 1.5);
      const target = 72 + Math.min(48, HRS.act * 30);
      HRS.v += (target - HRS.v) * Math.min(1, dt / 8);
    }
    HRS.q = (HRS.q || new T.Quaternion()).copy(q); HRS.t = now;
  }
  function hrSim() { const s = performance.now() / 1000; return HRS.v + 1.3 * Math.sin(s * 0.9) + 0.7 * Math.sin(s * 2.3 + 1); }

  function liveValues(m, fresh) {
    const fr = (m && m.freshness) || {};
    const ok = (key, v) => fresh && v != null && (!fr[key] || fr[key].status === 'valid');
    const env = m && ok('environment', m.environment) ? m.environment : null;
    const realHr = m && ok('heart_rate', m.heart_rate_bpm) ? numOr(m.heart_rate_bpm) : NaN;
    const simHr = !Number.isFinite(realHr) && fresh && wearerStreaming(m) && HRS.q ? hrSim() : NaN;
    return {
      hrSim: Number.isFinite(simHr),
      hr: Number.isFinite(realHr) ? realHr : simHr,
      temp: env ? numOr(env.temperature_c) : NaN, hum: env ? numOr(env.humidity_percent) : NaN,
      aqi: env ? numOr(env.air_quality_index) : NaN, eco2: env ? numOr(env.eco2_ppm) : NaN,
      tvoc: env ? numOr(env.tvoc_ppb) : NaN, noise: env ? numOr(env.noise_relative_db) : NaN,
      hrSt: !fresh ? 'none' : fr.heart_rate ? fr.heart_rate.status : m && m.heart_rate_bpm != null ? 'valid' : 'unavailable',
      envSt: !fresh ? 'none' : fr.environment ? fr.environment.status : env ? 'valid' : 'unavailable',
    };
  }

  // live history for the traces: 10 Hz samples, NaN where a value is unavailable
  const LB_CAP = 6000, LB = { fps: 10, n: 0, last: 0 };
  ['hr', 'temp', 'hum', 'eco2', 'noise'].forEach((k) => { LB[k] = new Float32Array(LB_CAP); });
  function push(v) {
    if (LB.n === LB_CAP) { const h = LB_CAP / 2; ['hr', 'temp', 'hum', 'eco2', 'noise'].forEach((k) => LB[k].copyWithin(0, h)); LB.n = h; }
    ['hr', 'temp', 'hum', 'eco2', 'noise'].forEach((k) => { LB[k][LB.n] = v[k]; }); LB.n++;
  }
  function sample(m) { const now = performance.now(); if (now - LB.last < 100) return; LB.last = now; push(liveValues(m, true)); }
  setInterval(() => { if (LV.ever && performance.now() - LB.last > 1000) { LB.last = performance.now(); push(liveValues(null, false)); } }, 250);

  const FALL = { normal: ['ok', 'No fall detected'], dismissed: ['ok', 'Alert dismissed'], suspected: ['watch', 'Suspected fall, confirming'],
    near_fall: ['watch', 'Near fall logged, no alert'], man_down_warning: ['watch', 'Not moving, checking in'],
    fall_confirmed: ['alert', 'Fall confirmed. Supervisor alerted'], man_down_critical: ['alert', 'Man down. Supervisor alerted'] };
  const ZONE = { neutral: 'Neutral posture', moderate: 'Moderate bend', high: 'Deep bend', lying: 'Lying down' };
  const ST_WORD = { valid: 'live', stale: 'stale', unavailable: '—', invalid: 'invalid' };

  function updateBadge(eff) {
    const now = performance.now(), fresh = liveFresh(now), m = LV.latest, b = $('src-badge');
    let kind, text;
    if (eff === 'live' && fresh && m && !wearerStreaming(m)) {
      kind = 'wait'; text = 'Live: backend up, vest not connected';
    } else if (eff === 'live' && fresh && m) {
      const src = m.connection && m.connection.source;
      // a recording played through the backend (TELEMETRY_SOURCE=replay, or the detection hub's demo scenarios)
      const recorded = src === 'replay' || !!m.replay || (m.demo && m.demo.live === false && m.demo.phase !== 'live');
      kind = src === 'hardware' && !recorded ? 'vest' : 'backend';
      text = recorded ? 'Live: backend replay' : src === 'hardware' ? 'Live: vest' : src === 'simulated' ? 'Live: backend simulated' : `Live: ${src || 'backend'}`;
    } else if (eff === 'live') {
      kind = 'wait'; text = LV.state === 'open' ? 'Live: no data from backend' : 'Live: waiting for backend';
    } else {
      kind = 'replay';
      text = fresh && wearerStreaming(m) ? 'Simulated replay (backend live)' : LV.state === 'open' ? 'Replay: backend up, vest not connected' : 'Offline: simulated replay';
    }
    if (b.dataset.kind !== kind) b.dataset.kind = kind;
    $('src-text').textContent = text;
    b.title = `${WS_URL}: ${LV.state === 'open' ? `connected, ${LV.rate} msg/s` : LV.state === 'connecting' ? 'connecting' : `not reachable, retrying in ${Math.max(0, Math.ceil((LV.retryAt - now) / 1000))} s`}`;
  }

  function applySource(eff) {
    shownSrc = eff; SEC.dataset.src = eff; lastUi = 0;
    $('pods-c3').textContent = eff === 'live' ? 'Data' : 'Peak';
    if (eff !== 'live') {
      zeroBtn.hidden = true;
      $('spv-replay').className = 'pill replay';
      els.rows.forEach((row) => { row.dataset.s = 'valid'; });
      ['st-hr', 'st-env', 'st-aq', 'st-noise'].forEach((id) => { $(id).textContent = ''; });
      document.querySelectorAll('#live .sen').forEach((x) => x.removeAttribute('data-off'));
      $('live-note').textContent = 'Simulated run. Other sensors synthetic. Not a medical device.';
    }
  }

  let lastTrace = 0;
  function liveFrame(dt, forceUi) {
    const now = performance.now(), fresh = liveFresh(now), m = LV.latest;
    const status = REGIONS.map((reg) => {
      const slot = m && m.imus.find((i) => i && i.region === reg);
      let s = (slot && slot.freshness && slot.freshness.status) || 'unavailable';
      if (!STATUS_COLORS[s]) s = 'unavailable';
      if (!fresh && s === 'valid') s = 'stale';
      return s;
    });
    if (view) { view.pose(liveQ, dt, status); view.render(); }
    if (forceUi || now - lastUi > 120) { lastUi = now; liveUi(m, fresh, status); }
    // the traces change at 10 Hz (LB samples); redrawing four canvases every frame forced a layout per canvas
    // and cost the spine frame rate, so they redraw at most every 100 ms
    if (!forceUi && now - lastTrace < 100) return;
    lastTrace = now;
    const L = LB, f = L.n - 1;
    trace('c-hr', L, f, [{ a: L.hr, lo: 55, hi: 170, color: C.ink, w: 2, hi2: true }]);
    trace('c-env', L, f, [{ a: L.hum, lo: 30, hi: 70, color: C.ink3, dash: [4, 3] }, { a: L.temp, lo: 10, hi: 35, color: C.ink, w: 2, hi2: true }]);
    trace('c-aq', L, f, [{ a: L.eco2, lo: 400, hi: 1200, color: C.ink, w: 2, hi2: true }]);
    if (micLive()) trace('c-noise', MB, MB.n - 1, [{ a: MB.noise, lo: 30, hi: 100, color: C.ink2, bars: true }]);
    else trace('c-noise', L, f, [{ a: L.noise, lo: 40, hi: 95, color: C.ink2, bars: true }]);
  }

  function liveUi(m, fresh, status) {
    updateBadge('live');
    // fall / posture status
    let level = 'ok', head, sub;
    const fs = fresh && m && m.fall_state && typeof m.fall_state === 'object' ? m.fall_state : null;
    const alerts = fresh && m && Array.isArray(m.alerts) ? m.alerts : [];
    const tp = fresh && m && m.trunk_posture && m.trunk_posture.trunk ? m.trunk_posture : null;
    const zoneText = tp ? ZONE[tp.zone] || tp.zone : 'Spine sensors streaming';
    if (!fresh) {
      head = LV.state === 'open' ? 'Backend connected, no data' : 'Waiting for the backend';
      sub = LV.state === 'connecting' ? `Connecting to ${WS_URL}` : LV.state === 'open' ? 'No snapshot in the last second' : `Not reachable, retrying in ${Math.max(0, Math.ceil((LV.retryAt - performance.now()) / 1000))} s`;
    } else if (fs) {
      const e = FALL[fs.state] || ['watch', fs.label || fs.state];
      level = e[0]; head = e[1];
      const conf = typeof fs.confidence === 'number' ? ` · ${Math.round(fs.confidence * 100)}% confidence` : '';
      sub = (fs.label && fs.label !== head ? fs.label : zoneText) + conf;
    } else if (alerts.length) {
      level = 'watch'; head = 'Posture warning';
      sub = alerts[0].summary + (alerts[0].sustained_seconds ? ` for ${alerts[0].sustained_seconds.toFixed(0)} s` : '');
    } else {
      head = 'No posture warning'; sub = `${zoneText}. No fall detector on this stream`;
    }
    if (els.status.dataset.level !== level) els.status.dataset.level = level;
    els.status.firstElementChild.textContent = head;
    els.act.textContent = sub;
    // trunk lean: backend trunk_posture when present, else the upper-thoracic segment's tilt
    let tr = NaN, lying = false;
    if (tp && Number.isFinite(tp.trunk.flexion_deg)) {
      // signed like the replay: tilt from vertical, + when the lean is forward
      tr = Number.isFinite(tp.trunk.inclination_deg) ? (tp.trunk.flexion_deg >= 0 ? 1 : -1) * tp.trunk.inclination_deg : tp.trunk.flexion_deg;
      lying = !!tp.lying;
    } else if (have[3]) {
      upV.set(0, 1, 0).applyQuaternion(liveQ[3]);
      tr = (upV.z <= 0 ? 1 : -1) * Math.acos(clamp(upV.y, -1, 1)) * DEG; // +Z is posterior: forward bend tips up toward -Z
    }
    els.trunk.textContent = Number.isFinite(tr) ? Math.round(Math.abs(tr)) + '°' : '—';
    els.trunkCap.textContent = !Number.isFinite(tr) ? 'trunk lean, no data' : lying ? 'lying down'
      : Math.abs(tr) >= 60 ? `trunk ${tr >= 0 ? 'forward' : 'back'}, severe bend` : Math.abs(tr) >= 20 ? `trunk ${tr >= 0 ? 'forward' : 'back'}, bent` : 'trunk lean';
    // per-sensor rows
    els.rows.forEach((row) => {
      const k = +row.dataset.k, c = row.children, sp = fresh && m && m.segments && m.segments[REGIONS[k]];
      let tilt = NaN;
      if (sp && Number.isFinite(sp.inclination_deg)) tilt = sp.inclination_deg;
      else if (have[k]) { upV.set(0, 1, 0).applyQuaternion(liveQ[k]); tilt = Math.acos(clamp(upV.y, -1, 1)) * DEG; }
      c[1].textContent = Number.isFinite(tilt) ? Math.round(tilt) + '°' : '—';
      c[2].textContent = ST_WORD[status[k]] || status[k];
      row.dataset.s = status[k] === 'valid' ? 'valid' : status[k] === 'stale' ? 'stale' : 'unavailable';
    });
    // spine panel pill
    const pill = $('spv-replay');
    if (fresh && m && wearerStreaming(m)) {
      pill.className = 'pill ok';
      pill.textContent = `Live \u00b7 ${LV.rate} msg/s` + (usingSeg ? ' \u00b7 gravity model' : have.some(Boolean) ? ' \u00b7 client neutral' : '');
      pill.title = `sequence ${m.sequence}`;
    } else {
      pill.className = 'pill wait';
      pill.textContent = !fresh ? (LV.state === 'open' ? 'Live \u00b7 no data' : 'Live \u00b7 connecting') : 'Live \u00b7 vest not connected';
    }
    zeroBtn.hidden = usingSeg || !rawQ.some(Boolean);
    // right column: environment / heart rate per the availability rules
    const v = liveValues(m, fresh);
    const setV = (id, x, dp) => { $(id).textContent = Number.isFinite(x) ? x.toFixed(dp) : '—'; };
    const why = (st) => (st === 'none' ? 'no data' : st === 'stale' ? 'stale' : st === 'unavailable' ? 'not in this stream' : '');
    setV('v-hr', v.hr, 0);
    setV('v-temp', v.temp, 1); setV('v-hum', v.hum, 0);
    const aqi = Number.isFinite(v.aqi) ? Math.round(v.aqi) : NaN;
    setV('v-aqi', aqi, 0);
    els.aqm.forEach((x, i) => { if (Number.isFinite(aqi) && i < aqi) x.setAttribute('data-on', ''); else x.removeAttribute('data-on'); });
    setV('v-eco2', v.eco2, 0); setV('v-tvoc', v.tvoc, 0); setV('v-noise', micLive() ? MIC.db : v.noise, 0);
    const card = (id, st, val) => { $(id).textContent = Number.isFinite(val) ? '' : why(st) || 'no reading'; const li = $(id).closest('.sen'); if (Number.isFinite(val)) li.removeAttribute('data-off'); else li.setAttribute('data-off', ''); };
    card('st-hr', v.hrSt, v.hr); card('st-env', v.envSt, v.temp); card('st-aq', v.envSt, Number.isFinite(aqi) ? aqi : v.eco2); card('st-noise', micLive() ? 'valid' : v.envSt, micLive() ? MIC.db : v.noise);
    $('live-note').textContent = `Live from ${WS_URL}${m && m.connection ? ` (source ${m.connection.source}, ESP32 ${m.connection.esp32})` : ''}. ${v.hrSim ? ' Heart rate is simulated in this demo.' : ''} Prototype, not a medical device.`;
  }

  // ---------- state, clock, UI ----------
  let RUNS = null, run = null, t = 0, playing = !reduce.matches, active = false, raf = 0, last = 0, lastUi = 0, loading = null;
  const els = { status: $('live-status'), act: $('live-activity'), trunk: $('live-trunk'), trunkCap: $('live-trunk-cap'), time: $('live-time'),
    head: $('live-head'), loop: $('live-loop'), ticks: $('live-ticks'), track: $('live-track'), play: $('live-play'),
    rows: [...document.querySelectorAll('#pod-rows tr')], aqm: [...document.querySelectorAll('#aq-meter i')] };
  const anchorsQ = ANCHORS.map(() => new T.Quaternion()), qa = new T.Quaternion(), qb = new T.Quaternion();

  function setRun(id) {
    run = RUNS.find((r) => r.id === id) || RUNS[0];
    document.querySelectorAll('#live .runs button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.run === run.id)));
    // reduced motion: open on the most telling moment instead of playing from the start
    const alertEv = run.events.find((e) => e.alert) || run.events[run.events.length - 1];
    t = playing ? run.loopStart : alertEv ? Math.min(run.duration, alertEv.t + 1) : run.loopStart + 10;
    const D = run.duration;
    els.loop.style.left = (run.loopStart / D) * 100 + '%'; els.loop.style.right = '0';
    els.ticks.innerHTML = '';
    run.events.forEach((e) => { const i = document.createElement('i'); i.style.left = (e.t / D) * 100 + '%'; if (e.alert) i.dataset.alert = ''; i.title = e.kind.replace('_', ' '); els.ticks.appendChild(i); });
    frame(0, true);
  }

  function frame(dt, forceUi) {
    const eff = effective();
    if (eff !== shownSrc) { applySource(eff); forceUi = true; }
    if (eff === 'live') { liveFrame(dt, forceUi); return; }
    if (!run) return;
    const fp = clamp(t * run.fps, 0, run.F - 1), f = Math.floor(fp), f1 = Math.min(f + 1, run.F - 1), a = fp - f;
    for (let k = 0; k < 4; k++) {
      const o = f * 16 + k * 4, o1 = f1 * 16 + k * 4;
      qa.set(run.D[o], run.D[o + 1], run.D[o + 2], run.D[o + 3]); qb.set(run.D[o1], run.D[o1 + 1], run.D[o1 + 2], run.D[o1 + 3]);
      anchorsQ[k].copy(qa).slerp(qb, a);
    }
    const stale = run.age[f] >= 200;
    const s = stale ? 'stale' : 'valid';
    if (view) { view.pose(anchorsQ, dt, [s, s, s, s]); view.render(); }
    const now = performance.now();
    if (forceUi || now - lastUi > 120) { lastUi = now; ui(f, stale); }
    traces(f);
  }

  function ui(f, stale) {
    const r = run, st = detectorState(r, t), ph = r.phase[f];
    if (els.status.dataset.level !== st.level) els.status.dataset.level = st.level;
    els.status.firstElementChild.textContent = st.head;
    const doing = ph && PHASE_LABEL[ph] && ph !== 1 ? PHASE_LABEL[ph] : ph === 1 ? 'Calibrating the neutral posture' : activityAt(r, t);
    els.act.textContent = doing;
    const tr = r.trunk[f], kind = PHASE_KIND[ph];
    els.trunk.textContent = Math.round(Math.abs(tr)) + '\u00b0';
    els.trunkCap.textContent = kind === 'fall' || kind === 'lying' ? 'on the ground'
      : Math.abs(tr) >= 60 ? `trunk ${tr >= 0 ? 'forward' : 'back'}, severe bend` : Math.abs(tr) >= 20 ? `trunk ${tr >= 0 ? 'forward' : 'back'}, bent` : 'trunk lean';
    els.rows.forEach((row) => {
      const k = +row.dataset.k, c = row.children;
      c[1].textContent = Math.round(r.tilt[f * 4 + k]) + '\u00b0';
      c[2].textContent = r.acc[f * 4 + k].toFixed(2) + ' g';
      row.dataset.s = stale ? 'stale' : 'valid';
    });
    els.head.style.left = (t / r.duration) * 100 + '%';
    els.time.textContent = `${t.toFixed(1)} of ${r.duration.toFixed(0)} s`;
    $('spv-replay').textContent = `Replay ${t.toFixed(0)}s / ${r.duration.toFixed(0)}s \u00b7 ${doing}`;
    $('v-hr').textContent = Math.round(r.hr[f]);
    $('v-temp').textContent = r.temp[f].toFixed(1);
    $('v-hum').textContent = Math.round(r.hum[f]);
    const aqi = aqiOf(r.eco2[f]);
    $('v-aqi').textContent = aqi;
    els.aqm.forEach((m, i) => { if (i < aqi) m.setAttribute('data-on', ''); else m.removeAttribute('data-on'); });
    $('v-eco2').textContent = Math.round(r.eco2[f]);
    $('v-tvoc').textContent = Math.round(r.tvoc[f]);
    $('v-noise').textContent = Math.round(micLive() ? MIC.db : r.noise[f]);
    updateBadge('replay');
  }

  function traces(f) {
    const r = run;
    trace('c-hr', r, f, [{ a: r.hr, lo: 55, hi: 170, color: C.ink, w: 2, hi2: true }], { marks: true });
    trace('c-env', r, f, [{ a: r.hum, lo: 30, hi: 70, color: C.ink3, dash: [4, 3] }, { a: r.temp, lo: 10, hi: 35, color: C.ink, w: 2, hi2: true }]);
    trace('c-aq', r, f, [{ a: r.eco2, lo: 400, hi: 1200, color: C.ink, w: 2, hi2: true }]);
    if (micLive()) trace('c-noise', MB, MB.n - 1, [{ a: MB.noise, lo: 30, hi: 100, color: C.ink2, bars: true }]);
    else trace('c-noise', r, f, [{ a: r.noise, lo: 40, hi: 95, color: C.ink2, bars: true }]);
  }

  function loop(now) {
    raf = 0;
    if (!active) return;
    const dt = last ? Math.min(0.1, (now - last) / 1000) : 0; last = now;
    if (playing && run && shownSrc !== 'live') { t += dt; if (t >= run.duration) t = run.loopStart; }
    frame(dt);
    raf = requestAnimationFrame(loop);
  }
  function setActive(on) {
    on = on && !document.hidden;
    if (on === active) return;
    active = on; last = 0;
    if (on) { ensureLoaded(); if (!raf) raf = requestAnimationFrame(loop); } else if (raf) { cancelAnimationFrame(raf); raf = 0; }
  }

  function ensureLoaded() {
    if (loading) return loading;
    els.act.textContent = 'Loading the run';
    loading = Promise.all([fetch('data/live-runs.json').then((x) => x.json()), fetch('data/spine-model.json').then((x) => x.json())])
      .then(([runs, model]) => {
        RUNS = runs.runs.map(decodeRun);
        try { view = buildView(model); } catch (e) { view = null; window.__liveErr = String(e && e.stack || e); els.act.textContent = '3D view unavailable in this browser'; }
        setRun(RUNS[0].id);
      })
      .catch(() => { els.act.textContent = 'Could not load the run data'; });
    return loading;
  }

  // ---------- wiring ----------
  // spine panel controls (as the frontend's footer: view presets, interpolation, sensor axes)
  const spv = $('spine-well');
  spv.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => {
    spv.querySelectorAll('[data-view]').forEach((x) => x.classList.toggle('active', x === b));
    if (view) view.setPreset(b.dataset.view);
  }));
  spv.querySelectorAll('[data-interp]').forEach((b) => b.addEventListener('click', () => {
    spv.querySelectorAll('[data-interp]').forEach((x) => x.classList.toggle('active', x === b));
    INTERP = b.dataset.interp; frame(0, true);
  }));
  const axBtn = $('spv-axes');
  axBtn.addEventListener('click', () => { const on = !axBtn.classList.contains('active'); axBtn.classList.toggle('active', on); axBtn.setAttribute('aria-pressed', String(on)); if (view) view.setAxes(on); });
  axBtn.setAttribute('aria-pressed', 'false');
  document.querySelectorAll('#live .runs button').forEach((b) => b.addEventListener('click', () => { if (RUNS) setRun(b.dataset.run); }));
  function setPlaying(p) { playing = p; els.play.textContent = p ? 'Pause' : 'Play'; els.play.setAttribute('aria-pressed', String(p)); }
  setPlaying(playing);
  els.play.addEventListener('click', () => setPlaying(!playing));
  els.track.addEventListener('click', (e) => {
    if (!run) return; const b = els.track.getBoundingClientRect();
    t = clamp(((e.clientX - b.left) / b.width) * run.duration, 0, run.duration); frame(0, true);
  });
  // data source: Auto (live when the backend streams) / Live / Replay; ?mode= or the L key; remembered per browser
  const zeroBtn = $('spv-zero');
  zeroBtn.addEventListener('click', () => { rawQ.forEach((q, k) => { if (q) neutral[k] = q.clone(); }); if (LV.latest) onSnapshot(LV.latest); });
  const modeBtns = [...document.querySelectorAll('#live .src-mode button')];
  function setMode(next) {
    mode = ['auto', 'live', 'replay'].includes(next) ? next : 'auto';
    modeBtns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
    try { localStorage.setItem('insite-pitch.source', mode); } catch (e) { /* storage blocked */ }
    frame(0, true);
  }
  modeBtns.forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
  document.addEventListener('keydown', (e) => {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || (e.key !== 'l' && e.key !== 'L')) return;
    const tag = (e.target.tagName || '').toLowerCase(); if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
    setMode(mode === 'auto' ? 'live' : mode === 'live' ? 'replay' : 'auto');
  });
  {
    let m0 = null;
    try { m0 = new URLSearchParams(location.search).get('mode'); } catch (e) { /* none */ }
    if (!m0) { try { m0 = localStorage.getItem('insite-pitch.source'); } catch (e) { /* storage blocked */ } }
    mode = ['auto', 'live', 'replay'].includes(m0) ? m0 : 'auto';
    modeBtns.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === mode)));
  }
  // ---- Connect vest: ask the backend to scan for the strip over BLE (POST /device/connect), like the
  // spine frontend's DeviceConnect. Default format "both" (quaternions + raw). With the default socket the
  // HTTP calls go through the vite proxy (/backend -> backend), so no CORS is needed; with ?ws= they go to
  // that host directly. ?format=quaternion|both|raw_fusion overrides the default choice.
  const API = (() => {
    try {
      if (new URLSearchParams(location.search).get('ws')) {
        const u = new URL(WS_URL); u.protocol = u.protocol === 'wss:' ? 'https:' : 'http:'; return u.origin;
      }
    } catch (e) { /* fall through */ }
    return '/backend';
  })();
  const vestBtn = $('vest-btn'), vestFmt = $('vest-format'), vestMsg = $('vest-msg');
  const VEST = { busy: false, diag: null, problem: null };
  try { const f = new URLSearchParams(location.search).get('format'); if (['both', 'quaternion', 'raw_fusion'].includes(f)) vestFmt.value = f; } catch (e) { /* default */ }
  function vestActive() {
    const m = LV.latest, fresh = liveFresh(performance.now());
    const streaming = fresh && m && m.connection && m.connection.esp32 === 'connected';
    const st = VEST.diag && VEST.diag.state;
    return streaming || ['connecting', 'connected', 'subscribed'].includes(st);
  }
  function vestUi() {
    const m = LV.latest, fresh = liveFresh(performance.now());
    const src = fresh && m && m.connection && m.connection.source;
    const sim = src === 'simulated' || src === 'replay';
    const on = vestActive();
    vestBtn.dataset.state = on ? 'on' : 'off';
    vestBtn.textContent = VEST.busy ? (on ? 'Disconnecting…' : 'Scanning for vest…') : on ? 'Disconnect vest' : 'Connect vest';
    vestBtn.disabled = VEST.busy || LV.state !== 'open' || sim;
    vestFmt.disabled = VEST.busy || on || sim;
    let msg = '', kind = '';
    if (LV.state !== 'open') msg = 'Start the backend to connect the vest.';
    else if (sim) msg = `Backend is running a ${src} source. Restart it with TELEMETRY_SOURCE=manual to connect the vest.`;
    else if (VEST.problem) { msg = VEST.problem; kind = 'err'; }
    else if (VEST.diag && VEST.diag.state === 'error' && VEST.diag.last_error) { msg = `Vest error: ${VEST.diag.last_error}`; kind = 'err'; }
    else if (VEST.diag && ['connecting', 'connected'].includes(VEST.diag.state) && !(m && m.connection && m.connection.esp32 === 'connected')) msg = 'Connecting to the vest…';
    if (vestMsg.textContent !== msg) vestMsg.textContent = msg;
    if ((vestMsg.dataset.kind || '') !== kind) vestMsg.dataset.kind = kind;
  }
  async function vestCall(path) {
    VEST.busy = true; VEST.problem = null; vestUi();
    try {
      const r = await fetch(API + path, { method: 'POST' });
      if (!r.ok) {
        const body = await r.json().catch(() => null);
        VEST.problem = (body && body.detail) || `Backend answered HTTP ${r.status}`;
      } else if (path.startsWith('/device/connect')) {
        setMode('auto'); // switch to the live stream as soon as data arrives
      }
    } catch (e) {
      VEST.problem = API === '/backend' ? 'Backend unreachable on the proxy (is it running on port 8000?)' : `Backend unreachable at ${API}`;
    } finally {
      VEST.busy = false; pollDiag(); vestUi();
    }
  }
  async function pollDiag() {
    if (LV.state !== 'open') { VEST.diag = null; return; }
    try {
      const r = await fetch(`${API}/diagnostics`);
      if (r.ok) { const b = await r.json(); VEST.diag = { state: b.state, last_error: b.last_error || null }; }
    } catch (e) { VEST.diag = null; }
  }
  vestBtn.addEventListener('click', () => {
    if (VEST.busy) return;
    if (vestActive()) vestCall('/device/disconnect');
    else vestCall(`/device/connect?device_format=${encodeURIComponent(vestFmt.value)}`);
  });
  setInterval(() => { pollDiag().then(vestUi); }, 1000);
  vestUi();

  connect();
  window.__live = { state: () => ({ mode, shown: shownSrc, eff: effective(), active, raf: !!raf, socket: LV.state, run: run && run.id, view: !!view, t }) };
  // the badge keeps updating while the deck sits on another page
  setInterval(() => { if (!active) updateBadge(effective()); }, 500);
  const io = new IntersectionObserver((es) => es.forEach((en) => setActive(en.isIntersecting)), { threshold: 0.05 });
  io.observe(SEC);
  document.addEventListener('deck:page', (e) => { if (e.detail && e.detail.id === 'live') setActive(true); });
  document.addEventListener('visibilitychange', () => setActive(!document.hidden && SEC.getBoundingClientRect().top < innerHeight && SEC.getBoundingClientRect().bottom > 0));
  // warm the data shortly before the reader reaches the page
  new IntersectionObserver((es, o) => { if (es.some((x) => x.isIntersecting)) { ensureLoaded(); o.disconnect(); } }, { rootMargin: '100% 0px' }).observe(SEC);
})();
