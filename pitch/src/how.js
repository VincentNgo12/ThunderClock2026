/* Page 3 (#how): the Anny body (NAVER LABS Europe, Apache-2.0; mesh, rig, weights from MakeHuman/MPFB2, CC0) in a neutral pose,
   wearing a vest cut from its own torso skin, with the sensors on the vest and leader-line callouts.
   Assembled by deck_build/make_how_js.py from sim_head.js, anny_block.js (quaternions, poseAnny, annySkin), sim_anny.js and how_tail.js.
   Do not hand-edit; edit how_tail.js and re-run. */
(function () {
  'use strict';
  const SEC = document.getElementById('how');
  if (!SEC || !window.THREE) return;
  const THREE = window.THREE;
  const DEG = 180 / Math.PI;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const V = (x = 0, y = 0, z = 0) => new THREE.Vector3(x, y, z);
  function b64buf(s) { const bin = atob(s); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u.buffer; }
  const DATA = 'data/';
  const J = { hip_l: 0, hip_r: 1, knee_l: 2, knee_r: 3, ankle_l: 4, ankle_r: 5, heel_l: 6, heel_r: 7, toe_l: 8, toe_r: 9,
    shoulder_l: 10, shoulder_r: 11, elbow_l: 12, elbow_r: 13, hand_l: 14, hand_r: 15, neck: 16, head: 17 };
  const ANNY_Q_BONES = ['upperleg01.L', 'upperleg02.L', 'lowerleg01.L', 'lowerleg02.L', 'foot.L', 'upperleg01.R', 'upperleg02.R', 'lowerleg01.R', 'lowerleg02.R', 'foot.R',
    'clavicle.L', 'shoulder01.L', 'upperarm01.L', 'upperarm02.L', 'lowerarm01.L', 'lowerarm02.L', 'wrist.L', 'clavicle.R', 'shoulder01.R', 'upperarm01.R', 'upperarm02.R',
    'lowerarm01.R', 'lowerarm02.R', 'wrist.R', 'neck01', 'neck02', 'neck03', 'head', 'eye.L', 'eye.R'];
  const ANNY_FING_CH = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
  const ANNY_CARRIERS = [['root', 0], ['pelvis.L', 0], ['pelvis.R', 0], ['spine05', 0], ['spine03', 1], ['spine02', 2], ['spine01', 3]];
  const ANNY_CARRIER_LOCALS = ['pelvis.L', 'pelvis.R', 'spine05', 'spine04', 'spine03', 'spine02', 'spine01'];
  let ANNY = null;

// ---- quaternion helpers on flat arrays (xyzw, Hamilton; the same arithmetic as frames.py, in doubles) ----
function aqMul(o, oo, a, ao, b, bo) {
  const ax = a[ao], ay = a[ao + 1], az = a[ao + 2], aw = a[ao + 3], bx = b[bo], by = b[bo + 1], bz = b[bo + 2], bw = b[bo + 3];
  o[oo] = aw * bx + ax * bw + ay * bz - az * by;
  o[oo + 1] = aw * by - ax * bz + ay * bw + az * bx;
  o[oo + 2] = aw * bz + ax * by - ay * bx + az * bw;
  o[oo + 3] = aw * bw - ax * bx - ay * by - az * bz;
}
function aqCMul(o, oo, a, ao, b, bo) {   // conj(a) (x) b
  const ax = -a[ao], ay = -a[ao + 1], az = -a[ao + 2], aw = a[ao + 3], bx = b[bo], by = b[bo + 1], bz = b[bo + 2], bw = b[bo + 3];
  o[oo] = aw * bx + ax * bw + ay * bz - az * by;
  o[oo + 1] = aw * by - ax * bz + ay * bw + az * bx;
  o[oo + 2] = aw * bz + ax * by - ay * bx + az * bw;
  o[oo + 3] = aw * bw - ax * bx - ay * by - az * bz;
}
function aqSlerp(o, oo, a, ao, b, bo, u) {   // shortest arc, normalised output (frames.qslerp)
  const ax = a[ao], ay = a[ao + 1], az = a[ao + 2], aw = a[ao + 3];
  let bx = b[bo], by = b[bo + 1], bz = b[bo + 2], bw = b[bo + 3];
  let d = ax * bx + ay * by + az * bz + aw * bw;
  if (d < 0) { bx = -bx; by = -by; bz = -bz; bw = -bw; d = -d; }
  const th = Math.acos(Math.min(1, d)), s = Math.sin(th);
  let wa, wb;
  if (s < 1e-9) { wa = 1 - u; wb = u; } else { wa = Math.sin((1 - u) * th) / s; wb = Math.sin(u * th) / s; }
  const x = wa * ax + wb * bx, y = wa * ay + wb * by, z = wa * az + wb * bz, w = wa * aw + wb * bw, n = Math.sqrt(x * x + y * y + z * z + w * w);
  o[oo] = x / n; o[oo + 1] = y / n; o[oo + 2] = z / n; o[oo + 3] = w / n;
}
function aqAxis(o, oo, ax, axo, ang) { const s = Math.sin(ang / 2); o[oo] = ax[axo] * s; o[oo + 1] = ax[axo + 1] * s; o[oo + 2] = ax[axo + 2] * s; o[oo + 3] = Math.cos(ang / 2); }
function aqNormQ(o, oo, a, ao, k) {   // o = normalize(a / k)
  const x = a[ao] / k, y = a[ao + 1] / k, z = a[ao + 2] / k, w = a[ao + 3] / k, n = Math.sqrt(x * x + y * y + z * z + w * w);
  o[oo] = x / n; o[oo + 1] = y / n; o[oo + 2] = z / n; o[oo + 3] = w / n;
}
function aqAngleDeg(a, ao, b, bo) {   // rotation angle between two unit quaternions (robust near 0)
  const ax = a[ao], ay = a[ao + 1], az = a[ao + 2], aw = a[ao + 3], bx = -b[bo], by = -b[bo + 1], bz = -b[bo + 2], bw = b[bo + 3];
  const x = aw * bx + ax * bw + ay * bz - az * by, y = aw * by - ax * bz + ay * bw + az * bx, z = aw * bz + ax * by - ay * bx + az * bw, w = aw * bw - ax * bx - ay * by - az * bz;
  return 2 * Math.atan2(Math.sqrt(x * x + y * y + z * z), Math.abs(w)) * DEG;
}


const _aq = new Float64Array(8);
function poseAnny(h, f) {
  const A = ANNY, nb = A.nb, L = h._L, Wq = h._Wq, Wp = h._Wp, S = h._S, D = h.D, par = A.parent;
  // 1. D_k = Ry(-hCal) (x) q_k; carriers W_b = D_k (x) K_b; spine04 = slerp(D_0, D_1, alpha) (x) K.
  if (f.neutral) for (let k = 0; k < 4; k++) { D[4 * k] = 0; D[4 * k + 1] = 0; D[4 * k + 2] = 0; D[4 * k + 3] = 1; }
  else { _aq[0] = 0; _aq[1] = Math.sin(-f.hcal / 2); _aq[2] = 0; _aq[3] = Math.cos(-f.hcal / 2); for (let k = 0; k < 4; k++) aqMul(D, 4 * k, _aq, 0, f.segq, 4 * k); }
  L.set(A.rlq);
  for (const c of A.carriers) aqMul(Wq, 4 * c.b, D, 4 * c.k, A.K, 4 * c.b);
  aqSlerp(_aq, 0, D, 0, D, 4, A.alpha); aqMul(Wq, 4 * A.filler, _aq, 0, A.K, 4 * A.filler);
  L[0] = Wq[0]; L[1] = Wq[1]; L[2] = Wq[2]; L[3] = Wq[3];
  for (let i = 0; i < A.carrierLocals.length; i++) { const b = A.carrierLocals[i]; aqCMul(L, 4 * b, Wq, 4 * par[b], Wq, 4 * b); }
  // 2. the 30 animated locals, fingers (base (x) aa(axis, angle) [(x) aa(axis2, abd)]), toes (restLocal (x) aa(axis, mtp)).
  for (let j = 0; j < 30; j++) { const o = 4 * A.qb[j]; L[o] = f.q[4 * j]; L[o + 1] = f.q[4 * j + 1]; L[o + 2] = f.q[4 * j + 2]; L[o + 3] = f.q[4 * j + 3]; }
  for (let hd = 0; hd < 2; hd++) for (let j = 0; j < 15; j++) {
    const i = 15 * hd + j, o = 4 * A.fb[i];
    aqAxis(_aq, 0, A.faxis, 3 * i, f.fing[16 * hd + ANNY_FING_CH[j]]); aqMul(L, o, A.fbase, 4 * i, _aq, 0);
    if (j === 0) { aqAxis(_aq, 4, A.fax2, 3 * hd, f.fing[16 * hd + 1]); aqMul(L, o, L, o, _aq, 4); }
  }
  for (let j = 0; j < 10; j++) { const o = 4 * A.tb[j]; aqAxis(_aq, 0, A.taxis, 3 * j, f.mtp[j < 5 ? 0 : 1]); aqMul(L, o, A.rlq, o, _aq, 0); }
  annySkin(h, f.root);
  // HUD: trunk from upright = acos((D_3 (x) y) . y)
  h.trunkDeg = Math.acos(clamp(1 - 2 * (D[12] * D[12] + D[14] * D[14]), -1, 1)) * DEG;
  // 5-7. pods, gel pads, items, strap
  if (h.anatHook && h.anatHook.pose) h.anatHook.pose(h);   // 9. anatomy palette (spec 18.5), when attached
}
// Steps 3-4 from the locals in h._L and the root position (the anatomy self-test poses enter here directly).
function annySkin(h, root) {
  const A = ANNY, nb = A.nb, L = h._L, Wq = h._Wq, Wp = h._Wp, S = h._S, par = A.parent;
  // 3. FK in bone order, then S_b = [R(wq_b) | wp_b] . invBind_b.
  Wq[0] = L[0]; Wq[1] = L[1]; Wq[2] = L[2]; Wq[3] = L[3]; Wp[0] = root[0]; Wp[1] = root[1]; Wp[2] = root[2];
  const rlp = A.rlp, ib = A.ib;
  for (let c = 1; c < nb; c++) {
    const p = par[c];
    aqMul(Wq, 4 * c, Wq, 4 * p, L, 4 * c);
    const qx = Wq[4 * p], qy = Wq[4 * p + 1], qz = Wq[4 * p + 2], qw = Wq[4 * p + 3], vx = rlp[3 * c], vy = rlp[3 * c + 1], vz = rlp[3 * c + 2];
    const tx = 2 * (qy * vz - qz * vy), ty = 2 * (qz * vx - qx * vz), tz = 2 * (qx * vy - qy * vx);
    Wp[3 * c] = Wp[3 * p] + vx + qw * tx + (qy * tz - qz * ty);
    Wp[3 * c + 1] = Wp[3 * p + 1] + vy + qw * ty + (qz * tx - qx * tz);
    Wp[3 * c + 2] = Wp[3 * p + 2] + vz + qw * tz + (qx * ty - qy * tx);
  }
  for (let b = 0; b < nb; b++) {
    const x = Wq[4 * b], y = Wq[4 * b + 1], z = Wq[4 * b + 2], w = Wq[4 * b + 3];
    const r00 = 1 - 2 * (y * y + z * z), r01 = 2 * (x * y - w * z), r02 = 2 * (x * z + w * y);
    const r10 = 2 * (x * y + w * z), r11 = 1 - 2 * (x * x + z * z), r12 = 2 * (y * z - w * x);
    const r20 = 2 * (x * z - w * y), r21 = 2 * (y * z + w * x), r22 = 1 - 2 * (x * x + y * y);
    const o = 12 * b;
    for (let cc = 0; cc < 4; cc++) {
      const i0 = ib[o + cc], i1 = ib[o + 4 + cc], i2 = ib[o + 8 + cc];
      S[o + cc] = r00 * i0 + r01 * i1 + r02 * i2;
      S[o + 4 + cc] = r10 * i0 + r11 * i1 + r12 * i2;
      S[o + 8 + cc] = r20 * i0 + r21 * i1 + r22 * i2;
    }
    S[o + 3] += Wp[3 * b]; S[o + 7] += Wp[3 * b + 1]; S[o + 11] += Wp[3 * b + 2];
  }
  h.wq.set(Wq); h.wp.set(Wp); h.S.set(S);
  // 4. matrix-first LBS over the CSR weights: M = sum w S_b; pos = M [v0;1]; n = normalize(M3 n0). M -> skinM.
  const t0 = performance.now();
  const nv = A.nv, off = A.off, sb = A.sb, Wt = A.W, p0 = A.pos0, n0 = A.n0, PA = h.P.array, NA = h.N.array, SM = h.skinM;
  const doN = !h.halfNormals || (h.frameNo & 1) === 0;
  for (let v = 0; v < nv; v++) {
    let m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0, m5 = 0, m6 = 0, m7 = 0, m8 = 0, m9 = 0, m10 = 0, m11 = 0;
    for (let e = off[v], e1 = off[v + 1]; e < e1; e++) {
      const w = Wt[e], o = 12 * sb[e];
      m0 += w * S[o]; m1 += w * S[o + 1]; m2 += w * S[o + 2]; m3 += w * S[o + 3];
      m4 += w * S[o + 4]; m5 += w * S[o + 5]; m6 += w * S[o + 6]; m7 += w * S[o + 7];
      m8 += w * S[o + 8]; m9 += w * S[o + 9]; m10 += w * S[o + 10]; m11 += w * S[o + 11];
    }
    const i3 = 3 * v, x = p0[i3], y = p0[i3 + 1], z = p0[i3 + 2];
    PA[i3] = m0 * x + m1 * y + m2 * z + m3; PA[i3 + 1] = m4 * x + m5 * y + m6 * z + m7; PA[i3 + 2] = m8 * x + m9 * y + m10 * z + m11;
    if (doN) {
      const a = n0[i3], b = n0[i3 + 1], c = n0[i3 + 2];
      const nx = m0 * a + m1 * b + m2 * c, ny = m4 * a + m5 * b + m6 * c, nz = m8 * a + m9 * b + m10 * c, il = 1 / (Math.sqrt(nx * nx + ny * ny + nz * nz) || 1);
      NA[i3] = nx * il; NA[i3 + 1] = ny * il; NA[i3 + 2] = nz * il;
    }
    const i12 = 12 * v;
    SM[i12] = m0; SM[i12 + 1] = m1; SM[i12 + 2] = m2; SM[i12 + 3] = m3; SM[i12 + 4] = m4; SM[i12 + 5] = m5;
    SM[i12 + 6] = m6; SM[i12 + 7] = m7; SM[i12 + 8] = m8; SM[i12 + 9] = m9; SM[i12 + 10] = m10; SM[i12 + 11] = m11;
  }
  h.perf.lbs = performance.now() - t0;
  h.P.needsUpdate = true; if (doN) h.N.needsUpdate = true;
  h.frameNo++;
}


  // ---- Anny loading: prepareAnny without the vest items (pods, strap, gel pads) ----
  function prepareAnny(d) {
    const B = d.b64, nv = d.nv, nf = d.nf, nb = d.nb;
    const f64 = (s) => Float64Array.from(new Float32Array(b64buf(s)));
    const unit3 = (a) => { const o = new Float64Array(a.length); for (let i = 0; i < a.length; i += 3) { const n = Math.hypot(a[i], a[i + 1], a[i + 2]); o[i] = a[i] / n; o[i + 1] = a[i + 1] / n; o[i + 2] = a[i + 2] / n; } return o; };
    const pos0 = new Float32Array(b64buf(B.pos)), idx = new Uint16Array(b64buf(B.idx)), off = new Uint32Array(b64buf(B.skinOff));
    const sb = new Uint8Array(b64buf(B.skinBone)), sw = new Uint16Array(b64buf(B.skinW)), W = new Float32Array(sw.length);
    if (pos0.length !== nv * 3 || idx.length !== nf * 3 || off.length !== nv + 1 || off[nv] !== sw.length) throw new Error('anny: array sizes');
    for (let v = 0; v < nv; v++) { let t = 0; for (let e = off[v]; e < off[v + 1]; e++) t += sw[e]; for (let e = off[v]; e < off[v + 1]; e++) W[e] = sw[e] / t; }
    const acc = new Float64Array(nv * 3), n0 = new Float32Array(nv * 3);
    for (let f = 0; f < nf; f++) {
      const a = idx[3 * f] * 3, b = idx[3 * f + 1] * 3, c = idx[3 * f + 2] * 3;
      const ux = pos0[b] - pos0[a], uy = pos0[b + 1] - pos0[a + 1], uz = pos0[b + 2] - pos0[a + 2];
      const vx = pos0[c] - pos0[a], vy = pos0[c + 1] - pos0[a + 1], vz = pos0[c + 2] - pos0[a + 2];
      const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
      acc[a] += cx; acc[a + 1] += cy; acc[a + 2] += cz; acc[b] += cx; acc[b + 1] += cy; acc[b + 2] += cz; acc[c] += cx; acc[c + 1] += cy; acc[c + 2] += cz;
    }
    for (let v = 0; v < nv * 3; v += 3) { const n = Math.hypot(acc[v], acc[v + 1], acc[v + 2]) || 1; n0[v] = acc[v] / n; n0[v + 1] = acc[v + 1] / n; n0[v + 2] = acc[v + 2] / n; }
    const names = d.bones.name, parent = Int32Array.from(d.bones.parent), bi = {};
    names.forEach((n, i) => { bi[n] = i; });
    const rlp = f64(B.restLocalPos), rlq = f64(B.restLocalQuat), rwq = f64(B.restWorldQuat), ib = f64(B.invBind);
    const PIY = new Float64Array([0, 1, 0, 0]), K = new Float64Array(nb * 4);
    for (let b = 0; b < nb; b++) aqMul(K, 4 * b, PIY, 0, rwq, 4 * b);
    const fg = d.fingers, nq = new Int16Array(b64buf(d.neutral.q)), nfi = new Int16Array(b64buf(d.neutral.fing));
    const neutral = { root: Float64Array.from(d.neutral.root), q: new Float64Array(120), fing: new Float64Array(32) };
    for (let j = 0; j < 30; j++) aqNormQ(neutral.q, 4 * j, nq, 4 * j, 32767);
    for (let j = 0; j < 32; j++) neutral.fing[j] = nfi[j] / 1e4;
    return {
      d, nv, nf, nb, height: d.height, pos0, idx, off, sb, W, n0, names, parent, bi, rlp, rlq, rwq, ib, K, alpha: d.alpha,
      carriers: ANNY_CARRIERS.map(([n, k]) => ({ b: bi[n], k })), carrierLocals: Int32Array.from(ANNY_CARRIER_LOCALS.map((n) => bi[n])), filler: bi.spine04,
      qb: Int32Array.from(ANNY_Q_BONES.map((n) => bi[n])),
      fb: Int32Array.from(fg.bones.map((n) => bi[n])), fbase: f64(fg.base), faxis: unit3(f64(fg.axis)), fax2: unit3(f64(fg.axis2Thumb)),
      tb: Int32Array.from(d.toes.bones.map((n) => bi[n])), taxis: unit3(f64(d.toes.axis)),
      neutral, anatomyFile: d.anatomy,
    };
  }
  function prepareAnnyClip(id, d) {
    const B = d.b64, f32 = (s) => new Float32Array(b64buf(s)), i16 = (s) => new Int16Array(b64buf(s));
    const c = { id, n: d.n, nl: d.nl, fps: d.fps, lf: d.limbFps, t0: d.t0, T: (d.n - 1) / d.fps, hcal: d.hCalDeg * Math.PI / 180,
      segq: f32(B.segq), root: f32(B.root), q: i16(B.q), mtp: i16(B.mtp), fing: i16(B.fing), box: f32(B.box), env: d.env || {}, captions: d.captions || [] };
    if (c.segq.length !== c.n * 16 || c.q.length !== c.nl * 120 || c.fing.length !== c.nl * 32 || c.box.length !== c.nl * 7) throw new Error(`clip ${id}: array sizes`);
    return c;
  }
  const makeAnnySample = () => ({ neutral: true, hcal: 0, segq: new Float64Array(16), root: new Float64Array(3), q: new Float64Array(120), fing: new Float64Array(32), mtp: new Float64Array(2), box: new Float64Array(7) });
  function annyNeutralSample(f) {
    const N = ANNY.neutral;
    f.neutral = true; f.hcal = 0; f.root.set(N.root); f.q.set(N.q); f.fing.set(N.fing); f.mtp.fill(0); f.box.fill(0); f.box[6] = 1;
    for (let k = 0; k < 4; k++) { f.segq[4 * k] = 0; f.segq[4 * k + 1] = 0; f.segq[4 * k + 2] = 0; f.segq[4 * k + 3] = 1; }
    return f;
  }
  // One posed body: its own skinned positions/normals; the anatomy meshes share the decoded geometry.
  function makeBody(skinMat, glassMat) {
    const A = ANNY, nv = A.nv, nb = A.nb;
    const root = new THREE.Group();
    const geo = new THREE.BufferGeometry();
    const P = new THREE.BufferAttribute(new Float32Array(nv * 3), 3).setUsage(THREE.DynamicDrawUsage);
    const N = new THREE.BufferAttribute(new Float32Array(nv * 3), 3).setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('position', P); geo.setAttribute('normal', N); geo.setIndex(new THREE.BufferAttribute(A.idx, 1));
    const body = new THREE.Mesh(geo, skinMat), glass = new THREE.Mesh(geo, glassMat);
    body.frustumCulled = glass.frustumCulled = false; glass.renderOrder = 10;
    root.add(body, glass);
    const h = {
      root, body, glass, P, N, geo, anat: null,
      wq: new Float32Array(nb * 4), wp: new Float32Array(nb * 3), S: new Float32Array(nb * 12), skinM: new Float32Array(nv * 12),
      _L: new Float64Array(nb * 4), _Wq: new Float64Array(nb * 4), _Wp: new Float64Array(nb * 3), _S: new Float64Array(nb * 12), D: new Float64Array(16),
      smp: makeAnnySample(), frameNo: 0, halfNormals: false, perf: { lbs: 0 }, trunkDeg: 0,
    };
    poseAnny(h, annyNeutralSample(h.smp));
    return h;
  }


  // ================= Page 3 (#how): Anny in a hi-vis vest, sensors on the vest, leader-line callouts =================
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const stage = SEC.querySelector('.vest-stage'), canvas = stage.querySelector('canvas'), svg = stage.querySelector('svg.leaders');
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  const VEST_OFF = 0.009;                       // vest surface: 9 mm off the skin along the normal
  const NS = 'http://www.w3.org/2000/svg';
  let started = false, ready = false, active = false, raf = 0, renderer, scene, camera, controls, H, back = 1, spin = !reduce.matches, lastT = 0;
  const marks = {};                             // key -> [{p, n}] world anchor points for the callouts
  const cos = [...stage.querySelectorAll('.co')].map((el) => ({ el, key: el.dataset.k, left: el.classList.contains('co-l'), lines: [], dots: [] }));

  function smoothNormals(P, idx, nv) {
    const acc = new Float32Array(nv * 3);
    for (let f = 0; f < idx.length; f += 3) {
      const a = idx[f] * 3, b = idx[f + 1] * 3, c = idx[f + 2] * 3;
      const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2], vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
      const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
      acc[a] += cx; acc[a + 1] += cy; acc[a + 2] += cz; acc[b] += cx; acc[b + 1] += cy; acc[b + 2] += cz; acc[c] += cx; acc[c + 1] += cy; acc[c + 2] += cz;
    }
    for (let v = 0; v < nv * 3; v += 3) { const n = Math.hypot(acc[v], acc[v + 1], acc[v + 2]) || 1; acc[v] /= n; acc[v + 1] /= n; acc[v + 2] /= n; }
    return acc;
  }

  function build(d) {
    const A = ANNY, ink = css('--ink'), yellow = css('--accent'), blue = css('--accent-2'), band = css('--surface');
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(3, Math.max(2, window.devicePixelRatio || 1)));   // supersampled: the vest cut lines are shader discards
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
    scene.add(camera);
    scene.add(new THREE.HemisphereLight(0xffffff, 0x9a9390, 0.72));   // lights stay white
    const key = new THREE.DirectionalLight(0xffffff, 0.42); key.position.set(-1.2, 1.6, 2.5); camera.add(key);

    H = makeBody(new THREE.MeshLambertMaterial({ color: 0xbfbab7 }), new THREE.MeshBasicMaterial({ visible: false }));
    H.glass.visible = false; scene.add(H.root);
    const P = H.P.array, idx = A.idx, nv = A.nv, N = smoothNormals(P, idx, nv);
    H.N.array.set(N); H.N.needsUpdate = true;
    const bonePos = (name) => { const b = A.bi[name]; return V(H._Wp[3 * b], H._Wp[3 * b + 1], H._Wp[3 * b + 2]); };
    const boneQ = (b) => new THREE.Quaternion(H._Wq[4 * b], H._Wq[4 * b + 1], H._Wq[4 * b + 2], H._Wq[4 * b + 3]);
    const surf = (tri, bary) => {
      const p = V(), n = V();
      for (let k = 0; k < 3; k++) { const v = 3 * idx[3 * tri + k], w = bary[k]; p.x += w * P[v]; p.y += w * P[v + 1]; p.z += w * P[v + 2]; n.x += w * N[v]; n.y += w * N[v + 1]; n.z += w * N[v + 2]; }
      return { p, n: n.normalize() };
    };
    const pods = d.pods.map((p) => surf(p.tri, p.bary));
    back = Math.sign(pods[1].p.z) || 1;                        // which way the back faces after the display turn

    // ---- the vest: Anny's own torso skin, offset 9 mm; the cut lines (hem, armholes, neck) are drawn in the shader ----
    const grp = A.names.map((n) => /^(upperarm|lowerarm|wrist|finger|metacarpal)/.test(n) ? 1 : /^(upperleg|lowerleg|foot|toe)/.test(n) ? 2 : /^(neck|head|eye)/.test(n) ? 3 : 0);
    const shL = bonePos('upperarm01.L'), shR = bonePos('upperarm01.R'), neck = bonePos('neck01'), hip = bonePos('root');
    const HEM = hip.y - 0.025, keep = new Uint8Array(nv);
    for (let v = 0; v < nv; v++) {
      const w = [0, 0, 0, 0];
      for (let e = A.off[v]; e < A.off[v + 1]; e++) w[grp[A.sb[e]]] += A.W[e];
      keep[v] = (w[1] < 0.5 && w[2] < 0.97 && w[3] < 0.8 && P[3 * v + 1] > HEM - 0.05 && P[3 * v + 1] < neck.y + 0.08) ? 1 : 0;
    }
    const vIdx = [];
    for (let f = 0; f < idx.length; f += 3) if (keep[idx[f]] && keep[idx[f + 1]] && keep[idx[f + 2]]) vIdx.push(idx[f], idx[f + 1], idx[f + 2]);
    const vp = new Float32Array(nv * 3);
    for (let i = 0; i < nv * 3; i++) vp[i] = P[i] + N[i] * VEST_OFF;
    const vg = new THREE.BufferGeometry();
    vg.setAttribute('position', new THREE.BufferAttribute(vp, 3)); vg.setAttribute('normal', new THREE.BufferAttribute(N, 3)); vg.setIndex(vIdx);
    const U = { uHem: { value: HEM }, uShL: { value: shL }, uShR: { value: shR }, uNeck: { value: neck }, uBack: { value: back },
      uBandY: { value: HEM + 0.07 }, uYel: { value: new THREE.Color(yellow) }, uInk: { value: new THREE.Color(ink) }, uBand: { value: new THREE.Color(band) } };
    const vm = new THREE.MeshLambertMaterial({ color: 0xffffff, side: THREE.DoubleSide });
    vm.onBeforeCompile = (sh) => {
      Object.assign(sh.uniforms, U);
      sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vHP;').replace('#include <begin_vertex>', '#include <begin_vertex>\nvHP = position;');
      sh.fragmentShader = sh.fragmentShader.replace('#include <common>', `#include <common>
        varying vec3 vHP; uniform float uHem, uBack, uBandY; uniform vec3 uShL, uShR, uNeck, uYel, uInk, uBand;
        float armhole(vec3 p, vec3 s) { vec3 a = s + vec3(0., .035, 0.), b = s - vec3(0., .17, 0.); float h = clamp(dot(p - a, b - a) / dot(b - a, b - a), 0., 1.); return length((p - a - (b - a) * h) * vec3(1., 1., .8)) - .075; }`)
        .replace('#include <color_fragment>', `#include <color_fragment>
        vec3 hp = vHP; float e = hp.y - uHem;
        e = min(e, min(armhole(hp, uShL), armhole(hp, uShR)));
        e = min(e, length((hp - uNeck - vec3(0., .02, 0.)) * vec3(1., .75, 1.)) - .083);
        bool front = hp.z * uBack < 0.;
        float vy = uNeck.y - .2;
        if (front) e = min(e, (abs(hp.x) - (hp.y - vy) * .4) * .92);
        if (e < 0.) discard;
        vec3 col = uYel;
        float by = hp.y - uBandY;
        if (abs(by) < .019 || abs(by - .085) < .019 || (by > .085 && abs(abs(hp.x) - .105) < .017)) col = uBand;
        if (front && abs(hp.x) < .004 && hp.y < vy + .01) col = uInk;
        if (e < .011) col = uInk;
        diffuseColor.rgb = col;`);
    };
    const vest = new THREE.Mesh(vg, vm); vest.frustumCulled = false; H.root.add(vest);

    // ---- sensors on top of the vest ----
    const blueM = new THREE.MeshLambertMaterial({ color: blue }), inkM = new THREE.MeshLambertMaterial({ color: ink, side: THREE.DoubleSide }), yelM = new THREE.MeshLambertMaterial({ color: yellow });
    const UP = V(0, 1, 0);
    const place = (mesh, p, n, lift) => {
      const x = V().crossVectors(UP, n).normalize(), y = V().crossVectors(n, x).normalize();
      mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, n));
      mesh.position.copy(p).addScaledVector(n, lift); H.root.add(mesh); return mesh;
    };
    const mark = (k, p, n, lift) => { (marks[k] = marks[k] || []).push({ p: p.clone().addScaledVector(n, lift), n: n.clone() }); };
    // spine strip: an ink ribbon over the midline anchors from below S1 to above T2, four blue pods on it
    const mt = new Uint32Array(b64buf(d.midline.tri)), mb = new Float32Array(b64buf(d.midline.bary));
    const mid = [];
    for (let i = 0; i < mt.length; i++) { const s = surf(mt[i], [mb[3 * i], mb[3 * i + 1], mb[3 * i + 2]]); if (s.p.y > pods[0].p.y - 0.05 && s.p.y < pods[3].p.y + 0.05) mid.push(s); }
    mid.sort((a, b) => a.p.y - b.p.y);
    const rp = [], ri = [];
    mid.forEach((s, i) => {
      for (const sx of [-1, 1]) rp.push(sx * 0.024, s.p.y + s.n.y * (VEST_OFF + 0.004), s.p.z + s.n.z * (VEST_OFF + 0.004) + back * 0.001);
      if (i) { const a = 2 * i - 2; ri.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); }
    });
    const rg = new THREE.BufferGeometry(); rg.setAttribute('position', new THREE.Float32BufferAttribute(rp, 3)); rg.setIndex(ri); rg.computeVertexNormals();
    H.root.add(new THREE.Mesh(rg, inkM));
    const podGeo = new THREE.BoxGeometry(0.058, 0.066, 0.02);
    pods.forEach((s) => { s.p.x = 0; s.n.x = 0; s.n.normalize(); place(new THREE.Mesh(podGeo, blueM), s.p, s.n, VEST_OFF + 0.014); mark('spine', s.p, s.n, VEST_OFF + 0.026); });
    // chest pods, collar microphone
    const boxS = new THREE.BoxGeometry(0.056, 0.056, 0.02), mic = new THREE.CylinderGeometry(0.017, 0.017, 0.02, 20).rotateX(Math.PI / 2);
    const ledGeo = new THREE.BoxGeometry(0.03, 0.008, 0.004);
    for (const a of d.attach) {
      if (a.kind !== 'skinned' || !['climate', 'air', 'sound'].includes(a.mount)) continue;
      const s = surf(a.tri, a.bary);
      const m = place(new THREE.Mesh(a.mount === 'sound' ? mic : boxS, blueM), s.p, s.n, VEST_OFF + 0.01);
      if (a.mount !== 'sound') { const led = new THREE.Mesh(ledGeo, yelM); led.position.set(0, 0.014, 0.011); m.add(led); }
      mark(a.mount, s.p, s.n, VEST_OFF + 0.022);
    }
    // ear clip (heart rate) and the hip pouch (ESP32-S3): rigid on their bones
    for (const a of d.attach) {
      if (a.kind !== 'rigid' || !['hr', 'core'].includes(a.mount)) continue;
      const b = A.bi[a.bone], q = boneQ(b), p = V(a.p[0], a.p[1], a.p[2]).applyQuaternion(q).add(bonePos(a.bone));
      const n = V(0, 0, 1).applyQuaternion(q.clone().multiply(new THREE.Quaternion(a.q[0], a.q[1], a.q[2], a.q[3])));
      if (a.mount === 'hr') {
        p.x = -p.x; n.x = -n.x;   // shown on the ear nearer the default camera
        place(new THREE.Mesh(new THREE.BoxGeometry(0.022, 0.036, 0.02), blueM), p, n, 0.008); mark('hr', p, n, 0.02);
      } else {
        n.y = 0; n.normalize();
        const m = place(new THREE.Mesh(new THREE.BoxGeometry(0.115, 0.085, 0.04), inkM), p, n, VEST_OFF + 0.022);
        const flap = new THREE.Mesh(new THREE.BoxGeometry(0.117, 0.016, 0.042), yelM); flap.position.set(0, 0.02, 0); m.add(flap);
        mark('core', p, n, VEST_OFF + 0.044);
      }
    }

    // ---- camera: knees up, 3/4 from behind, drag to turn, no zoom ----
    controls = new THREE.OrbitControls(camera, canvas);
    controls.enableZoom = false; controls.enablePan = false; controls.enableDamping = true; controls.dampingFactor = 0.12; controls.rotateSpeed = 0.7;
    controls.minPolarAngle = Math.PI * 0.36; controls.maxPolarAngle = Math.PI * 0.56;
    controls.target.set(0, 1.2, 0);
    controls.addEventListener('start', () => { spin = false; setPressed(null); });
    canvas.style.touchAction = 'pan-y';
    setView('back');

    // ---- callouts: one line and dot per anchor ----
    for (const c of cos) for (const m of (marks[c.key] || [])) {
      const l = document.createElementNS(NS, 'line'), o = document.createElementNS(NS, 'circle'); o.setAttribute('r', 5);
      svg.append(l, o); c.lines.push(l); c.dots.push(o);
    }
    new ResizeObserver(resize).observe(stage); resize();
    ready = true; stage.classList.add('ready');
  }

  function setPressed(v) { stage.querySelectorAll('.vest-views button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === v))); }
  function setView(v) {
    const az = v === 'front' ? 0.5 : Math.PI - 0.62, dist = DIST, el = 0.1;
    const s = -back;                                         // front direction along z
    camera.position.set(controls.target.x + Math.sin(az) * Math.cos(el) * dist, controls.target.y + Math.sin(el) * dist, controls.target.z + s * Math.cos(az) * Math.cos(el) * dist);
    controls.update(); setPressed(v);
  }
  stage.querySelectorAll('.vest-views button').forEach((b) => b.addEventListener('click', () => { if (!ready) return; spin = false; setView(b.dataset.view); draw(); }));

  const _v = V(), _c = V();
  let SW = 0, SH = 0, anchors = [], DIST = 2.8;
  function resize() {
    const r = stage.getBoundingClientRect(); SW = Math.round(r.width); SH = Math.round(r.height);
    if (!SW || !SH) return;
    renderer.setSize(SW, SH, false); camera.aspect = SW / SH;
    // keep the figure the same height; on a narrow stage widen the view instead of cropping it
    // wide stage: knees up. Narrow stage (phone): the whole body, so the callouts fit beside it.
    const narrow = SW < 560, dist = narrow ? 4.3 : 2.8, ty = narrow ? 1.0 : 1.2;
    _v.copy(camera.position).sub(controls.target).setLength(dist); controls.target.y = ty; camera.position.copy(controls.target).add(_v); DIST = dist;
    camera.updateProjectionMatrix();
    svg.setAttribute('viewBox', `0 0 ${SW} ${SH}`);
    anchors = cos.map((c) => { const b = c.el.getBoundingClientRect(); return { x: (c.left ? b.right : b.left) - r.left, y: b.bottom - r.top - 1 }; });
    draw();
  }
    function draw() {
    if (!renderer || !SW) return;
    controls.update(); renderer.render(scene, camera);
    cos.forEach((c, i) => {
      const a = anchors[i], ms = marks[c.key] || [];
      ms.forEach((m, j) => {
        _c.copy(camera.position).sub(m.p); const hid = _c.dot(m.n) < 0.02 * _c.length();
        _v.copy(m.p).project(camera); const x = (_v.x * 0.5 + 0.5) * SW, y = (-_v.y * 0.5 + 0.5) * SH;
        const l = c.lines[j], o = c.dots[j];
        l.setAttribute('x1', a.x); l.setAttribute('y1', a.y); l.setAttribute('x2', x.toFixed(1)); l.setAttribute('y2', y.toFixed(1));
        o.setAttribute('cx', x.toFixed(1)); o.setAttribute('cy', y.toFixed(1));
        l.classList.toggle('hid', hid); o.classList.toggle('hid', hid);
      });
    });
  }
  function frame(t) {
    raf = 0; if (!active) return;
    const dt = Math.min(0.05, (t - (lastT || t)) / 1000); lastT = t;
    if (spin && ready) { const p = camera.position, c = controls.target, a = dt * 0.16, dx = p.x - c.x, dz = p.z - c.z; p.x = c.x + dx * Math.cos(a) - dz * Math.sin(a); p.z = c.z + dx * Math.sin(a) + dz * Math.cos(a); }
    if (ready) draw();
    raf = requestAnimationFrame(frame);
  }
  function setActive(on) {
    on = !!on && !document.hidden;
    if (on === active) return; active = on; lastT = 0;
    if (on && !raf) raf = requestAnimationFrame(frame);
    if (!on && raf) { cancelAnimationFrame(raf); raf = 0; }
  }
  async function start() {
    if (started) return; started = true;
    try {
      const d = await (await fetch(DATA + 'anny-avg-male.v1.json')).json();
      ANNY = prepareAnny(d); build(d);
    } catch (e) { stage.classList.add('failed'); if (window.console) console.warn('how: 3D figure unavailable', e); }
  }
  const inView = () => { const r = SEC.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight; };
  new IntersectionObserver((es, o) => { if (es.some((x) => x.isIntersecting)) { start(); o.disconnect(); } }, { rootMargin: '60% 0px' }).observe(SEC);
  new IntersectionObserver((es) => es.forEach((en) => setActive(en.isIntersecting)), { threshold: 0.02 }).observe(SEC);
  document.addEventListener('deck:page', (e) => { if (!e.detail) return; if (e.detail.id === 'how') { start(); setActive(true); } else if (matchMedia('(min-width:900px) and (min-height:620px)').matches) setActive(false); });
  document.addEventListener('visibilitychange', () => setActive(!document.hidden && inView()));
  window.__how = { view: (v) => { spin = false; if (ready) { setView(v); draw(); } }, spin: (b) => { spin = !!b; }, ready: () => ready, marks, cam: () => camera };
})();
