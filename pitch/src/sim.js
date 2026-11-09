/* Page 4 (#simulation): six simulated scenes on one Anny body, drawn by ONE WebGL renderer with scissor viewports.
   Adapted from the InSite page (insite_anny_build: anny_block.js, mon_anny.js, anat_core.js, anat_block.js):
   - Anny (NAVER LABS Europe, Apache-2.0; mesh, rig, weights from MakeHuman/MPFB2, CC0), CPU linear blend skinning per tile.
   - Run tiles: realism-v2 ThunderSim runs posed through the crew-monitor solve (trunk from the simulated segments,
     limbs by two-bone IK to the simulated joints). Clip tiles: the baked lift-and-carry and simulated-faint clips.
   - Anatomy: BodyParts3D (DBCLS, CC BY 4.0) fitted to Anny, posed on the GPU from a per-tile palette texture.
   Frame: +X wearer-right, +Y up, +Z posterior. Quaternions (x, y, z, w). Everything shown is simulated. */
(function () {
  'use strict';
  const SEC = document.getElementById('simulation');
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

const _sa4 = new Float64Array(4), _sb4 = new Float64Array(4);
function annyLerp(o, src, w, scale, len, il, ul) { for (let k = 0; k < len; k++) { const a = src[il * w + k] / scale, b = src[(il + 1) * w + k] / scale; o[k] = a + (b - a) * ul; } }
function sampleAnny(c, tc, f) {
  tc = clamp(tc, 0, c.T);
  const x = tc * c.fps, i = Math.min(Math.floor(x + 1e-9), c.n - 2), u = clamp(x - i, 0, 1);
  for (let k = 0; k < 4; k++) aqSlerp(f.segq, 4 * k, c.segq, 16 * i + 4 * k, c.segq, 16 * (i + 1) + 4 * k, u);
  const xl = tc * c.lf, il = Math.min(Math.floor(xl + 1e-9), c.nl - 2), ul = clamp(xl - il, 0, 1);
  for (let j = 0; j < 30; j++) { aqNormQ(_sa4, 0, c.q, (il * 30 + j) * 4, 32767); aqNormQ(_sb4, 0, c.q, ((il + 1) * 30 + j) * 4, 32767); aqSlerp(f.q, 4 * j, _sa4, 0, _sb4, 0, ul); }
  annyLerp(f.root, c.root, 3, 1, 3, il, ul); annyLerp(f.fing, c.fing, 32, 1e4, 32, il, ul); annyLerp(f.mtp, c.mtp, 2, 1e4, 2, il, ul); annyLerp(f.box, c.box, 7, 1, 3, il, ul);
  aqNormQ(_sa4, 0, c.box, il * 7 + 3, 1); aqNormQ(_sb4, 0, c.box, (il + 1) * 7 + 3, 1); aqSlerp(f.box, 3, _sa4, 0, _sb4, 0, ul);
  f.neutral = false; f.hcal = c.hcal; f.il = il; f.ul = ul; f.tc = tc;
  return f;
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

// ---- plain-array helpers ----
function mMat2Q(o, oo, m00, m01, m02, m10, m11, m12, m20, m21, m22) {   // rotation matrix -> unit quaternion (xyzw)
  const tr = m00 + m11 + m22; let x, y, z, w;
  if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; w = 0.25 * s; x = (m21 - m12) / s; y = (m02 - m20) / s; z = (m10 - m01) / s; }
  else if (m00 > m11 && m00 > m22) { const s = Math.sqrt(1 + m00 - m11 - m22) * 2; w = (m21 - m12) / s; x = 0.25 * s; y = (m01 + m10) / s; z = (m02 + m20) / s; }
  else if (m11 > m22) { const s = Math.sqrt(1 + m11 - m00 - m22) * 2; w = (m02 - m20) / s; x = (m01 + m10) / s; y = 0.25 * s; z = (m12 + m21) / s; }
  else { const s = Math.sqrt(1 + m22 - m00 - m11) * 2; w = (m10 - m01) / s; x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s; }
  const n = Math.sqrt(x * x + y * y + z * z + w * w); o[oo] = x / n; o[oo + 1] = y / n; o[oo + 2] = z / n; o[oo + 3] = w / n;
}
// Frame with +Y along d and +X along ref (made perpendicular to d); returns its quaternion.
function mBasisQ(o, oo, dx, dy, dz, rx, ry, rz) {
  let n = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1; dx /= n; dy /= n; dz /= n;
  const p = rx * dx + ry * dy + rz * dz; let xx = rx - p * dx, xy = ry - p * dy, xz = rz - p * dz; n = Math.sqrt(xx * xx + xy * xy + xz * xz);
  if (n < 1e-6) { if (Math.abs(dx) < 0.9) { xx = 0; xy = -dz; xz = dy; } else { xx = -dy; xy = dx; xz = 0; } n = Math.sqrt(xx * xx + xy * xy + xz * xz); }
  xx /= n; xy /= n; xz /= n;
  const zx = xy * dz - xz * dy, zy = xz * dx - xx * dz, zz = xx * dy - xy * dx;
  mMat2Q(o, oo, xx, dx, zx, xy, dy, zy, xz, dz, zz);
}
function mRot(q, qo, v, vo, out, oo) {   // out = q v q*
  const qx = q[qo], qy = q[qo + 1], qz = q[qo + 2], qw = q[qo + 3], vx = v[vo], vy = v[vo + 1], vz = v[vo + 2];
  const tx = 2 * (qy * vz - qz * vy), ty = 2 * (qz * vx - qx * vz), tz = 2 * (qx * vy - qy * vx);
  out[oo] = vx + qw * tx + (qy * tz - qz * ty); out[oo + 1] = vy + qw * ty + (qz * tx - qx * tz); out[oo + 2] = vz + qw * tz + (qx * ty - qy * tx);
}
// FK of the whole rig from locals L and the root position (annySkin step 3 without the skinning).
function monFK(L, root, Wq, Wp) {
  const A = ANNY, nb = A.nb, par = A.parent, rlp = A.rlp;
  Wq[0] = L[0]; Wq[1] = L[1]; Wq[2] = L[2]; Wq[3] = L[3]; Wp[0] = root[0]; Wp[1] = root[1]; Wp[2] = root[2];
  for (let c = 1; c < nb; c++) {
    const p = par[c];
    aqMul(Wq, 4 * c, Wq, 4 * p, L, 4 * c);
    mRot(Wq, 4 * p, rlp, 3 * c, Wp, 3 * c);
    Wp[3 * c] += Wp[3 * p]; Wp[3 * c + 1] += Wp[3 * p + 1]; Wp[3 * c + 2] += Wp[3 * p + 2];
  }
}
// poseAnny step 1: carriers W_b = D_k (x) K_b, spine04 = slerp(D_0, D_1, alpha) (x) K, their locals; every other local at rest.
const _mcq = new Float64Array(4);
function monCarriers(D, L, Wq) {
  const A = ANNY, par = A.parent;
  L.set(A.rlq);
  for (const c of A.carriers) aqMul(Wq, 4 * c.b, D, 4 * c.k, A.K, 4 * c.b);
  aqSlerp(_mcq, 0, D, 0, D, 4, A.alpha); aqMul(Wq, 4 * A.filler, _mcq, 0, A.K, 4 * A.filler);
  L[0] = Wq[0]; L[1] = Wq[1]; L[2] = Wq[2]; L[3] = Wq[3];
  for (let i = 0; i < A.carrierLocals.length; i++) { const b = A.carrierLocals[i]; aqCMul(L, 4 * b, Wq, 4 * par[b], Wq, 4 * b); }
}
const mSub = (a, ao, b, bo, o) => { o[0] = a[ao] - b[bo]; o[1] = a[ao + 1] - b[bo + 1]; o[2] = a[ao + 2] - b[bo + 2]; return o; };
const mLen = (v) => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

// ---- the rig constants for the monitor IK, built once from the neutral Anny ----
function monAnnyRig(h) {
  const A = ANNY, nb = A.nb, bi = A.bi;
  const D = new Float64Array(16); for (let k = 0; k < 4; k++) D[4 * k + 3] = 1;
  const L = new Float64Array(nb * 4), RQ = new Float64Array(nb * 4), RP = new Float64Array(nb * 3);
  monCarriers(D, L, RQ); monFK(L, [0, 0, 0], RQ, RP);          // the rest pose in the posed frame (+Z posterior)
  const t = [0, 0, 0], u = [0, 0, 0];
  const dir = (a, b) => { mSub(RP, 3 * b, RP, 3 * a, t); const n = mLen(t); return [t[0] / n, t[1] / n, t[2] / n, n]; };
  // Neutral pose (makeAnny posed it): soles, ankle height and the elbow hinge in the rig's own twist.
  const NQ = h.wq, NP = h.wp, PA = h.P.array;
  let minY = Infinity; for (let i = 1; i < PA.length; i += 3) minY = Math.min(minY, PA[i]);
  const legs = ['L', 'R'].map((s) => {
    const ul = bi[`upperleg01.${s}`], ul2 = bi[`upperleg02.${s}`], ll = bi[`lowerleg01.${s}`], ll2 = bi[`lowerleg02.${s}`], ft = bi[`foot.${s}`], pel = bi[`pelvis.${s}`];
    const th = dir(ul, ll), sh = dir(ll, ft);
    // sole: the lowest vertices bound mostly to this foot or its toes; heel = most posterior, toe = most anterior
    const own = new Set([ft]); for (let b = ft + 1; b < nb; b++) if (own.has(A.parent[b])) own.add(b);   // the foot and every toe bone under it
    // 26 directions (cube faces, edges, corners): the foot's extreme point along each, so any tilt of the foot is clamped
    const dirs = []; for (let a = -1; a <= 1; a++) for (let b2 = -1; b2 <= 1; b2++) for (let c = -1; c <= 1; c++) if (a || b2 || c) dirs.push([a, b2, c]);
    let heel = null, toe = null; const ext = new Array(dirs.length).fill(null), mine = [];
    for (let v = 0; v < A.nv; v++) {
      let best = -1, bw = 0; for (let e = A.off[v]; e < A.off[v + 1]; e++) if (A.W[e] > bw) { bw = A.W[e]; best = A.sb[e]; }
      if (!own.has(best)) continue;
      const x = PA[3 * v], y = PA[3 * v + 1], z = PA[3 * v + 2];
      mine.push(v);
      // the foot's extreme points (x, y, z min and max), so a foot on its side or edge is clamped too
      dirs.forEach((d, i) => { const val = d[0] * x + d[1] * y + d[2] * z; if (!ext[i] || val > ext[i][3]) ext[i] = [x, y, z, val]; });
      if (y > minY + 0.012) continue;
      if (!heel || z > heel[2]) heel = [x, y, z];
      if (!toe || z < toe[2]) toe = [x, y, z];
    }
    // into the foot bone's neutral frame
    const loc = (p) => { const o = [0, 0, 0], d = [p[0] - NP[3 * ft], p[1] - NP[3 * ft + 1], p[2] - NP[3 * ft + 2]]; const qi = [-NQ[4 * ft], -NQ[4 * ft + 1], -NQ[4 * ft + 2], NQ[4 * ft + 3]]; mRot(qi, 0, d, 0, o, 0); return o; };
    // rest foot frame: the neutral foot's sole forward (heel->toe), carried into rest through the foot's own rotation
    const fwdN = [toe[0] - heel[0], 0, toe[2] - heel[2]], qi = [-NQ[4 * ft], -NQ[4 * ft + 1], -NQ[4 * ft + 2], NQ[4 * ft + 3]], fl = [0, 0, 0], fr = [0, 0, 0];
    mRot(qi, 0, fwdN, 0, fl, 0); mRot(RQ, 4 * ft, fl, 0, fr, 0);
    const ankleH = NP[3 * ft + 1] - minY;
    const pts = [loc(heel), loc(toe), ...ext.map((e) => loc(e))];
    return { s, ul, ul2, ll, ll2, ft, pel, l1: th[3], l2: sh[3], th: th.slice(0, 3), sh: sh.slice(0, 3), pts, fwd: fr, ankleH, verts: mine };
  });
  const arms = ['L', 'R'].map((s) => {
    const ua = bi[`upperarm01.${s}`], ua2 = bi[`upperarm02.${s}`], la = bi[`lowerarm01.${s}`], la2 = bi[`lowerarm02.${s}`], wr = bi[`wrist.${s}`], so = bi[`shoulder01.${s}`];
    const up = dir(ua, la), fo = dir(la, wr);
    // elbow hinge: from the neutral arm if it is bent enough, else the anatomical one (+X with the arm hanging, forearm forward)
    mSub(NP, 3 * la, NP, 3 * ua, t); mSub(NP, 3 * wr, NP, 3 * la, u);
    let hx = t[1] * u[2] - t[2] * u[1], hy = t[2] * u[0] - t[0] * u[2], hz = t[0] * u[1] - t[1] * u[0];
    const hn = Math.sqrt(hx * hx + hy * hy + hz * hz) / (mLen(t) * mLen(u));
    let hinge;
    if (hn > 0.12) {   // > 7 degrees: carry it into rest through the upper arm's rotation
      const n = Math.sqrt(hx * hx + hy * hy + hz * hz), hl = [0, 0, 0], hr = [0, 0, 0];
      mRot([-NQ[4 * ua], -NQ[4 * ua + 1], -NQ[4 * ua + 2], NQ[4 * ua + 3]], 0, [hx / n, hy / n, hz / n], 0, hl, 0); mRot(RQ, 4 * ua, hl, 0, hr, 0);
      hinge = hr;
    } else hinge = [1, 0, 0];
    const palm = 0.5 * dir(wr, bi[`finger3-1.${s}`])[3];   // wrist -> palm centre (half way to the middle knuckle)
    return { s, ua, ua2, la, la2, wr, so, palm, l1: up[3], l2: fo[3], up: up.slice(0, 3), fo: fo.slice(0, 3), hinge, neutralBend: Math.asin(Math.min(1, hn)) * DEG };
  });
  const legLen = (legs[0].l1 + legs[0].l2 + legs[0].ankleH + legs[1].l1 + legs[1].l2 + legs[1].ankleH) / 2;
  const neck = ['neck01', 'neck02', 'neck03', 'head'].map((n) => bi[n]);
  const rest = {};   // per-chain inverse rest frames: B_rest^-1 (x) RQ_b, so W_b = B_target (x) rest
  const restQ = (o, b, d, r) => { mBasisQ(_mcq, 0, d[0], d[1], d[2], r[0], r[1], r[2]); _mcq[0] = -_mcq[0]; _mcq[1] = -_mcq[1]; _mcq[2] = -_mcq[2]; aqMul(o, 0, _mcq, 0, RQ, 4 * b); return o; };
  for (const g of legs) { g.rTh = restQ(new Float64Array(4), g.ul, g.th, [-1, 0, 0]); g.rSh = restQ(new Float64Array(4), g.ll, g.sh, [-1, 0, 0]); g.rFt = restQ(new Float64Array(4), g.ft, g.fwd, [-1, 0, 0]); }
  for (const g of arms) { g.rUp = restQ(new Float64Array(4), g.ua, g.up, g.hinge); g.rFo = restQ(new Float64Array(4), g.la, g.fo, g.hinge); }
  return { legs, arms, legLen, neck, RQ, RP, L: new Float64Array(nb * 4), Wq: new Float64Array(nb * 4), Wp: new Float64Array(nb * 3), D: new Float64Array(16) };
}

// Two-bone IK in arrays: hip/shoulder a, target t, pole p; writes the middle joint to m and the reached end to e.
function mTwoBone(a, t, p, l1, l2, m, e) {
  let dx = t[0] - a[0], dy = t[1] - a[1], dz = t[2] - a[2], L = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
  dx /= L; dy /= L; dz /= L; L = clamp(L, Math.abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4);
  const x = (l1 * l1 - l2 * l2 + L * L) / (2 * L), hg = Math.sqrt(Math.max(0, l1 * l1 - x * x));
  let px = p[0] - a[0], py = p[1] - a[1], pz = p[2] - a[2]; const pd = px * dx + py * dy + pz * dz; px -= pd * dx; py -= pd * dy; pz -= pd * dz;
  let pn = Math.sqrt(px * px + py * py + pz * pz);
  if (pn < 1e-6) { px = 0; py = 0; pz = -1; const q = -dz; px -= q * dx; py -= q * dy; pz -= q * dz; pn = Math.sqrt(px * px + py * py + pz * pz) || 1; }
  px /= pn; py /= pn; pz /= pn;
  m[0] = a[0] + dx * x + px * hg; m[1] = a[1] + dy * x + py * hg; m[2] = a[2] + dz * x + pz * hg;
  e[0] = a[0] + dx * L; e[1] = a[1] + dy * L; e[2] = a[2] + dz * L;
}
// Highest support under (x, z) at or below yTop, in metres: the deck, or the top of a box prop (boards, walls) under it.
function monSupport(scene, x, z, yTop) {
  let y = (scene && scene.ground_y) || 0;
  for (const p of (scene && scene.props) || []) {
    if (p.t !== 'box' || !p.s) continue;
    const top = p.p[1] + p.s[1] / 2; if (top <= y || top > yTop + 0.03) continue;
    const c = Math.cos(p.yaw || 0), s = Math.sin(p.yaw || 0), dx = x - p.p[0], dz = z - p.p[2];
    const lx = dx * c - dz * s, lz = dx * s + dz * c;
    if (Math.abs(lx) <= p.s[0] / 2 + 0.01 && Math.abs(lz) <= p.s[2] / 2 + 0.01) y = top;
  }
  return y;
}

// ---- the per-frame solve: sim pose (world metres) -> an Anny sample (Anny units = world / k) ----
const _ma = [0, 0, 0], _mb = [0, 0, 0], _mc = [0, 0, 0], _md = [0, 0, 0], _me = [0, 0, 0], _mf = [0, 0, 0], _mg = [0, 0, 0];
const _mq1 = new Float64Array(4), _mq2 = new Float64Array(4), _mq3 = new Float64Array(4), _mq4 = new Float64Array(4);
const _mv = new Float64Array(3), _mw = new Float64Array(3);
// Pose input in plain arrays: q[16] segment quaternions (xyzw), hq[4], j[54] joints (world metres).
function monAnnySolve(R, f, Pq, Phq, Pj, body, k, scene, stat) {
  const A = ANNY, L = R.L, Wq = R.Wq, Wp = R.Wp, D = R.D, ik = 1 / k;
  for (let i = 0; i < 16; i++) D[i] = Pq[i];
  for (let kk = 0; kk < 4; kk++) aqNormQ(D, 4 * kk, D, 4 * kk, 1);
  monCarriers(D, L, Wq);
  monFK(L, [0, 0, 0], Wq, Wp);
  const J3 = (j, o) => { o[0] = Pj[3 * j] * ik; o[1] = Pj[3 * j + 1] * ik; o[2] = Pj[3 * j + 2] * ik; return o; };
  // pelvis: hip-joint centre, raised by the leg-length difference along the pelvis segment's +Y
  const lg = R.legs, simLeg = (body.thigh + body.shank + body.ankle_h) * ik;
  J3(J.hip_l, _ma); J3(J.hip_r, _mb);
  const up = _mv; mRot(D, 0, [0, 1, 0], 0, up, 0);
  const dUp = R.legLen - simLeg;
  const root = f.root;
  for (let i = 0; i < 3; i++) root[i] = (_ma[i] + _mb[i]) / 2 + up[i] * dUp - (Wp[3 * lg[0].ul + i] + Wp[3 * lg[1].ul + i]) / 2;
  for (let b = 0; b < A.nb; b++) { Wp[3 * b] += root[0]; Wp[3 * b + 1] += root[1]; Wp[3 * b + 2] += root[2]; }
  const px = _mw; mRot(D, 0, [1, 0, 0], 0, px, 0);   // pelvis lateral (+X right)
  let lift = 0;
  // legs
  for (const g of lg) {
    const sfx = g.s === 'L' ? 'l' : 'r', hip = [Wp[3 * g.ul], Wp[3 * g.ul + 1], Wp[3 * g.ul + 2]];
    const knee = J3(J[`knee_${sfx}`], [0, 0, 0]), ank = J3(J[`ankle_${sfx}`], [0, 0, 0]), heel = J3(J[`heel_${sfx}`], [0, 0, 0]), toe = J3(J[`toe_${sfx}`], [0, 0, 0]);
    mSub(knee, 0, ank, 0, _mc); const sl = mLen(_mc) || 1, adj = g.ankleH - body.ankle_h * ik;
    const tgt = [ank[0] + _mc[0] / sl * adj, ank[1] + _mc[1] / sl * adj, ank[2] + _mc[2] / sl * adj];
    // foot frame target: sole forward from the sim heel->toe
    let fx = toe[0] - heel[0], fy = toe[1] - heel[1], fz = toe[2] - heel[2];
    for (let pass = 0; pass < 2; pass++) {
      mTwoBone(hip, tgt, knee, g.l1, g.l2, _md, _me);
      const tx = _md[0] - hip[0], ty = _md[1] - hip[1], tz = _md[2] - hip[2], sx = _me[0] - _md[0], sy = _me[1] - _md[1], sz = _me[2] - _md[2];
      const tn = Math.sqrt(tx * tx + ty * ty + tz * tz), sn = Math.sqrt(sx * sx + sy * sy + sz * sz);
      // knee hinge: thigh x shin (a forward knee gives -X), blended toward the pelvis axis as the leg straightens
      let hx = (ty * sz - tz * sy) / (tn * sn) - 0.08 * px[0], hy = (tz * sx - tx * sz) / (tn * sn) - 0.08 * px[1], hz = (tx * sy - ty * sx) / (tn * sn) - 0.08 * px[2];
      mBasisQ(_mq1, 0, tx, ty, tz, hx, hy, hz); aqMul(_mq1, 0, _mq1, 0, g.rTh, 0);          // W_upperleg01
      mBasisQ(_mq2, 0, sx, sy, sz, hx, hy, hz); aqMul(_mq2, 0, _mq2, 0, g.rSh, 0);          // W_lowerleg01
      mBasisQ(_mq3, 0, fx, fy, fz, hx, hy, hz); aqMul(_mq3, 0, _mq3, 0, g.rFt, 0);          // W_foot
      // sole points against the support under them
      let pen = -Infinity, pmax = -Infinity;
      for (const sp of g.pts) {
        mRot(_mq3, 0, sp, 0, _mf, 0); const wx = _me[0] + _mf[0], wy = _me[1] + _mf[1], wz = _me[2] + _mf[2];
        const sup = monSupport(scene, wx * k, wz * k, Math.max(wy * k, heel[1] * k, toe[1] * k)) * ik;
        pen = Math.max(pen, sup - wy); pmax = Math.max(pmax, sup - wy);
      }
      if (pass === 0 && pen > 1e-4) { tgt[1] += pen; lift = Math.max(lift, pen * k); continue; }
      if (stat) stat.pen = Math.max(stat.pen || 0, pen * k);
      break;
    }
    // locals: upperleg01 under pelvis.X, upperleg02 at rest, lowerleg01 under it, lowerleg02 at rest, foot under it
    aqCMul(L, 4 * g.ul, Wq, 4 * g.pel, _mq1, 0);
    aqMul(_mq4, 0, _mq1, 0, A.rlq, 4 * g.ul2); aqCMul(L, 4 * g.ll, _mq4, 0, _mq2, 0);
    aqMul(_mq4, 0, _mq2, 0, A.rlq, 4 * g.ll2); aqCMul(L, 4 * g.ft, _mq4, 0, _mq3, 0);
  }
  // arms: shoulder from the trunk FK (clavicle, shoulder01 at rest); the hand on the simulated hand, elbow toward the sim elbow
  const cx = _mv; mRot(D, 12, [1, 0, 0], 0, cx, 0);
  for (const g of R.arms) {
    const sfx = g.s === 'L' ? 'l' : 'r', sh = [Wp[3 * g.ua], Wp[3 * g.ua + 1], Wp[3 * g.ua + 2]];
    const el = J3(J[`elbow_${sfx}`], [0, 0, 0]), hd = J3(J[`hand_${sfx}`], [0, 0, 0]);
    // the sim's ragdoll can leave an arm under the deck when lying: keep elbow and hand on or above the support
    const eS = monSupport(scene, el[0] * k, el[2] * k, el[1] * k) * ik + 0.04 * ik, hS = monSupport(scene, hd[0] * k, hd[2] * k, hd[1] * k) * ik + 0.035 * ik;
    if (el[1] < eS) el[1] = eS; if (hd[1] < hS) hd[1] = hS;
    // the sim hand point is the palm centre: the wrist sits one palm back along the forearm
    mSub(hd, 0, el, 0, _mc); const fl = mLen(_mc) || 1;
    const wt = [hd[0] - _mc[0] / fl * g.palm, hd[1] - _mc[1] / fl * g.palm, hd[2] - _mc[2] / fl * g.palm];
    mTwoBone(sh, wt, el, g.l1, g.l2, _md, _me);
    const ux = _md[0] - sh[0], uy = _md[1] - sh[1], uz = _md[2] - sh[2], fx = _me[0] - _md[0], fy = _me[1] - _md[1], fz = _me[2] - _md[2];
    const un = Math.sqrt(ux * ux + uy * uy + uz * uz), fn = Math.sqrt(fx * fx + fy * fy + fz * fz);
    const hx = (uy * fz - uz * fy) / (un * fn) + 0.08 * cx[0], hy = (uz * fx - ux * fz) / (un * fn) + 0.08 * cx[1], hz = (ux * fy - uy * fx) / (un * fn) + 0.08 * cx[2];
    mBasisQ(_mq1, 0, ux, uy, uz, hx, hy, hz); aqMul(_mq1, 0, _mq1, 0, g.rUp, 0);
    mBasisQ(_mq2, 0, fx, fy, fz, hx, hy, hz); aqMul(_mq2, 0, _mq2, 0, g.rFo, 0);
    aqCMul(L, 4 * g.ua, Wq, 4 * g.so, _mq1, 0);
    aqMul(_mq4, 0, _mq1, 0, A.rlq, 4 * g.ua2); aqCMul(L, 4 * g.la, _mq4, 0, _mq2, 0);
    if (stat) { stat.hand = Math.max(stat.hand || 0, Math.hypot(_me[0] + fx / fn * g.palm - hd[0], _me[1] + fy / fn * g.palm - hd[1], _me[2] + fz / fn * g.palm - hd[2]) * k); }
  }
  // neck and head: neck01..03 at 1/4, 1/2, 3/4 of the way from the upper-trunk segment to the head
  aqNormQ(_mq3, 0, Phq, 0, 1);
  let parQ = _mq4; parQ.set(Wq.subarray(4 * A.parent[R.neck[0]], 4 * A.parent[R.neck[0]] + 4));
  for (let i = 0; i < 4; i++) {
    const b = R.neck[i];
    if (i < 3) aqSlerp(_mq1, 0, D, 12, _mq3, 0, (i + 1) / 4); else _mq1.set(_mq3);
    aqMul(_mq2, 0, _mq1, 0, A.K, 4 * b);
    aqCMul(L, 4 * b, parQ, 0, _mq2, 0);
    parQ.set(_mq2);
  }
  // the sample poseAnny takes
  f.neutral = false; f.hcal = 0; f.segq.set(D);
  for (let j = 0; j < 30; j++) { const o = 4 * A.qb[j]; f.q[4 * j] = L[o]; f.q[4 * j + 1] = L[o + 1]; f.q[4 * j + 2] = L[o + 2]; f.q[4 * j + 3] = L[o + 3]; }
  f.fing.set(A.neutral.fing); f.mtp.fill(0);
  if (stat) stat.lift = Math.max(stat.lift || 0, lift);
  return f;
}

// After skinning: how far the lowest non-foot vertex sits below the deck (world metres); the root is raised by it.
function monAnnyPen(h, R, k, ground) {
  if (!R.bodyV) { const foot = new Set([...R.legs[0].verts, ...R.legs[1].verts]), o = []; for (let v = 0; v < ANNY.nv; v++) if (!foot.has(v)) o.push(v); R.bodyV = Int32Array.from(o); }
  const PA = h.P.array, BV = R.bodyV; let m = Infinity;
  for (let i = 0; i < BV.length; i++) { const y = PA[3 * BV[i] + 1]; if (y < m) m = y; }
  return Math.max(0, ground - m * k);
}



/* ---- Anatomy layer core (spec 13.3, 18.3 palette, 18.5 shader mirror). No THREE here: node runs this file as is
   (insite_anny_build/anat_node_test.js) to check it against annylift/anat_palette.js and the file's self-test. ---- */
const ANAT_NANNY = 104, ANAT_NROWS = 185, ANAT_TEXW = 2048;
// Every deflated b64 field of anatomy-anny.v1.json, as [object, key] pairs (inflated once, in parallel, at load).
function anatZFields(doc) {
  const L = [];
  const add = (o, keys) => { if (o) for (const k of keys) if (typeof o[k] === 'string') L.push(o[k]); };
  add(doc.parts.b64, ['g', 'nv', 'nf', 'pal', 'edit']);
  add(doc.b64, ['pos', 'idx', 'mwI', 'mwW', 'anchor', 'mdrv', 'mbeta', 'cart']);
  add(doc.rings, ['J0', 'off', 'v']);
  add(doc.costal, ['pins', 'row']);
  for (const f of doc.follow) add(f, ['anc', 'x']);
  for (const p of doc.patella) add(p, ['pts', 'skinV', 'front']);
  if (doc.selftest) add(doc.selftest.b64, ['verts', 'pos', 'pal', 'rings', 'locals', 'root']);
  return L;
}
function anatView(u8, T) { const b = (u8.byteOffset % T.BYTES_PER_ELEMENT) ? u8.slice() : u8; return new T(b.buffer, b.byteOffset, b.byteLength / T.BYTES_PER_ELEMENT); }
function fnv1a32(u8) { let h = 0x811c9dc5; for (let i = 0; i < u8.length; i++) { h ^= u8[i]; h = Math.imul(h, 0x01000193) >>> 0; } return h.toString(16).padStart(8, '0'); }

// Symmetric 3x3 eigenvectors (cyclic Jacobi), sorted by eigenvalue, descending. In: [a00,a01,a02,a11,a12,a22].
function anatEig3(c) {
  const A = [[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]], E = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 24; sweep++) {
    const off = A[0][1] * A[0][1] + A[0][2] * A[0][2] + A[1][2] * A[1][2];
    if (off < 1e-30) break;
    for (const [p, q] of [[0, 1], [0, 2], [1, 2]]) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const cs = 1 / Math.sqrt(t * t + 1), sn = t * cs;
      for (let k = 0; k < 3; k++) { const akp = A[k][p], akq = A[k][q]; A[k][p] = cs * akp - sn * akq; A[k][q] = sn * akp + cs * akq; }
      for (let k = 0; k < 3; k++) { const apk = A[p][k], aqk = A[q][k]; A[p][k] = cs * apk - sn * aqk; A[q][k] = sn * apk + cs * aqk; }
      for (let k = 0; k < 3; k++) { const ekp = E[k][p], ekq = E[k][q]; E[k][p] = cs * ekp - sn * ekq; E[k][q] = sn * ekp + cs * ekq; }
    }
  }
  return [0, 1, 2].map((i) => ({ l: A[i][i], v: [E[0][i], E[1][i], E[2][i]] })).sort((a, b) => b.l - a.l);
}

// Decode (13.3): dequantised Float32 positions, part-local zigzag indices -> Uint32 per group, area-weighted normals per
// part, fibre UVs (build.py fibre_uv: per-part PCA, v along the long axis, u across / 0.03 m, packed u/32 + 0.5) for the
// muscle groups, and the per-vertex shader attributes. Groups: 0 bone, 1 bone_measured, 2 cartilage, 3 muscle, 4 muscle_cut.
function anatDecode(doc, z) {
  const g = anatView(z(doc.parts.b64.g), Uint8Array), pnv = anatView(z(doc.parts.b64.nv), Uint32Array), pnf = anatView(z(doc.parts.b64.nf), Uint32Array);
  const ppal = anatView(z(doc.parts.b64.pal), Uint8Array), np = g.length;
  const lo = doc.q.lo, step = doc.q.step, D = anatView(z(doc.b64.pos), Uint16Array), Z = anatView(z(doc.b64.idx), Uint16Array);
  let totv = 0, totf = 0; for (let p = 0; p < np; p++) { totv += pnv[p]; totf += pnf[p]; }
  if (D.length !== totv * 3 || Z.length !== totf * 3) throw new Error('anatomy: array sizes');
  const GN = 5, gnv = new Uint32Array(GN), gnf = new Uint32Array(GN);
  for (let p = 0; p < np; p++) { if (g[p] >= GN) throw new Error('anatomy: group'); gnv[g[p]] += pnv[p]; gnf[g[p]] += pnf[p]; }
  const G = [];
  for (let k = 0; k < GN; k++) G.push({ nv: gnv[k], nf: gnf[k], pos: new Float32Array(gnv[k] * 3), nor: new Float32Array(gnv[k] * 3), idx: new Uint32Array(gnf[k] * 3), parts: [] });
  const mwI = anatView(z(doc.b64.mwI), Uint8Array), mwW = anatView(z(doc.b64.mwW), Uint8Array), anchor = anatView(z(doc.b64.anchor), Uint16Array);
  const mdrv = anatView(z(doc.b64.mdrv), Uint8Array), mbeta = anatView(z(doc.b64.mbeta), Uint8Array), cart = anatView(z(doc.b64.cart), Uint8Array);
  const nvMus = gnv[3] + gnv[4], nvCart = gnv[2];
  if (mwI.length !== nvMus * 8 || mwW.length !== nvMus * 8 || anchor.length !== nvMus || mdrv.length !== nvMus || mbeta.length !== nvMus || cart.length !== nvCart * 3) throw new Error('anatomy: weight sizes');
  for (const k of [3, 4]) { const n = gnv[k]; Object.assign(G[k], { uv: new Float32Array(n * 2), mwI0: new Uint8Array(n * 4), mwI1: new Uint8Array(n * 4), mwW0: new Uint8Array(n * 4), mwW1: new Uint8Array(n * 4), anchor: new Uint16Array(n), mdrv: new Uint8Array(n), mbeta: new Uint8Array(n) }); }
  G[0].pidx = new Uint8Array(gnv[0]); G[1].pidx = new Uint8Array(gnv[1]);
  Object.assign(G[2], { pidx2: new Uint8Array(nvCart * 2), pw: new Uint8Array(nvCart) });
  // global vertex o -> (group, local index); musI / cartI count in file order, as the attributes are stored
  const vg = new Uint8Array(totv), vl = new Uint32Array(totv), gv = new Uint32Array(GN), gf = new Uint32Array(GN);
  let o = 0, fo = 0, musI = 0, cartI = 0;
  const acc = new Float64Array(3);
  for (let p = 0; p < np; p++) {
    const k = g[p], Gk = G[k], v0 = gv[k], n = pnv[p];
    acc[0] = acc[1] = acc[2] = 0;
    for (let i = 0; i < n; i++, o++) {
      const l = v0 + i; vg[o] = k; vl[o] = l;
      for (let c = 0; c < 3; c++) { acc[c] = (acc[c] + D[3 * o + c]) & 0xFFFF; Gk.pos[3 * l + c] = lo[c] + step[c] * acc[c]; }
      if (k <= 1) Gk.pidx[l] = ppal[p];
      else if (k === 2) { Gk.pidx2[2 * l] = cart[3 * cartI]; Gk.pidx2[2 * l + 1] = cart[3 * cartI + 1]; Gk.pw[l] = cart[3 * cartI + 2]; cartI++; }
      else {
        for (let c = 0; c < 4; c++) { Gk.mwI0[4 * l + c] = mwI[8 * musI + c]; Gk.mwI1[4 * l + c] = mwI[8 * musI + 4 + c]; Gk.mwW0[4 * l + c] = mwW[8 * musI + c]; Gk.mwW1[4 * l + c] = mwW[8 * musI + 4 + c]; }
        Gk.anchor[l] = anchor[musI]; Gk.mdrv[l] = mdrv[musI]; Gk.mbeta[l] = mbeta[musI]; musI++;
      }
    }
    // indices: zigzag deltas of the part's corner stream, restarted per part
    let ci = 0; const nf3 = pnf[p] * 3, f0 = gf[k] * 3;
    for (let i = 0; i < nf3; i++) {
      const zz = Z[3 * fo + i], d = (zz >>> 1) ^ -(zz & 1); ci += d;
      if (ci < 0 || ci >= n) throw new Error('anatomy: index range');
      Gk.idx[f0 + i] = v0 + ci;
    }
    Gk.parts.push({ p, v0, nv: n, f0: gf[k], nf: pnf[p] });
    gv[k] += n; gf[k] += pnf[p]; fo += pnf[p];
  }
  // normals (area-weighted) and fibre UVs, per group (parts never share vertices, so this is per part)
  for (let k = 0; k < GN; k++) {
    const Gk = G[k], P = Gk.pos, I = Gk.idx, N = new Float64Array(Gk.nv * 3);
    for (let f = 0; f < I.length; f += 3) {
      const a = 3 * I[f], b = 3 * I[f + 1], c = 3 * I[f + 2];
      const ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2], vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
      const cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
      N[a] += cx; N[a + 1] += cy; N[a + 2] += cz; N[b] += cx; N[b + 1] += cy; N[b + 2] += cz; N[c] += cx; N[c + 1] += cy; N[c + 2] += cz;
    }
    for (let i = 0; i < N.length; i += 3) { const l = Math.hypot(N[i], N[i + 1], N[i + 2]) || 1; Gk.nor[i] = N[i] / l; Gk.nor[i + 1] = N[i + 1] / l; Gk.nor[i + 2] = N[i + 2] / l; }
    if (k < 3) continue;
    for (const pt of Gk.parts) {
      const m = [0, 0, 0], C = [0, 0, 0, 0, 0, 0];
      for (let i = pt.v0; i < pt.v0 + pt.nv; i++) for (let c = 0; c < 3; c++) m[c] += P[3 * i + c];
      for (let c = 0; c < 3; c++) m[c] /= pt.nv;
      for (let i = pt.v0; i < pt.v0 + pt.nv; i++) {
        const x = P[3 * i] - m[0], y = P[3 * i + 1] - m[1], zq = P[3 * i + 2] - m[2];
        C[0] += x * x; C[1] += x * y; C[2] += x * zq; C[3] += y * y; C[4] += y * zq; C[5] += zq * zq;
      }
      const e = anatEig3(C), e1 = e[0].v, e2 = e[1].v;
      let mn = Infinity, mx = -Infinity;
      for (let i = pt.v0; i < pt.v0 + pt.nv; i++) { const p1 = (P[3 * i] - m[0]) * e1[0] + (P[3 * i + 1] - m[1]) * e1[1] + (P[3 * i + 2] - m[2]) * e1[2]; if (p1 < mn) mn = p1; if (p1 > mx) mx = p1; }
      for (let i = pt.v0; i < pt.v0 + pt.nv; i++) {
        const x = P[3 * i] - m[0], y = P[3 * i + 1] - m[1], zq = P[3 * i + 2] - m[2];
        const p1 = x * e1[0] + y * e1[1] + zq * e1[2], p2 = x * e2[0] + y * e2[1] + zq * e2[2];
        Gk.uv[2 * i] = Math.min(1, Math.max(0, p2 / 0.03 / 32 + 0.5)); Gk.uv[2 * i + 1] = Math.min(1, Math.max(0, (p1 - mn) / (mx - mn + 1e-9)));
      }
    }
  }
  return { G, np, totv, totf, vg, vl, names: doc.parts.name, g, ppal };
}

// ---- palette (18.3, 15.3 step 9): allocation-free port of annylift/anat_palette.js, same arithmetic in doubles ----
function anatPrepare(doc, z) {
  const side = (arr) => { const out = {}; for (const e of arr) out[e.side] = e; return out; };
  const pf = side(doc.pins.forearm), psc = side(doc.pins.sc), fl = side(doc.follow), pat = side(doc.patella);
  const cp = anatView(z(doc.costal.pins), Float32Array), cr = anatView(z(doc.costal.row), Uint8Array);
  const pin = (ps) => (ps[0] === 'head' ? { t: 0, i: ps[1] } : ps[0] === 'bone' || ps[0] === 'row' ? { t: 1, i: ps[1], p: Float64Array.from(ps[2]) } : ps[0] === 'ring' ? { t: 2, i: ps[1] } : null);
  const F = (a) => Float64Array.from(a);
  const virt = doc.palette.virtual.map((o) => {
    const s = o.id[o.id.length - 1], e = { row: o.row, kind: o.kind, id: o.id };
    switch (o.kind) {
      case 'level': Object.assign(e, { a: o.a, b: o.b, u: o.u, c: F(o.c) }); break;
      case 'seg': {
        const pa = pin(o.pa), pb = pin(o.pb); if (!pa || !pb) throw new Error('anatomy: pin ' + o.pa[0] + '/' + o.pb[0]);
        Object.assign(e, { pa, pb, P0: F(o.P0), D0: F(o.D0), rollBone: o.roll[0] === 'bone', roll: o.roll[1] }); break;
      }
      case 'radius': Object.assign(e, { us0: o.ustar[0], us1: o.ustar[1], usRow: o.ustar[2], usP0: F(o.ustar[3]), usD0: F(o.ustar[4]), la02: o.la02, RH0: F(pf[s].RH0), DR0: F(pf[s].DR0) }); break;
      case 'ulna': Object.assign(e, { la01: o.la01, radRow: o.radRow, humRow: o.humRow, E0: F(o.E0), UH0: F(pf[s].UH0) }); break;
      case 'sternum': Object.assign(e, { rotRow: o.rotRow, m0: F(o.m0), rows: Int32Array.from(cr), pins: Float64Array.from(cp) }); break;
      case 'follow': Object.assign(e, { pivot: o.pivot, anc: anatView(z(fl[s].anc), Uint16Array), x: Float64Array.from(anatView(z(fl[s].x), Float32Array)), g0: F(fl[s].g0) }); break;
      case 'clav': Object.assign(e, { scRow: o.scRow, scapRow: o.scapRow, clav: o.clav, SC0: F(psc[s].SC0), AC0: F(psc[s].AC0) }); break;
      case 'patella': Object.assign(e, { femRow: o.femRow, tibRow: o.tibRow, f: o.f, knee: o.knee, k0: F(o.k0), pts: Float64Array.from(anatView(z(pat[s].pts), Float32Array)), skinV: anatView(z(pat[s].skinV), Uint16Array), front: anatView(z(pat[s].front), Uint16Array), margin: pat[s].margin }); break;
      case 'carpal': Object.assign(e, { la02: o.la02, wrist: o.wrist, f: o.f, c0: F(o.c0) }); break;
      default: throw new Error('anatomy: palette kind ' + o.kind);
    }
    return e;
  });
  if (doc.palette.nAnny !== ANAT_NANNY || virt.length + ANAT_NANNY !== ANAT_NROWS) throw new Error('anatomy: palette size');
  virt.forEach((e, i) => { if (e.row !== ANAT_NANNY + i) throw new Error('anatomy: palette order'); });
  return {
    virt, n: doc.rings.n, J0: Float64Array.from(anatView(z(doc.rings.J0), Float32Array)), off: anatView(z(doc.rings.off), Uint32Array), rv: anatView(z(doc.rings.v), Uint16Array),
    P: new Float64Array(ANAT_NROWS * 12), dq: new Float64Array(ANAT_NANNY * 4), J: new Float64Array(doc.rings.n * 3), ms: 0,
  };
}
const _pq = new Float64Array(16), _pR = new Float64Array(36), _pv = new Float64Array(24), _pU = new Float64Array(12), _pM = new Float64Array(12);
function apQ2M(q, qo, R, ro) {   // normalise, then the rotation matrix (frames q2m)
  const n = Math.sqrt(q[qo] * q[qo] + q[qo + 1] * q[qo + 1] + q[qo + 2] * q[qo + 2] + q[qo + 3] * q[qo + 3]);
  const x = q[qo] / n, y = q[qo + 1] / n, z = q[qo + 2] / n, w = q[qo + 3] / n;
  R[ro] = 1 - 2 * (y * y + z * z); R[ro + 1] = 2 * (x * y - w * z); R[ro + 2] = 2 * (x * z + w * y);
  R[ro + 3] = 2 * (x * y + w * z); R[ro + 4] = 1 - 2 * (x * x + z * z); R[ro + 5] = 2 * (y * z - w * x);
  R[ro + 6] = 2 * (x * z - w * y); R[ro + 7] = 2 * (y * z + w * x); R[ro + 8] = 1 - 2 * (x * x + y * y);
}
function apSlerp(o, oo, a, ao, b, bo, u) {   // anat_palette.qslerp (normalised output)
  let bx = b[bo], by = b[bo + 1], bz = b[bo + 2], bw = b[bo + 3];
  let d = a[ao] * bx + a[ao + 1] * by + a[ao + 2] * bz + a[ao + 3] * bw;
  if (d < 0) { bx = -bx; by = -by; bz = -bz; bw = -bw; d = -d; }
  const th = Math.acos(Math.min(1, Math.max(-1, d))), s = Math.sin(th);
  let wa, wb; if (s < 1e-9) { wa = 1 - u; wb = u; } else { wa = Math.sin((1 - u) * th) / s; wb = Math.sin(u * th) / s; }
  const x = wa * a[ao] + wb * bx, y = wa * a[ao + 1] + wb * by, z = wa * a[ao + 2] + wb * bz, w = wa * a[ao + 3] + wb * bw;
  const n = Math.hypot(x, y, z, w); o[oo] = x / n; o[oo + 1] = y / n; o[oo + 2] = z / n; o[oo + 3] = w / n;
}
function apM2Q(R, ro, q, qo) {   // Shepperd, w >= 0 (frames.mat2q)
  const m0 = R[ro], m1 = R[ro + 1], m2 = R[ro + 2], m3 = R[ro + 3], m4 = R[ro + 4], m5 = R[ro + 5], m6 = R[ro + 6], m7 = R[ro + 7], m8 = R[ro + 8];
  const tr = m0 + m4 + m8; let k = 0, best = m0;
  if (m4 > best) { k = 1; best = m4; } if (m8 > best) { k = 2; best = m8; } if (tr > best) k = 3;
  let x, y, z, w, r;
  if (k === 3) { r = Math.sqrt(1 + tr) * 2; x = (m7 - m5) / r; y = (m2 - m6) / r; z = (m3 - m1) / r; w = 0.25 * r; }
  else if (k === 0) { r = Math.sqrt(1 + m0 - m4 - m8) * 2; x = 0.25 * r; y = (m1 + m3) / r; z = (m2 + m6) / r; w = (m7 - m5) / r; }
  else if (k === 1) { r = Math.sqrt(1 + m4 - m0 - m8) * 2; x = (m1 + m3) / r; y = 0.25 * r; z = (m5 + m7) / r; w = (m2 - m6) / r; }
  else { r = Math.sqrt(1 + m8 - m0 - m4) * 2; x = (m2 + m6) / r; y = (m5 + m7) / r; z = 0.25 * r; w = (m3 - m1) / r; }
  const n = Math.hypot(x, y, z, w); x /= n; y /= n; z /= n; w /= n;
  if (w < 0) { x = -x; y = -y; z = -z; w = -w; }
  q[qo] = x; q[qo + 1] = y; q[qo + 2] = z; q[qo + 3] = w;
}
function apRotBetween(ux, uy, uz, vx, vy, vz, R, ro) {   // frames.rot_between -> matrix
  let n = Math.hypot(ux, uy, uz); ux /= n; uy /= n; uz /= n;
  n = Math.hypot(vx, vy, vz); vx /= n; vy /= n; vz /= n;
  const c = ux * vx + uy * vy + uz * vz;
  _pq[12] = uy * vz - uz * vy; _pq[13] = uz * vx - ux * vz; _pq[14] = ux * vy - uy * vx; _pq[15] = 1 + c;
  if (1 + c < 1e-12) {
    const ax = Math.abs(ux) < 0.9 ? 1 : 0, ay = ax ? 0 : 1;
    let px = uy * 0 - uz * ay, py = uz * ax - ux * 0, pz = ux * ay - uy * ax; n = Math.hypot(px, py, pz); px /= n; py /= n; pz /= n;
    _pq[12] = px; _pq[13] = py; _pq[14] = pz; _pq[15] = 0;
  }
  apQ2M(_pq, 12, R, ro);
}
function apM3Mul(A, ao, B, bo, O, oo) {   // O = A B (O must not alias A or B)
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) O[oo + 3 * i + j] = A[ao + 3 * i] * B[bo + j] + A[ao + 3 * i + 1] * B[bo + 3 + j] + A[ao + 3 * i + 2] * B[bo + 6 + j];
}
function apRowR(P, po, R, ro) { R[ro] = P[po]; R[ro + 1] = P[po + 1]; R[ro + 2] = P[po + 2]; R[ro + 3] = P[po + 4]; R[ro + 4] = P[po + 5]; R[ro + 5] = P[po + 6]; R[ro + 6] = P[po + 8]; R[ro + 7] = P[po + 9]; R[ro + 8] = P[po + 10]; }
function apXf(T, to, x, y, z, o, oo) {
  o[oo] = T[to] * x + T[to + 1] * y + T[to + 2] * z + T[to + 3];
  o[oo + 1] = T[to + 4] * x + T[to + 5] * y + T[to + 6] * z + T[to + 7];
  o[oo + 2] = T[to + 8] * x + T[to + 9] * y + T[to + 10] * z + T[to + 11];
}
function apMk(P, po, R, ro, tx, ty, tz) {   // row = [R | t], t given; P may not alias R
  P[po] = R[ro]; P[po + 1] = R[ro + 1]; P[po + 2] = R[ro + 2]; P[po + 3] = tx;
  P[po + 4] = R[ro + 3]; P[po + 5] = R[ro + 4]; P[po + 6] = R[ro + 5]; P[po + 7] = ty;
  P[po + 8] = R[ro + 6]; P[po + 9] = R[ro + 7]; P[po + 10] = R[ro + 8]; P[po + 11] = tz;
}
function apMkC(P, po, R, ro, cx, cy, cz, px, py, pz) {   // row = [R | p - R c]
  apMk(P, po, R, ro, px - (R[ro] * cx + R[ro + 1] * cy + R[ro + 2] * cz), py - (R[ro + 3] * cx + R[ro + 4] * cy + R[ro + 5] * cz), pz - (R[ro + 6] * cx + R[ro + 7] * cy + R[ro + 8] * cz));
}
// seg(Pa, Pb, P0, D0, Rc) into row P[po]; Pa, Pb in v[0..5], Rc in _pR[ro..]
function apSeg(P, po, v, P0, D0, R, rco) {
  let ux = D0[0] - P0[0], uy = D0[1] - P0[1], uz = D0[2] - P0[2]; let n = Math.hypot(ux, uy, uz); ux /= n; uy /= n; uz /= n;
  const rx = R[rco] * ux + R[rco + 1] * uy + R[rco + 2] * uz, ry = R[rco + 3] * ux + R[rco + 4] * uy + R[rco + 5] * uz, rz = R[rco + 6] * ux + R[rco + 7] * uy + R[rco + 8] * uz;
  let wx = v[3] - v[0], wy = v[4] - v[1], wz = v[5] - v[2]; n = Math.hypot(wx, wy, wz); wx /= n; wy /= n; wz /= n;
  apRotBetween(rx, ry, rz, wx, wy, wz, R, 27); apM3Mul(R, 27, R, rco, R, 18);
  apMkC(P, po, R, 18, (P0[0] + D0[0]) / 2, (P0[1] + D0[1]) / 2, (P0[2] + D0[2]) / 2, (v[0] + v[3]) / 2, (v[1] + v[4]) / 2, (v[2] + v[5]) / 2);
}
// st = { S: Float32Array(104*12), wq, wp, rwq, skinM: Float32Array(nv*12), V, N }; out: Float32Array (>= 185*12), written
function anatPaletteP(A, st, out, rings) {
  const P = A.P, dq = A.dq, J = A.J, S = st.S, M = st.skinM, wq = st.wq, rwq = st.rwq, wp = st.wp, R = _pR, v = _pv;
  for (let i = 0; i < ANAT_NANNY * 12; i++) P[i] = S[i];
  for (let b = 0; b < ANAT_NANNY; b++) {   // dq_b = wq_b (x) conj(restWorldQuat_b)
    const ax = wq[4 * b], ay = wq[4 * b + 1], az = wq[4 * b + 2], aw = wq[4 * b + 3], bx = -rwq[4 * b], by = -rwq[4 * b + 1], bz = -rwq[4 * b + 2], bw = rwq[4 * b + 3];
    dq[4 * b] = aw * bx + ax * bw + ay * bz - az * by; dq[4 * b + 1] = aw * by - ax * bz + ay * bw + az * bx;
    dq[4 * b + 2] = aw * bz + ax * by - ay * bx + az * bw; dq[4 * b + 3] = aw * bw - ax * bx - ay * by - az * bz;
  }
  // ring-posed joint points J = mean_v(M_v) J0
  for (let k = 0; k < A.n; k++) {
    const s = A.off[k], e = A.off[k + 1]; _pM.fill(0);
    for (let i = s; i < e; i++) { const o = 12 * A.rv[i]; for (let c = 0; c < 12; c++) _pM[c] += M[o + c]; }
    for (let c = 0; c < 12; c++) _pM[c] /= (e - s);
    apXf(_pM, 0, A.J0[3 * k], A.J0[3 * k + 1], A.J0[3 * k + 2], J, 3 * k);
    if (rings) { rings[3 * k] = J[3 * k]; rings[3 * k + 1] = J[3 * k + 1]; rings[3 * k + 2] = J[3 * k + 2]; }
  }
  const pinInto = (pn, o) => {
    if (pn.t === 0) { v[o] = wp[3 * pn.i]; v[o + 1] = wp[3 * pn.i + 1]; v[o + 2] = wp[3 * pn.i + 2]; }
    else if (pn.t === 1) apXf(P, 12 * pn.i, pn.p[0], pn.p[1], pn.p[2], v, o);
    else { v[o] = J[3 * pn.i]; v[o + 1] = J[3 * pn.i + 1]; v[o + 2] = J[3 * pn.i + 2]; }
  };
  for (const e of A.virt) {
    const jo = 12 * e.row;
    switch (e.kind) {
      case 'level': {
        apSlerp(_pq, 0, dq, 4 * e.a, dq, 4 * e.b, e.u); apQ2M(_pq, 0, R, 0);
        apXf(P, 12 * e.a, e.c[0], e.c[1], e.c[2], v, 0); apXf(P, 12 * e.b, e.c[0], e.c[1], e.c[2], v, 3);
        const u = e.u;
        apMkC(P, jo, R, 0, e.c[0], e.c[1], e.c[2], (1 - u) * v[0] + u * v[3], (1 - u) * v[1] + u * v[4], (1 - u) * v[2] + u * v[5]);
        break;
      }
      case 'seg':
        pinInto(e.pa, 0); pinInto(e.pb, 3);
        if (e.rollBone) apQ2M(dq, 4 * e.roll, R, 0); else apRowR(P, 12 * e.roll, R, 0);
        apSeg(P, jo, v, e.P0, e.D0, R, 0);
        break;
      case 'radius': {
        v[0] = wp[3 * e.us0]; v[1] = wp[3 * e.us0 + 1]; v[2] = wp[3 * e.us0 + 2]; v[3] = wp[3 * e.us1]; v[4] = wp[3 * e.us1 + 1]; v[5] = wp[3 * e.us1 + 2];
        apRowR(P, 12 * e.usRow, R, 0); apSeg(_pU, 0, v, e.usP0, e.usD0, R, 0);
        apXf(_pU, 0, e.RH0[0], e.RH0[1], e.RH0[2], v, 0); apXf(P, 12 * e.la02, e.DR0[0], e.DR0[1], e.DR0[2], v, 3);
        apQ2M(dq, 4 * e.la02, R, 0); apSeg(P, jo, v, e.RH0, e.DR0, R, 0);
        break;
      }
      case 'ulna':
        v[0] = wp[3 * e.la01]; v[1] = wp[3 * e.la01 + 1]; v[2] = wp[3 * e.la01 + 2]; apXf(P, 12 * e.radRow, e.UH0[0], e.UH0[1], e.UH0[2], v, 3);
        apRowR(P, 12 * e.humRow, R, 0); apSeg(P, jo, v, e.E0, e.UH0, R, 0);
        break;
      case 'sternum': {
        apRowR(P, 12 * e.rotRow, R, 0); let mx = 0, my = 0, mz = 0; const nr = e.rows.length;
        for (let i = 0; i < nr; i++) { apXf(P, 12 * e.rows[i], e.pins[3 * i], e.pins[3 * i + 1], e.pins[3 * i + 2], v, 0); mx += v[0]; my += v[1]; mz += v[2]; }
        apMkC(P, jo, R, 0, e.m0[0], e.m0[1], e.m0[2], mx / nr, my / nr, mz / nr);
        break;
      }
      case 'follow': {
        const g0 = e.g0; apXf(P, 12 * e.pivot, g0[0], g0[1], g0[2], v, 0);
        const gx = v[0], gy = v[1], gz = v[2];
        let Sxx = 0, Sxy = 0, Sxz = 0, Syx = 0, Syy = 0, Syz = 0, Szx = 0, Szy = 0, Szz = 0, shift = 0;
        for (let i = 0; i < e.anc.length; i++) {
          const x0 = e.x[3 * i], x1 = e.x[3 * i + 1], x2 = e.x[3 * i + 2];
          apXf(M, 12 * e.anc[i], x0, x1, x2, v, 3);
          const ax = x0 - g0[0], ay = x1 - g0[1], az = x2 - g0[2], bx = v[3] - gx, by = v[4] - gy, bz = v[5] - gz;
          Sxx += ax * bx; Sxy += ax * by; Sxz += ax * bz; Syx += ay * bx; Syy += ay * by; Syz += ay * bz; Szx += az * bx; Szy += az * by; Szz += az * bz;
          shift += Math.hypot(ax, ay, az) * Math.hypot(bx, by, bz);
        }
        const N00 = Sxx - Syy - Szz + shift, N01 = Sxy + Syx, N02 = Szx + Sxz, N03 = Syz - Szy;
        const N11 = -Sxx + Syy - Szz + shift, N12 = Syz + Szy, N13 = Szx - Sxz;
        const N22 = -Sxx - Syy + Szz + shift, N23 = Sxy - Syx, N33 = Sxx + Syy + Szz + shift;
        let q0 = dq[4 * e.pivot], q1 = dq[4 * e.pivot + 1], q2 = dq[4 * e.pivot + 2], q3 = dq[4 * e.pivot + 3];
        for (let it = 0; it < 12; it++) {
          const t0 = N00 * q0 + N01 * q1 + N02 * q2 + N03 * q3, t1 = N01 * q0 + N11 * q1 + N12 * q2 + N13 * q3;
          const t2 = N02 * q0 + N12 * q1 + N22 * q2 + N23 * q3, t3 = N03 * q0 + N13 * q1 + N23 * q2 + N33 * q3;
          const n = Math.hypot(t0, t1, t2, t3); q0 = t0 / n; q1 = t1 / n; q2 = t2 / n; q3 = t3 / n;
        }
        _pq[0] = q0; _pq[1] = q1; _pq[2] = q2; _pq[3] = q3; apQ2M(_pq, 0, R, 0);
        apMkC(P, jo, R, 0, g0[0], g0[1], g0[2], gx, gy, gz);
        break;
      }
      case 'clav': {
        apXf(P, 12 * e.scRow, e.SC0[0], e.SC0[1], e.SC0[2], v, 0); apXf(P, 12 * e.scapRow, e.AC0[0], e.AC0[1], e.AC0[2], v, 3);
        apQ2M(dq, 4 * e.clav, R, 9);
        let ux = e.AC0[0] - e.SC0[0], uy = e.AC0[1] - e.SC0[1], uz = e.AC0[2] - e.SC0[2]; let n = Math.hypot(ux, uy, uz); ux /= n; uy /= n; uz /= n;
        const rx = R[9] * ux + R[10] * uy + R[11] * uz, ry = R[12] * ux + R[13] * uy + R[14] * uz, rz = R[15] * ux + R[16] * uy + R[17] * uz;
        let wx = v[3] - v[0], wy = v[4] - v[1], wz = v[5] - v[2]; n = Math.hypot(wx, wy, wz); wx /= n; wy /= n; wz /= n;
        apRotBetween(rx, ry, rz, wx, wy, wz, R, 27); apM3Mul(R, 27, R, 9, R, 0);
        apMkC(P, jo, R, 0, e.SC0[0], e.SC0[1], e.SC0[2], v[0], v[1], v[2]);
        break;
      }
      case 'patella': {
        apRowR(P, 12 * e.femRow, R, 0); apM2Q(R, 0, _pq, 4); apRowR(P, 12 * e.tibRow, R, 0); apM2Q(R, 0, _pq, 8);
        apSlerp(_pq, 0, _pq, 4, _pq, 8, e.f); apQ2M(_pq, 0, R, 0);
        const k0 = e.k0, kb = 3 * e.knee;
        let tx = wp[kb] - (R[0] * k0[0] + R[1] * k0[1] + R[2] * k0[2]), ty = wp[kb + 1] - (R[3] * k0[0] + R[4] * k0[1] + R[5] * k0[2]), tz = wp[kb + 2] - (R[6] * k0[0] + R[7] * k0[1] + R[8] * k0[2]);
        if (e.pts.length) {
          const V = st.V, NN = st.N; let dmax = -Infinity;
          for (let i = 0; i < e.skinV.length; i++) {
            const a = e.pts[3 * i], b = e.pts[3 * i + 1], c = e.pts[3 * i + 2], vv = 3 * e.skinV[i];
            const px = R[0] * a + R[1] * b + R[2] * c, py = R[3] * a + R[4] * b + R[5] * c, pz = R[6] * a + R[7] * b + R[8] * c;
            const d = (px + tx - V[vv]) * NN[vv] + (py + ty - V[vv + 1]) * NN[vv + 1] + (pz + tz - V[vv + 2]) * NN[vv + 2] + e.margin;
            if (d > dmax) dmax = d;
          }
          const delta = Math.max(0, dmax); let nx = 0, ny = 0, nz = 0;
          for (let i = 0; i < e.front.length; i++) { const vv = 3 * e.front[i]; nx += NN[vv]; ny += NN[vv + 1]; nz += NN[vv + 2]; }
          const n = Math.hypot(nx, ny, nz); tx -= delta * nx / n; ty -= delta * ny / n; tz -= delta * nz / n;
        }
        apMk(P, jo, R, 0, tx, ty, tz);
        break;
      }
      case 'carpal': {
        apSlerp(_pq, 0, dq, 4 * e.la02, dq, 4 * e.wrist, e.f); apQ2M(_pq, 0, R, 0);
        const wb = 3 * e.wrist; apMkC(P, jo, R, 0, e.c0[0], e.c0[1], e.c0[2], wp[wb], wp[wb + 1], wp[wb + 2]);
        break;
      }
    }
  }
  for (let i = 0; i < ANAT_NROWS * 12; i++) out[i] = P[i];
  return out;
}

// CPU mirror of the 18.5 vertex shader for one vertex (group k, local index l): Float32 storage of every texture and
// attribute, the /ws renormalisation of the u8 weights, the mbeta mix and the anchor-plane clamp. pal: Float32 (185*12);
// skinP: the uSkinP texel data (Float32, 8 floats per skin vertex). Returns [x, y, z] in metres.
function anatShadeVertex(Dc, k, l, pal, skinP, eps) {
  const G = Dc.G[k], f = Math.fround, x = G.pos[3 * l], y = G.pos[3 * l + 1], z = G.pos[3 * l + 2];
  const app = (m) => [f(m[0] * x + m[1] * y + m[2] * z + m[3]), f(m[4] * x + m[5] * y + m[6] * z + m[7]), f(m[8] * x + m[9] * y + m[10] * z + m[11])];
  const row = (j) => pal.subarray(12 * j, 12 * j + 12);
  if (k <= 1) return app(row(G.pidx[l]));
  if (k === 2) {
    const ra = row(G.pidx2[2 * l]), rb = row(G.pidx2[2 * l + 1]), w = G.pw[l] / 255, m = new Array(12);
    for (let c = 0; c < 12; c++) m[c] = f((1 - w) * ra[c] + w * rb[c]);
    return app(m);
  }
  const I = [...G.mwI0.subarray(4 * l, 4 * l + 4), ...G.mwI1.subarray(4 * l, 4 * l + 4)], W = [...G.mwW0.subarray(4 * l, 4 * l + 4), ...G.mwW1.subarray(4 * l, 4 * l + 4)];
  let ws = 0; for (let c = 0; c < 8; c++) ws += W[c];
  const m = new Array(12).fill(0);
  for (let s = 0; s < 8; s++) { const r = row(I[s]), w = W[s] / ws; for (let c = 0; c < 12; c++) m[c] += w * r[c]; }
  const q = row(G.mdrv[l]), beta = G.mbeta[l] / 255;
  for (let c = 0; c < 12; c++) m[c] = f(m[c] * (1 - beta) + q[c] * beta);
  const p = app(m), a = 8 * G.anchor[l];
  const d = Math.max((p[0] - skinP[a]) * skinP[a + 4] + (p[1] - skinP[a + 1]) * skinP[a + 5] + (p[2] - skinP[a + 2]) * skinP[a + 6] + eps, 0);
  return [p[0] - d * skinP[a + 4], p[1] - d * skinP[a + 5], p[2] - d * skinP[a + 6]];
}


  const ANAT_U = { uPal: { value: null }, uSkinP: { value: null }, uEps: { value: 0.0015 }, uWin: { value: 0 }, uWinB: { value: new THREE.Vector4(0.035, 0, 0, -0.02) }, uDbg: { value: 0 } };

const ANAT_GLSL_PAL = `
uniform highp sampler2D uPal;
varying vec3 vRest;
vec3 gP; vec3 gN;
ivec2 anatTix(int i) { return ivec2(i & 2047, i >> 11); }
void anatPal3(int j, out vec4 a, out vec4 b, out vec4 c) {
  a = texelFetch(uPal, anatTix(3 * j), 0); b = texelFetch(uPal, anatTix(3 * j + 1), 0); c = texelFetch(uPal, anatTix(3 * j + 2), 0); }
void anatApply(vec4 m0, vec4 m1, vec4 m2) {
  vec4 p4 = vec4(position, 1.0);
  gP = vec3(dot(m0, p4), dot(m1, p4), dot(m2, p4));
  gN = normalize(vec3(dot(m0.xyz, normal), dot(m1.xyz, normal), dot(m2.xyz, normal)));
  vRest = position; }
`;
const ANAT_GLSL = {
  rigid: ANAT_GLSL_PAL + `
attribute float pidx;
void anatXform() { vec4 a, b, c; anatPal3(int(pidx + 0.5), a, b, c); anatApply(a, b, c); }
`,
  blend2: ANAT_GLSL_PAL + `
attribute vec2 pidx2; attribute float pw;
void anatXform() {
  vec4 a, b, c, d, e, f; anatPal3(int(pidx2.x + 0.5), a, b, c); anatPal3(int(pidx2.y + 0.5), d, e, f);
  anatApply(mix(a, d, pw), mix(b, e, pw), mix(c, f, pw)); }
`,
  skin: ANAT_GLSL_PAL + `
uniform highp sampler2D uSkinP; uniform float uEps;
attribute vec4 mwI0, mwI1, mwW0, mwW1; attribute float anchor, mdrv, mbeta;
varying float vAnatD;
void anatAddW(float j, float w, inout vec4 m0, inout vec4 m1, inout vec4 m2) {
  vec4 a, b, c; anatPal3(int(j + 0.5), a, b, c); m0 += w * a; m1 += w * b; m2 += w * c; }
void anatXform() {
  float ws = dot(mwW0, vec4(1.0)) + dot(mwW1, vec4(1.0));
  vec4 m0 = vec4(0.0), m1 = vec4(0.0), m2 = vec4(0.0);
  anatAddW(mwI0.x, mwW0.x / ws, m0, m1, m2); anatAddW(mwI0.y, mwW0.y / ws, m0, m1, m2);
  anatAddW(mwI0.z, mwW0.z / ws, m0, m1, m2); anatAddW(mwI0.w, mwW0.w / ws, m0, m1, m2);
  anatAddW(mwI1.x, mwW1.x / ws, m0, m1, m2); anatAddW(mwI1.y, mwW1.y / ws, m0, m1, m2);
  anatAddW(mwI1.z, mwW1.z / ws, m0, m1, m2); anatAddW(mwI1.w, mwW1.w / ws, m0, m1, m2);
  vec4 q0, q1, q2; anatPal3(int(mdrv + 0.5), q0, q1, q2);
  m0 = mix(m0, q0, mbeta); m1 = mix(m1, q1, mbeta); m2 = mix(m2, q2, mbeta);
  anatApply(m0, m1, m2);
  int ai = int(anchor + 0.5);
  vec3 aP = texelFetch(uSkinP, anatTix(2 * ai), 0).xyz, aN = texelFetch(uSkinP, anatTix(2 * ai + 1), 0).xyz;
  vAnatD = dot(gP - aP, aN);
  gP -= max(vAnatD + uEps, 0.0) * aN;          // safety clamp: never closer than eps under the posed skin
}
`,
};

let anatShaderMiss = null;
function anatRep(src, a, b, what) {
  if (src.indexOf(a) < 0) { anatShaderMiss = anatShaderMiss || `${what}: ${a}`; return src; }
  return src.replace(a, b);
}
function anatShaderCheck() {   // 18.5: every replaced include must exist in this three.js build, or the anatomy stays off
  const L = THREE.ShaderLib;
  const need = [[L.physical.vertexShader, '#include <beginnormal_vertex>'], [L.physical.vertexShader, '#include <begin_vertex>'], [L.physical.vertexShader, 'void main() {'],
    [L.physical.fragmentShader, '#include <output_fragment>'], [L.physical.fragmentShader, '#include <map_fragment>'], [L.physical.fragmentShader, '#include <bumpmap_pars_fragment>'], [L.physical.fragmentShader, 'void main() {'], [L.depth.vertexShader, '#include <begin_vertex>'], [L.depth.vertexShader, 'void main() {']];
  return need.every(([s, a]) => s.indexOf(a) >= 0);
}
function anatPatch(mat, mode, depth) {
  mat.onBeforeCompile = (sh) => {
    sh.uniforms.uPal = ANAT_U.uPal; sh.uniforms.uSkinP = ANAT_U.uSkinP; sh.uniforms.uEps = ANAT_U.uEps;
    let vs = anatRep(sh.vertexShader, 'void main() {', ANAT_GLSL[mode] + '\nvoid main() {', mode);
    if (depth) vs = anatRep(vs, '#include <begin_vertex>', 'anatXform(); vec3 transformed = gP;', mode);
    else {   // transform once per vertex: beginnormal precedes begin_vertex in the physical / standard shaders
      vs = anatRep(vs, '#include <beginnormal_vertex>', 'anatXform(); vec3 objectNormal = gN;', mode);
      vs = anatRep(vs, '#include <begin_vertex>', 'vec3 transformed = gP;', mode);
    }
    sh.vertexShader = vs;
    if (depth) return;
    let fs = sh.fragmentShader;
    if (mode === 'skin') {
      sh.uniforms.uWin = ANAT_U.uWin; sh.uniforms.uWinB = ANAT_U.uWinB; sh.uniforms.uDbg = ANAT_U.uDbg;
      // posterior midline window (spine module selected): a cut-away over the four measured vertebrae, in A0 coordinates
      fs = anatRep(fs, 'void main() {', `uniform float uWin; uniform vec4 uWinB; uniform float uDbg; varying vec3 vRest; varying float vAnatD;
void main() {
  if (uWin > 0.5 && abs(vRest.x) < uWinB.x && vRest.y > uWinB.y && vRest.y < uWinB.z && vRest.z < uWinB.w) discard;`, mode);
      fs = anatRep(fs, '#include <output_fragment>', '#include <output_fragment>\n  if (uDbg > 0.5) gl_FragColor.rgb = mix(gl_FragColor.rgb, vec3(1.0, 0.08, 0.05), step(-0.0015, vAnatD));', mode);
    }
    if (mat.bumpMap) {   // muscle fibres: a subtle sheen up close that vanishes once the stripes near a pixel
      fs = anatRep(fs, '#include <map_fragment>', ANAT_FIB_MAP, mode);
      fs = anatRep(fs, '#include <bumpmap_pars_fragment>', ANAT_FIB_GLSL + ANAT_FIB_BUMP, mode);   // after uv_pars, before map use
    }
    // Open left side: DoubleSide, with the far side's back faces shaded 0.6x so the cut never shows the background
    fs = anatRep(fs, '#include <output_fragment>', '#include <output_fragment>\n  if (!gl_FrontFacing) gl_FragColor.rgb *= 0.6;', mode);
    sh.fragmentShader = fs;
  };
  mat.customProgramCacheKey = () => (depth ? 'anat-depth-' : 'anat-') + mode + (!depth && mat.bumpMap ? '-fib' : '');
  mat.needsUpdate = true;
  return mat;
}

async function anatInflate(s) {
  const u8 = new Uint8Array(b64buf(s));
  if (typeof DecompressionStream === 'function') {
    try { return new Uint8Array(await new Response(new Blob([u8]).stream().pipeThrough(new DecompressionStream('deflate-raw'))).arrayBuffer()); } catch (e) { /* fall back */ }
  }
  if (!window.fflate) await new Promise((res, rej) => { const sc = document.createElement('script'); sc.src = '/vendor/fflate/fflate.umd.js'; sc.onload = res; sc.onerror = () => rej(new Error('fflate')); document.head.appendChild(sc); });
  return window.fflate.inflateSync(u8);
}


  // ======================= anatomy (shared geometry, per-tile palette textures) =======================
  let AA = null, anatState = 'idle';   // idle | loading | ready | failed
  async function loadAnatomy(renderer) {
    if (!renderer.capabilities.isWebGL2 || renderer.capabilities.maxTextureSize < ANAT_TEXW) throw new Error('anatomy: needs WebGL2');
    if (!anatShaderCheck()) throw new Error('anatomy: shader include missing');
    const res = await fetch(DATA + (ANNY.anatomyFile || 'anatomy-anny.v1.json'));
    if (!res.ok) throw new Error(`anatomy ${res.status}`);
    const doc = await res.json();
    if (doc.v !== 2 || doc.palette.nAnny !== ANNY.nb) throw new Error('anatomy: format');
    const strs = anatZFields(doc), out = await Promise.all(strs.map(anatInflate)), zmap = new Map();
    strs.forEach((s, i) => zmap.set(s, out[i]));
    const z = (s) => { const u = zmap.get(s); if (!u) throw new Error('anatomy: field'); return u; };
    const Dc = anatDecode(doc, z), A = anatPrepare(doc, z);
    ANAT_U.uEps.value = doc.eps;
    const geos = Dc.G.map((Gk, k) => {
      const g = new THREE.BufferGeometry(), at = (n, arr, size, norm) => g.setAttribute(n, new THREE.BufferAttribute(arr, size, !!norm));
      at('position', Gk.pos, 3); at('normal', Gk.nor, 3); g.setIndex(new THREE.BufferAttribute(Gk.idx, 1));
      if (k <= 1) at('pidx', Gk.pidx, 1);
      else if (k === 2) { at('pidx2', Gk.pidx2, 2); at('pw', Gk.pw, 1, true); }
      else { at('uv', Gk.uv, 2); at('mwI0', Gk.mwI0, 4); at('mwI1', Gk.mwI1, 4); at('mwW0', Gk.mwW0, 4, true); at('mwW1', Gk.mwW1, 4, true); at('anchor', Gk.anchor, 1); at('mdrv', Gk.mdrv, 1); at('mbeta', Gk.mbeta, 1, true); }
      g.boundingSphere = new THREE.Sphere(V(0, 0.9, 0), 1.3);
      return g;
    });
    AA = { A, geos, rowsP: Math.ceil(2 * ANNY.nv / ANAT_TEXW) };
  }
  const dataTex = (data, h2) => { const t = new THREE.DataTexture(data, ANAT_TEXW, h2, THREE.RGBAFormat, THREE.FloatType); t.magFilter = t.minFilter = THREE.NearestFilter; t.generateMipmaps = false; t.flipY = false; t.needsUpdate = true; return t; };
  function attachAnatomy(tl) {
    if (!AA || tl.anat) return;
    const h = tl.h, names = ['bone', 'bone_measured', 'cartilage', 'muscle', 'muscle_cut'];
    const mats = [MAT.bone, MAT.boneM, MAT.cart, MAT.muscle, MAT.muscle], meshes = {};
    names.forEach((n, k) => { const m = new THREE.Mesh(AA.geos[k], mats[k]); m.frustumCulled = false; m.visible = false; h.root.add(m); meshes[n] = m; });
    const palData = new Float32Array(ANAT_TEXW * 4), skinData = new Float32Array(ANAT_TEXW * AA.rowsP * 4);
    tl.anat = { meshes, palData, skinData, uPal: dataTex(palData, 1), uSkinP: dataTex(skinData, AA.rowsP),
      st: { S: h.S, wq: h.wq, wp: h.wp, rwq: ANNY.rwq, skinM: h.skinM, V: h.P.array, N: h.N.array } };
  }
  function anatPose(tl) {
    const L = tl.anat; if (!L || !tl.anatOn) return;
    anatPaletteP(AA.A, L.st, L.palData, null); L.uPal.needsUpdate = true;
    if (tl.musOn) {
      const P = tl.h.P.array, N = tl.h.N.array, D = L.skinData, nv = ANNY.nv;
      for (let v = 0, i3 = 0, i8 = 0; v < nv; v++, i3 += 3, i8 += 8) { D[i8] = P[i3]; D[i8 + 1] = P[i3 + 1]; D[i8 + 2] = P[i3 + 2]; D[i8 + 4] = N[i3]; D[i8 + 5] = N[i3 + 1]; D[i8 + 6] = N[i3 + 2]; }
      L.uSkinP.needsUpdate = true;
    }
  }

  // ======================= tokens and materials (monochrome on the light paper ground) =======================
  let C = {};
  function readTokens() {
    const cs = getComputedStyle(document.documentElement), g = (n) => cs.getPropertyValue(n).trim();
    C = { bg: g('--bg'), ink: g('--ink'), ink2: g('--ink-2'), ink3: g('--ink-3'), rule: g('--rule'), surface: g('--surface'), alert: g('--alert'), accent: g('--accent') };
  }
  readTokens();
  const MAT = {};
  const TRUE = { muscle: '#8f3a33', bone: '#e6dabb', cart: '#cfd8d6', skin: '#d2a383' };
  function makeMaterials() {
    // anatomy in its true colours (the InSite vest page look): fixed, not palette tokens
    MAT.skin = new THREE.MeshStandardMaterial({ color: new THREE.Color(TRUE.skin), roughness: 0.75, metalness: 0 });
    MAT.glass = new THREE.ShaderMaterial({
      uniforms: { uC: { value: new THREE.Color(C.ink) }, uK: { value: 0.6 } },
      vertexShader: 'varying vec3 vN; varying vec3 vV; void main(){ vec4 mv = modelViewMatrix * vec4(position, 1.0); vV = -mv.xyz; vN = normalMatrix * normal; gl_Position = projectionMatrix * mv; }',
      fragmentShader: 'uniform vec3 uC; uniform float uK; varying vec3 vN; varying vec3 vV; void main(){ float f = 1.0 - abs(dot(normalize(vN), normalize(vV))); gl_FragColor = vec4(uC, clamp(pow(f, 3.0) * uK + 0.025, 0.0, 1.0)); }',
      transparent: true, depthWrite: false,
    });
    MAT.bone = anatPatch(new THREE.MeshStandardMaterial({ color: new THREE.Color(TRUE.bone), roughness: 0.6 }), 'rigid');
    MAT.boneM = anatPatch(new THREE.MeshStandardMaterial({ color: new THREE.Color(C.accent), emissive: new THREE.Color(C.accent), emissiveIntensity: 0.35, roughness: 0.5 }), 'rigid');
    MAT.cart = anatPatch(new THREE.MeshStandardMaterial({ color: new THREE.Color(TRUE.cart), roughness: 0.4 }), 'blend2');
    MAT.muscle = anatPatch(new THREE.MeshStandardMaterial({ color: new THREE.Color(TRUE.muscle), roughness: 0.55, side: THREE.DoubleSide }), 'skin');
    MAT.line = new THREE.LineBasicMaterial({ color: new THREE.Color(C.ink3) });
    MAT.lineFaint = new THREE.LineBasicMaterial({ color: new THREE.Color(C.rule) });
  }

  // ======================= runs =======================
  const PHASE_LABEL = { 2: 'Falling', 3: 'Impact', 4: 'Lying still', 5: 'Hanging in harness', 6: 'Recovering', 7: 'Near fall', 10: 'Free fall', 11: 'Collapsing' };
  function decodeRun(p) {
    const r = Object.assign({}, p);
    r.F = p.frames; r.pel = new Int32Array(b64buf(p.pelvis)); r.jn = new Int16Array(b64buf(p.joints));
    r.sq = new Int16Array(b64buf(p.segq)); r.hqa = new Int16Array(b64buf(p.headq));
    r.T0 = p.t0; r.T1 = p.t0 + (r.F - 1) / p.fps;
    r.events = (p.events || []).slice().sort((a, b) => a.t - b.t);
    return r;
  }
  function nlerp4(out, oo, arr, o0, o1, a, s) {
    const x0 = arr[o0] / s, y0 = arr[o0 + 1] / s, z0 = arr[o0 + 2] / s, w0 = arr[o0 + 3] / s;
    let x1 = arr[o1] / s, y1 = arr[o1 + 1] / s, z1 = arr[o1 + 2] / s, w1 = arr[o1 + 3] / s;
    if (x0 * x1 + y0 * y1 + z0 * z1 + w0 * w1 < 0) { x1 = -x1; y1 = -y1; z1 = -z1; w1 = -w1; }
    const x = x0 + (x1 - x0) * a, y = y0 + (y1 - y0) * a, z = z0 + (z1 - z0) * a, w = w0 + (w1 - w0) * a, n = Math.hypot(x, y, z, w) || 1;
    out[oo] = x / n; out[oo + 1] = y / n; out[oo + 2] = z / n; out[oo + 3] = w / n;
  }
  const _pc = [0, 0, 0];
  function sampleRun(r, t, m) {
    const x = clamp((t - r.T0) * r.fps, 0, r.F - 1), i0 = Math.floor(x), i1 = Math.min(i0 + 1, r.F - 1), a = x - i0;
    for (let c = 0; c < 3; c++) _pc[c] = (r.pel[i0 * 3 + c] + (r.pel[i1 * 3 + c] - r.pel[i0 * 3 + c]) * a) / 1000;
    for (let j = 0; j < 18; j++) for (let c = 0; c < 3; c++) {
      const o0 = i0 * 54 + j * 3 + c, o1 = i1 * 54 + j * 3 + c;
      m.j[3 * j + c] = _pc[c] + (r.jn[o0] + (r.jn[o1] - r.jn[o0]) * a) / 1000;
    }
    for (let k = 0; k < 4; k++) nlerp4(m.q, 4 * k, r.sq, i0 * 16 + 4 * k, i1 * 16 + 4 * k, a, 1e4);
    nlerp4(m.hq, 0, r.hqa, i0 * 4, i1 * 4, a, 1e4);
  }
  function runActivity(r, t) {
    const ph = r.phases.find((p) => t >= p[0] && t < p[1]);
    if (ph && PHASE_LABEL[ph[2]]) return PHASE_LABEL[ph[2]];
    const a = r.activities.find((x) => t >= x[0] && t < x[1]);
    return a ? a[2] : (r.activities.length ? r.activities[r.activities.length - 1][2] : '');
  }
  function runDetector(r, t) {
    let st = { text: 'No alarm', level: '' };
    for (const e of r.events) {
      if (e.t > t) break;
      if (e.kind === 'suspected_fall') st = { text: 'Suspected fall', level: 'watch' };
      else if (e.kind === 'fall_confirmed') st = { text: 'Fall confirmed', level: 'alert' };
      else if (e.kind === 'near_fall') st = { text: 'Near fall logged', level: 'watch' };
      else if (e.kind === 'recovered' || e.kind === 'cleared') st = { text: 'Cleared', level: '' };
    }
    return st;
  }

  // ======================= scene props as monochrome line drawings =======================
  function lineObj(pts, mat) { const g = new THREE.BufferGeometry().setFromPoints(pts); const l = new THREE.LineSegments(g, mat || MAT.line); l.frustumCulled = false; return l; }
  function boxEdges(sx, sy, sz, mat) { const l = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(sx, sy, sz)), mat || MAT.line); l.frustumCulled = false; return l; }
  function buildProps(g, props) {
    const fwdH = (h) => V(-Math.sin(h), 0, -Math.cos(h)), rightH = (h) => V(Math.cos(h), 0, -Math.sin(h));
    for (const p of props || []) {
      if (p.t === 'box') { const m = boxEdges(p.s[0], p.s[1], p.s[2]); m.position.set(p.p[0], p.p[1], p.p[2]); m.rotation.y = p.yaw || 0; g.add(m); }
      else if (p.t === 'ladder') {
        const B = V(...p.base), d = fwdH(p.yaw), rt = rightH(p.yaw), Hh = p.top_y - B.y, T = B.clone().addScaledVector(d, Hh / 4).add(V(0, Hh, 0)), pts = [];
        for (const sd of [-1, 1]) pts.push(B.clone().addScaledVector(rt, sd * p.w / 2), T.clone().addScaledVector(rt, sd * p.w / 2));
        for (let hh = 0.3; hh < Hh - 0.05; hh += 0.3) { const c = B.clone().lerp(T, hh / Hh); pts.push(c.clone().addScaledVector(rt, -p.w / 2), c.clone().addScaledVector(rt, p.w / 2)); }
        g.add(lineObj(pts));
      } else if (p.t === 'rebar') {
        const Cc = V(...p.p), d = fwdH(p.yaw), rt = rightH(p.yaw), sp = p.sp || 0.2, pts = [];
        const nA = Math.floor(p.w / sp) + 1, nB = Math.floor(p.l / sp) + 1;
        for (let i = 0; i < nA; i++) { const o = Cc.clone().addScaledVector(rt, -p.w / 2 + i * sp); o.y = Cc.y + 0.05; pts.push(o.clone().addScaledVector(d, -p.l / 2), o.clone().addScaledVector(d, p.l / 2)); }
        for (let i = 0; i < nB; i++) { const o = Cc.clone().addScaledVector(d, -p.l / 2 + i * sp); o.y = Cc.y + 0.066; pts.push(o.clone().addScaledVector(rt, -p.w / 2), o.clone().addScaledVector(rt, p.w / 2)); }
        g.add(lineObj(pts));
      } else if (p.t === 'cyl') { const m = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.CylinderGeometry(p.rt, p.r, p.h, 16)), MAT.line); m.position.set(p.p[0], p.p[1] + p.h / 2, p.p[2]); g.add(m); }
      else if (p.t === 'tube') g.add(lineObj([V(...p.a), V(...p.b)]));
    }
  }

  // ======================= tiles =======================
  const MODES = ['skin', 'muscle', 'bone', 'both'];
  const MODE_LABEL = { skin: 'Skin', muscle: 'Muscle', bone: 'Skeleton', both: 'Both' };
  const TILES = [
    { key: 'H-LAD', kind: 'run', name: 'Ladder fall', mode: 'both', az: 80, hold: 36.2,
      outcome: 'Fall confirmed about 10 s after impact, supervisor alerted. Expected: fall.' },
    { key: 'NF-STUMBLE', kind: 'run', name: 'Stumble', mode: 'muscle', az: -55, hold: 15.7,
      outcome: 'Near fall logged, no supervisor alert. Expected: near fall.' },
    { key: 'collapse', kind: 'clip', name: 'Faint', mode: 'skin', az: -40, hold: 3.94,
      outcome: 'SYNC seed 33. Simulated faints: falldet P(fall) 1.00 (REPORT 14.3).' },
    { key: 'A-REBAR', kind: 'run', name: 'Tying rebar', mode: 'bone', az: -60, hold: 20,
      outcome: 'No alarm in 101 s of bent work. Expected: no alarm.' },
    { key: 'A-CARRY', kind: 'run', name: 'Carrying loads', mode: 'muscle', az: -50, hold: 15,
      outcome: 'No alarm in 101 s of carrying. Expected: no alarm.' },
    { key: 'lift', kind: 'clip', name: 'Lifting 25.5 kg', mode: 'both', az: 70, hold: 19.3,
      outcome: 'Trunk from ThunderSim A-CARRY seed 0. Motion clip, not scored.' },
  ];
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  const $ = (id) => document.getElementById(id);
  const grid = $('sim-grid'), canvas = $('sim-canvas'), playBtn = $('sim-play'), noteEl = $('sim-load');
  if (!grid || !canvas) return;

  // DOM first, so the page reads correctly before (or without) WebGL
  for (const [i, cfg] of TILES.entries()) {
    const fig = document.createElement('figure'); fig.className = 'tile'; fig.dataset.i = i;
    fig.innerHTML = `<div class="scene" data-mode="${cfg.mode}"><div class="tile-st"><h3>${cfg.name}</h3><span class="act">Simulated</span><span class="det"></span></div>` +
      `<button type="button" class="mode" aria-label="Render mode ${MODE_LABEL[cfg.mode]}, select to change">${MODE_LABEL[cfg.mode]}</button><span class="tile-note" hidden></span></div>` +
      `<figcaption class="sr">${cfg.outcome}</figcaption>`;
    grid.appendChild(fig);
    cfg.el = fig; cfg.scene = fig.querySelector('.scene'); cfg.act = fig.querySelector('.act'); cfg.det = fig.querySelector('.det');
    cfg.btn = fig.querySelector('.mode'); cfg.note = fig.querySelector('.tile-note');
  }

  let renderer = null, scene = null, gridHelper = null, hemi = null, key = null, rim = null, MA = null;
  let tiles = [], ready = false, playing = !reduce.matches, active = false, raf = 0, last = 0, lastUi = 0, rr = 0, focus = 0, loading = null;
  const perf = { n: 0, frame: 0, pose: 0, render: 0, posed: 0, avg: null };
  window.__simPerf = () => perf.avg;

  function setNote(t) { if (noteEl) { noteEl.textContent = t || ''; noteEl.hidden = !t; } }
  function effMode(tl) {
    if (tl.mode === 'skin') return 'skin';
    if (anatState === 'ready' && tl.anat) return tl.mode;
    return anatState === 'failed' ? 'skin' : 'loading';
  }
  function applyMode(tl) {
    const h = tl.h, m = effMode(tl);
    h.body.visible = m === 'skin'; h.glass.visible = m === 'both' || m === 'loading';
    tl.anatOn = m === 'muscle' || m === 'bone' || m === 'both'; tl.musOn = m === 'muscle' || m === 'both';
    if (tl.anat) {
      const M = tl.anat.meshes, bone = m === 'bone' || m === 'both';
      M.muscle.visible = tl.musOn; M.muscle_cut.visible = m === 'muscle';
      M.bone.visible = M.bone_measured.visible = M.cartilage.visible = bone;
    }
    const cfg = tl.cfg; cfg.scene.dataset.mode = tl.mode;
    cfg.btn.textContent = MODE_LABEL[tl.mode]; cfg.btn.setAttribute('aria-label', `Render mode ${MODE_LABEL[tl.mode]}, select to change`);
    cfg.note.hidden = m !== 'loading' && !(anatState === 'failed' && tl.mode !== 'skin');
    cfg.note.textContent = m === 'loading' ? 'Loading anatomy' : 'Anatomy unavailable here, showing skin';
    tl.dirty = true;
  }

  function makeTile(cfg, R, clips) {
    const tl = { cfg, mode: cfg.mode, kind: cfg.kind, group: new THREE.Group(), cam: new THREE.PerspectiveCamera(30, 1.5, 0.05, 60),
      tgt: V(), head: 0, camInit: false, dirty: true, vis: false, rect: null, wait: 0 };
    tl.h = makeBody(MAT.skin, MAT.glass);
    tl.group.add(tl.h.root);
    if (cfg.kind === 'run') {
      const r = R.find((x) => x.id === cfg.key);
      tl.r = r; tl.k = r.body.H / ANNY.height; tl.ground = (r.scene && r.scene.ground_y) || 0;
      tl.m = { q: new Float64Array(16), hq: new Float64Array(4), j: new Float64Array(54) }; tl.f = makeAnnySample();
      tl.T0 = r.T0; tl.T1 = r.T1; buildProps(tl.group, r.scene && r.scene.props);
    } else {
      const c = clips[cfg.key]; tl.c = c; tl.k = 1; tl.ground = 0; tl.T0 = 0; tl.T1 = c.T;
      if (c.env && c.env.crate) {
        const s = c.env.crate.size; tl.crate = boxEdges(s[0], s[1], s[2], MAT.line); tl.crate.geometry.translate(0, s[1] / 2, 0); tl.group.add(tl.crate);
      }
      for (const p of (c.env && c.env.pallets) || []) { const m = boxEdges(1.2, 0.144, 0.8); m.position.set(p.pos[0], 0.072, p.pos[2]); m.rotation.y = p.yaw || 0; tl.group.add(m); }
    }
    const hold = cfg.kind === 'clip' ? cfg.hold - (tl.c.t0 || 0) : cfg.hold;
    tl.t = playing ? tl.T0 : clamp(hold, tl.T0, tl.T1);
    scene.add(tl.group); tl.group.visible = false;
    return tl;
  }
  // world position of a bone after posing
  function boneWorld(tl, b, out) { const h = tl.h, k = tl.k, y0 = h.root.position.y; return out.set(h.wp[3 * b] * k, h.wp[3 * b + 1] * k + y0, h.wp[3 * b + 2] * k); }
  const _bw = V(), _fw = V(), _q = new THREE.Quaternion();
  function poseTile(tl) {
    const t0 = performance.now(), h = tl.h;
    if (tl.kind === 'run') {
      sampleRun(tl.r, tl.t, tl.m);
      monAnnySolve(MA, tl.f, tl.m.q, tl.m.hq, tl.m.j, tl.r.body, tl.k, tl.r.scene, null);
      h.root.scale.setScalar(tl.k);
      poseAnny(h, tl.f);
      h.root.position.y = monAnnyPen(h, MA, tl.k, tl.ground);
    } else {
      const f = sampleAnny(tl.c, tl.t, h.smp);
      poseAnny(h, f);
      h.root.position.y = 0;
      if (tl.crate) { tl.crate.position.set(f.box[0], f.box[1], f.box[2]); tl.crate.quaternion.set(f.box[3], f.box[4], f.box[5], f.box[6]); }
    }
    anatPose(tl);
    // camera target: the lower trunk; heading from the pelvis segment while upright, held when lying
    boneWorld(tl, ANNY.bi.spine05, _bw);
    const D = h.D; _q.set(D[0], D[1], D[2], D[3]);
    _fw.set(0, 0, -1).applyQuaternion(_q); const upY = V(0, 1, 0).applyQuaternion(_q).y;
    if (upY > 0.5 && Math.hypot(_fw.x, _fw.z) > 0.3) tl.headGoal = Math.atan2(_fw.x, _fw.z);
    else if (tl.headGoal == null) tl.headGoal = 0;
    tl.goal = tl.goal || V(); tl.goal.copy(_bw);
    if (!tl.camInit) { tl.tgt.copy(tl.goal); tl.head = tl.headGoal; tl.camInit = true; }
    tl.dirty = false;
    return performance.now() - t0;
  }
  function updateCam(tl, dt) {
    const a = 1 - Math.exp(-dt * 3), ah = 1 - Math.exp(-dt * 1.2);
    tl.tgt.lerp(tl.goal, a);
    let d = tl.headGoal - tl.head; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; tl.head += d * ah;
    const r = tl.rect, aspect = r.w / Math.max(1, r.h), tn = Math.tan(15 * Math.PI / 180);
    const dist = Math.max(1.22 / tn, 1.05 / (tn * aspect)), el = 10 * Math.PI / 180;
    const th = tl.head + tl.cfg.az * Math.PI / 180;   // 0 = in front of the worker, +90 = wearer-left
    const ty = Math.max(0.78, tl.tgt.y);
    tl.cam.aspect = aspect; tl.cam.position.set(tl.tgt.x + Math.sin(th) * Math.cos(el) * dist, ty + Math.sin(el) * dist, tl.tgt.z + Math.cos(th) * Math.cos(el) * dist);
    tl.cam.lookAt(tl.tgt.x, ty, tl.tgt.z); tl.cam.updateProjectionMatrix();
    tl.dist = dist; tl.ty = ty;
  }
  function advance(tl, dt) {
    if (!playing) return;
    if (tl.wait > 0) { tl.wait -= dt; if (tl.wait <= 0) { tl.t = tl.T0; tl.camInit = false; } else return; }
    tl.t += dt; tl.dirty = true;
    if (tl.t >= tl.T1) { tl.t = tl.T1; tl.wait = 2; }
  }
  function updateUi(tl) {
    const cfg = tl.cfg;
    if (tl.kind === 'run') {
      cfg.act.textContent = `${runActivity(tl.r, tl.t)}, ${tl.t.toFixed(1)} s`;
      const st = runDetector(tl.r, tl.t); cfg.det.textContent = st.text; cfg.det.dataset.level = st.level;
    } else {
      const tt = tl.t + (tl.c.t0 || 0), cap = tl.c.captions.find((x) => tt >= x.t0 && tt < x.t1) || tl.c.captions[tl.c.captions.length - 1];
      cfg.act.textContent = cap ? cap.text.split(/[:,(]/)[0].trim() : '';
      cfg.det.textContent = tl.cfg.key === 'collapse' ? 'Simulated faint' : 'Simulated lift';
      cfg.det.dataset.level = '';
    }
  }

  function measure() {
    const g = grid.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5), W = Math.max(1, Math.round(g.width)), H = Math.max(1, Math.round(g.height));
    if (renderer.getPixelRatio() !== dpr) renderer.setPixelRatio(dpr);
    const sz = renderer.getSize(new THREE.Vector2());
    if (sz.x !== W || sz.y !== H) renderer.setSize(W, H, true);
    for (const tl of tiles) {
      const r = tl.cfg.scene.getBoundingClientRect();
      tl.rect = { x: r.left - g.left, y: H - (r.bottom - g.top), w: r.width, h: r.height };
      tl.vis = r.bottom > 0 && r.top < innerHeight && r.width > 2 && r.height > 2;
    }
    return { W, H };
  }
  const _surf = new THREE.Color();
  function frame(now) {
    raf = requestAnimationFrame(frame);
    const dt = Math.min(0.1, (now - (last || now)) / 1000); last = now;
    const f0 = performance.now();
    measure();
    for (const tl of tiles) advance(tl, dt);
    // CPU skinning budget: the focused tile every frame, plus two other visible tiles in turn
    let poseMs = 0, posed = 0;
    const vis = tiles.filter((t) => t.vis);
    const want = new Set(); const ft = tiles[focus];
    if (ft && ft.vis && ft.dirty) want.add(ft);
    for (let n = 0; n < vis.length && want.size < 3; n++) { const t = vis[(rr + n) % vis.length]; if (t.dirty || !t.camInit) want.add(t); }
    rr = (rr + 2) % Math.max(1, vis.length);
    for (const tl of want) { poseMs += poseTile(tl); posed++; }
    const r0 = performance.now();
    renderer.setScissorTest(false); renderer.setClearColor(0x000000, 0); renderer.clear();
    renderer.setScissorTest(true);
    for (const tl of tiles) tl.group.visible = false;
    for (const tl of vis) {
      if (!tl.camInit) continue;
      updateCam(tl, dt);
      const r = tl.rect;
      renderer.setViewport(r.x, r.y, r.w, r.h); renderer.setScissor(r.x, r.y, r.w, r.h);
      renderer.setClearColor(_surf, 1); renderer.clear();
      tl.group.visible = true;
      if (tl.anat) { ANAT_U.uPal.value = tl.anat.uPal; ANAT_U.uSkinP.value = tl.anat.uSkinP; }
      gridHelper.position.set(Math.round(tl.tgt.x / 0.5) * 0.5, tl.ground + 0.001, Math.round(tl.tgt.z / 0.5) * 0.5);
      scene.fog.near = tl.dist + 0.8; scene.fog.far = tl.dist + 5.5;
      key.position.set(tl.cam.position.x - (tl.tgt.z - tl.cam.position.z) * 0.6, tl.cam.position.y + 3, tl.cam.position.z + (tl.tgt.x - tl.cam.position.x) * 0.6);
      key.target.position.copy(tl.tgt);
      rim.position.set(2 * tl.tgt.x - tl.cam.position.x, tl.tgt.y + 2.5, 2 * tl.tgt.z - tl.cam.position.z); rim.target.position.copy(tl.tgt);
      key.target.updateMatrixWorld(); rim.target.updateMatrixWorld();
      renderer.render(scene, tl.cam);
      tl.group.visible = false;
    }
    const f1 = performance.now();
    if (now - lastUi > 200) { lastUi = now; for (const tl of vis) updateUi(tl); }
    perf.n++; perf.frame += f1 - f0; perf.pose += poseMs; perf.render += f1 - r0; perf.posed += posed;
    if (perf.n >= 20) {
      perf.avg = { frames: perf.n, frameCpuMs: +(perf.frame / perf.n).toFixed(2), poseMsPerFrame: +(perf.pose / perf.n).toFixed(2), renderCpuMs: +(perf.render / perf.n).toFixed(2),
        tilesPosedPerFrame: +(perf.posed / perf.n).toFixed(2), visibleTiles: vis.length, anatomy: anatState };
      Object.assign(perf, { n: 0, frame: 0, pose: 0, render: 0, posed: 0 });
    }
  }
  function setActive(on) {
    on = on && ready && !document.hidden;
    if (on === active) return;
    active = on;
    if (on) { last = 0; raf = requestAnimationFrame(frame); } else cancelAnimationFrame(raf);
  }
  function inView() { const r = SEC.getBoundingClientRect(); return r.top < innerHeight && r.bottom > 0; }

  async function getJSON(f) { const res = await fetch(DATA + f); if (!res.ok) throw new Error(`${f} ${res.status}`); return res.json(); }
  function ensureLoaded() {
    if (loading) return loading;
    setNote('Loading the body and six simulations (about 3 MB)');
    loading = (async () => {
      const [annyDoc, runsDoc, liftDoc, colDoc] = await Promise.all([getJSON('anny-avg-male.v1.json'), getJSON('sim-runs.json'), getJSON('lift-acarry-s0.v1.json'), getJSON('collapse-anny.v1.json')]);
      ANNY = prepareAnny(annyDoc);
      const R = runsDoc.runs.map(decodeRun), clips = { lift: prepareAnnyClip('lift', liftDoc), collapse: prepareAnnyClip('collapse', colDoc) };
      renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
      renderer.autoClear = false;
      _surf.set(C.surface);
      scene = new THREE.Scene();
      scene.fog = new THREE.Fog(new THREE.Color(C.surface), 6, 12);
      hemi = new THREE.HemisphereLight(0xffffff, new THREE.Color(C.rule), 0.75); // light ground: lights stay white, ink is for materials
      key = new THREE.DirectionalLight(0xffffff, 0.9);
      rim = new THREE.DirectionalLight(0xffffff, 0.45);
      scene.add(hemi, key, key.target, rim, rim.target);
      gridHelper = new THREE.GridHelper(12, 24, new THREE.Color(C.rule), new THREE.Color(C.rule));
      scene.add(gridHelper);
      makeMaterials();
      const h0 = makeBody(MAT.skin, MAT.glass);
      MA = monAnnyRig(h0);
      tiles = TILES.map((cfg) => makeTile(cfg, R, clips));
      tiles.forEach((tl) => { tl.cfg.scene.addEventListener('click', () => { focus = tiles.indexOf(tl); tl.mode = MODES[(MODES.indexOf(tl.mode) + 1) % MODES.length]; applyMode(tl); });
        tl.cfg.scene.addEventListener('pointerenter', () => { focus = tiles.indexOf(tl); }); applyMode(tl); });
      ready = true;
      setNote('Loading anatomy (about 3 MB)');
      setActive(inView());
      anatState = 'loading';
      try { await loadAnatomy(renderer); anatState = 'ready'; tiles.forEach((tl) => { attachAnatomy(tl); applyMode(tl); }); setNote(''); }
      catch (err) { console.warn(err); anatState = 'failed'; tiles.forEach(applyMode); setNote('The anatomy layer did not load on this device, so every scene shows the skin.'); }
    })().catch((err) => { console.error(err); setNote('The 3D scenes could not load here. The results above still stand.'); });
    return loading;
  }
  if (playBtn) {
    playBtn.textContent = playing ? 'Pause' : 'Play';
    playBtn.addEventListener('click', () => { playing = !playing; playBtn.textContent = playing ? 'Pause' : 'Play'; playBtn.setAttribute('aria-pressed', String(!playing)); });
  }
  // Snapped desktop deck: render only while this is the current page, so slide 5 never competes with the
  // live spine on slide 4 (a partly visible slide 5 used to switch the six scenes on). Stacked phone layout:
  // render while on screen. Preloading (about 6 MB to decode) never runs while the Live page is showing:
  // it happens in idle time on the earlier slides, or on arrival.
  const snapped = matchMedia('(min-width:900px) and (min-height:620px)');
  let current = null;
  const mine = () => (snapped.matches ? current === 'simulation' : inView());
  new IntersectionObserver((es) => { if (!snapped.matches) es.forEach((en) => { if (en.isIntersecting) ensureLoaded(); setActive(en.isIntersecting); }); }, { threshold: 0.02 }).observe(SEC);
  document.addEventListener('deck:page', (e) => {
    if (!e.detail) return;
    current = e.detail.id;
    if (current === 'simulation') { ensureLoaded(); setActive(true); } else if (snapped.matches) setActive(false);
  });
  document.addEventListener('visibilitychange', () => setActive(!document.hidden && mine()));
  const idle = window.requestIdleCallback || ((f) => setTimeout(f, 1500));
  idle(() => { if (current !== 'live') ensureLoaded(); }, { timeout: 4000 });
  window.__sim = { seek: (i, t) => { const tl = tiles[i]; if (tl) { tl.t = t; tl.dirty = true; tl.camInit = false; } }, play: (on) => { playing = on; }, mode: (i, m) => { const tl = tiles[i]; if (tl) { tl.mode = m; applyMode(tl); } }, ready: () => ready && anatState };
})();
