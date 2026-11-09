// File: body.js
// Purpose: The simulated worker (torso swept along the simulated spine, IK limbs, head with
//          hard hat, hi-vis vest, tools) and the 4-IMU strip (pods, strap, force and gyro
//          vectors, x-ray spine view).
import * as THREE from "three";
import { SLOT_COLORS, lerpFrame, quatFrame, placeBetween } from "./util.js";

const RINGS = 30;
const RADIAL = 24;
// Torso cross-section along the trunk: s, half-width (lateral), half-depth (m, for H = 1.75 m)
const PROFILE = [
  [0.0, 0.135, 0.105], [0.07, 0.172, 0.13], [0.2, 0.165, 0.118], [0.32, 0.148, 0.104], [0.45, 0.158, 0.112],
  [0.62, 0.173, 0.124], [0.8, 0.184, 0.12], [0.88, 0.196, 0.112], [0.94, 0.16, 0.098], [0.975, 0.11, 0.075], [1.0, 0.065, 0.06],
];
const COLORS = {
  pants: new THREE.Color("#2b3a55"), belt: new THREE.Color("#3b2a1a"), vest: new THREE.Color("#c6ef2f"),
  stripe: new THREE.Color("#e3e7ea"), shirt: new THREE.Color("#3a4250"), skin: new THREE.Color("#d29f78"),
  glove: new THREE.Color("#c39a5e"), boot: new THREE.Color("#5b3a1f"), hat: new THREE.Color("#ffd23f"),
};

function profileAt(s) {
  for (let i = 1; i < PROFILE.length; i++) {
    if (s <= PROFILE[i][0]) {
      const [s0, a0, b0] = PROFILE[i - 1], [s1, a1, b1] = PROFILE[i];
      const u = (s - s0) / (s1 - s0);
      return [a0 + (a1 - a0) * u, b0 + (b1 - b0) * u];
    }
  }
  return [PROFILE.at(-1)[1], PROFILE.at(-1)[2]];
}

function colorAt(s, angle) {
  if (s < 0.17) return COLORS.pants;
  if (s < 0.21) return COLORS.belt;
  if (s > 0.945) return COLORS.shirt;
  if ((s > 0.36 && s < 0.4) || (s > 0.53 && s < 0.57)) return COLORS.stripe;
  // vertical reflective straps over the chest and back
  const c = Math.cos(angle);
  if (s > 0.57 && Math.abs(Math.abs(Math.sin(angle)) - 0.55) < 0.07 && Math.abs(c) < 0.9) return COLORS.stripe;
  return COLORS.vest;
}

// Cubic Hermite through points p[i] with unit tangents d[i], chord-length parameter (as spine.spine_curve).
function hermite(p, d, samples, out) {
  const chord = [0];
  for (let i = 1; i < p.length; i++) chord.push(chord[i - 1] + p[i].distanceTo(p[i - 1]));
  for (let k = 0; k < samples; k++) {
    const u = (k / (samples - 1)) * chord.at(-1);
    let i = 0;
    while (i < p.length - 2 && u > chord[i + 1]) i++;
    const h = chord[i + 1] - chord[i];
    const t = h > 1e-9 ? (u - chord[i]) / h : 0;
    const t2 = t * t, t3 = t2 * t;
    const h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
    out[k].set(0, 0, 0).addScaledVector(p[i], h00).addScaledVector(d[i], h10 * h).addScaledVector(p[i + 1], h01).addScaledVector(d[i + 1], h11 * h);
  }
  return out;
}

export class Body {
  constructor(scene) {
    this.scene = scene;
    this.group = new THREE.Group();
    scene.add(this.group);
    this.xray = false;
    this.showForces = true;
    this.showGyro = false;
  }

  dispose() {
    this.scene.remove(this.group);
    this.group.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
    this.group = new THREE.Group();
    this.scene.add(this.group);
  }

  build(payload) {
    this.dispose();
    this.payload = payload;
    const b = payload.meta.body;
    const s = b.s, w = b.w;
    this.dims = b;
    this.materials = [];
    const m = (color, opts = {}) => {
      const mm = new THREE.MeshStandardMaterial({ color, roughness: 0.75, ...opts });
      this.materials.push(mm);
      return mm;
    };
    // torso
    const geo = new THREE.BufferGeometry();
    const n = RINGS * RADIAL;
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    const colors = new Float32Array(n * 3);
    for (let r = 0; r < RINGS; r++) {
      const sr = r / (RINGS - 1);
      for (let k = 0; k < RADIAL; k++) {
        const c = colorAt(sr, (k / RADIAL) * Math.PI * 2);
        colors.set([c.r, c.g, c.b], (r * RADIAL + k) * 3);
      }
    }
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const idx = [];
    for (let r = 0; r < RINGS - 1; r++) for (let k = 0; k < RADIAL; k++) {
      const a = r * RADIAL + k, bb = r * RADIAL + ((k + 1) % RADIAL), c = a + RADIAL, d = bb + RADIAL;
      idx.push(a, c, bb, bb, c, d);
    }
    // caps
    const capBottom = [], capTop = [];
    for (let k = 1; k < RADIAL - 1; k++) { capBottom.push(0, k + 1, k); capTop.push((RINGS - 1) * RADIAL, (RINGS - 1) * RADIAL + k, (RINGS - 1) * RADIAL + k + 1); }
    geo.setIndex([...idx, ...capBottom, ...capTop]);
    this.torsoMat = m(0xffffff, { vertexColors: true, roughness: 0.8 });
    this.torso = new THREE.Mesh(geo, this.torsoMat);
    this.torso.castShadow = true;
    this.torso.frustumCulled = false;
    this.group.add(this.torso);
    this.ringCenters = Array.from({ length: RINGS }, () => new THREE.Vector3());

    // limbs: tapered cylinders with exact bone lengths, plus joint spheres
    const pants = m(COLORS.pants), shirt = m(COLORS.shirt), skin = m(COLORS.skin, { roughness: 0.6 });
    const glove = m(COLORS.glove), boot = m(COLORS.boot, { roughness: 0.6 }), sole = m(0x1a1a1a);
    const limb = (len, r0, r1, material) => {
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r0, r1, len, 14), material);
      mesh.castShadow = true;
      this.group.add(mesh);
      return mesh;
    };
    const ball = (r, material) => {
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), material);
      mesh.castShadow = true;
      this.group.add(mesh);
      return mesh;
    };
    this.limbs = [];
    for (let side = 0; side < 2; side++) {
      this.limbs.push({ a: side, b: 2 + side, mesh: limb(b.thigh, 0.084 * w * s, 0.058 * s, pants), len: b.thigh });
      this.limbs.push({ a: 2 + side, b: 4 + side, mesh: limb(b.shank, 0.057 * s, 0.043 * s, pants), len: b.shank });
      this.limbs.push({ a: 10 + side, b: 12 + side, mesh: limb(b.upper_arm, 0.054 * w * s, 0.042 * s, shirt), len: b.upper_arm });
      this.limbs.push({ a: 12 + side, b: 14 + side, mesh: limb(b.forearm * 0.86, 0.042 * s, 0.032 * s, shirt), len: b.forearm * 0.86, trim: true });
    }
    this.joints = [];
    for (let side = 0; side < 2; side++) {
      this.joints.push({ j: side, mesh: ball(0.084 * w * s, pants) });
      this.joints.push({ j: 2 + side, mesh: ball(0.058 * s, pants) });
      this.joints.push({ j: 10 + side, mesh: ball(0.058 * w * s, this.vestShoulder || m(COLORS.vest)) });
      this.joints.push({ j: 12 + side, mesh: ball(0.043 * s, shirt) });
      this.joints.push({ j: 14 + side, mesh: ball(0.047 * s, glove) });
    }
    this.boots = [];
    for (let side = 0; side < 2; side++) {
      const g = new THREE.Group();
      const upper = new THREE.Mesh(new THREE.BoxGeometry(0.105 * s, 0.1 * s, 1), boot);
      upper.position.y = 0.05 * s; upper.castShadow = true; g.add(upper);
      const so = new THREE.Mesh(new THREE.BoxGeometry(0.112 * s, 0.025 * s, 1.02), sole);
      so.position.y = 0.012 * s; g.add(so);
      const cuff = new THREE.Mesh(new THREE.CylinderGeometry(0.05 * s, 0.052 * s, 0.1 * s, 12), boot);
      cuff.castShadow = true;
      this.group.add(g, cuff);
      this.boots.push({ group: g, upper, sole: so, cuff });
    }
    // neck and head
    this.neck = limb(1, 0.052 * s, 0.056 * s, skin);
    this.head = new THREE.Group();
    const skull = new THREE.Mesh(new THREE.SphereGeometry(0.1 * s, 24, 18), skin);
    skull.scale.set(0.8, 1.0, 0.92); skull.castShadow = true;
    this.head.add(skull);
    const glasses = new THREE.Mesh(new THREE.BoxGeometry(0.15 * s, 0.035 * s, 0.03 * s), m(0x161a1f, { roughness: 0.2, metalness: 0.4 }));
    glasses.position.set(0, 0.015 * s, -0.085 * s); this.head.add(glasses);
    const nose = new THREE.Mesh(new THREE.BoxGeometry(0.022 * s, 0.04 * s, 0.03 * s), skin);
    nose.position.set(0, -0.012 * s, -0.093 * s); this.head.add(nose);
    this.group.add(this.head);
    this.hat = new THREE.Group();
    const hatMat = m(COLORS.hat, { roughness: 0.45 });
    const dome = new THREE.Mesh(new THREE.SphereGeometry(0.108 * s, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2), hatMat);
    dome.scale.set(0.92, 0.9, 1.05); dome.castShadow = true; this.hat.add(dome);
    const brim = new THREE.Mesh(new THREE.CylinderGeometry(0.117 * s, 0.12 * s, 0.012 * s, 28), hatMat);
    brim.scale.set(0.95, 1, 1.12); brim.castShadow = true; this.hat.add(brim);
    const peak = new THREE.Mesh(new THREE.BoxGeometry(0.16 * s, 0.01 * s, 0.07 * s), hatMat);
    peak.position.set(0, 0.002, -0.12 * s); this.hat.add(peak);
    const ridge = new THREE.Mesh(new THREE.BoxGeometry(0.025 * s, 0.03 * s, 0.22 * s), hatMat);
    ridge.position.y = 0.092 * s; this.hat.add(ridge);
    this.hatOffset = new THREE.Vector3(0, 0.045 * s, 0.005);
    this.head.add(this.hat);
    this.hatCable = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineBasicMaterial({ color: 0x111111 }));
    this.hatCable.visible = false;
    this.group.add(this.hatCable);
    this.hat.position.copy(this.hatOffset);

    this.buildTools(m);
    this.buildStrip();
    this.buildXray();
    this.buildTrail(payload);
    this.setXray(this.xray);
  }

  buildTools(m) {
    const s = this.dims.s;
    const wood = m(0xb5835a), steel = m(0x8b939c, { metalness: 0.7, roughness: 0.35 }), red = m(0xd83a2a), dark = m(0x222222);
    this.hammer = new THREE.Group();
    const handle = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.016, 0.33, 8), wood);
    handle.position.y = 0.12; this.hammer.add(handle);
    const headM = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.035, 0.13), steel);
    headM.position.y = 0.28; this.hammer.add(headM);
    this.drill = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.07, 0.22), red); body.position.set(0, 0.07, 0); this.drill.add(body);
    const grip = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.14, 0.05), dark); grip.position.set(0, 0, 0.06); this.drill.add(grip);
    const bit = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, 0.16, 6), steel); bit.rotation.x = Math.PI / 2; bit.position.set(0, 0.07, -0.18); this.drill.add(bit);
    this.jack = new THREE.Group();
    const jb = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, 1, 14), red); jb.castShadow = true; this.jack.add(jb);
    const chisel = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.008, 0.25, 8), steel); this.jack.add(chisel);
    const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.5, 8), dark); bar.rotation.z = Math.PI / 2; this.jack.add(bar);
    this.jackParts = { jb, chisel, bar };
    for (const g of [this.hammer, this.drill, this.jack]) { g.visible = false; g.traverse((o) => { if (o.isMesh) o.castShadow = true; }); this.group.add(g); }
    this.dust = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ color: 0xd8d2c4, size: 0.03, transparent: true, opacity: 0.8 }));
    this.dust.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(60 * 3), 3));
    this.dust.visible = false;
    this.group.add(this.dust);
  }

  buildStrip() {
    const sz = [0.05, 0.034, 0.016];
    this.pods = [];
    for (let i = 0; i < 4; i++) {
      const g = new THREE.Group();
      const shell = new THREE.Mesh(new THREE.BoxGeometry(...sz), new THREE.MeshStandardMaterial({ color: SLOT_COLORS[i], roughness: 0.35, metalness: 0.1, emissive: new THREE.Color(SLOT_COLORS[i]), emissiveIntensity: 0.25 }));
      shell.castShadow = true;
      g.add(shell);
      const led = new THREE.Mesh(new THREE.SphereGeometry(0.004, 8, 6), new THREE.MeshBasicMaterial({ color: 0x9dff9d }));
      led.position.set(0.017, 0.01, 0.009);
      g.add(led);
      const axes = new THREE.AxesHelper(0.06);
      axes.visible = false;
      g.add(axes);
      this.group.add(g);
      const force = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 0.1, new THREE.Color(SLOT_COLORS[i]), 0.035, 0.022);
      force.line.material.linewidth = 2;
      force.renderOrder = 10;
      force.traverse((o) => { if (o.material) { o.material.depthTest = false; o.material.transparent = true; o.material.opacity = 0.95; } });
      this.group.add(force);
      const gyro = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 0.1, new THREE.Color(SLOT_COLORS[i]).lerp(new THREE.Color(0xffffff), 0.6), 0.025, 0.016);
      gyro.renderOrder = 10;
      gyro.traverse((o) => { if (o.material) { o.material.depthTest = false; o.material.transparent = true; o.material.opacity = 0.8; } });
      this.group.add(gyro);
      this.pods.push({ group: g, shell, led, axes, force, gyro });
    }
    const samples = 40;
    this.strapPts = Array.from({ length: samples }, () => new THREE.Vector3());
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(samples * 2 * 3), 3));
    const idx = [];
    for (let k = 0; k < samples - 1; k++) idx.push(2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 1, 2 * k + 3, 2 * k + 2);
    geo.setIndex(idx);
    this.strap = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x262a30, roughness: 0.6, side: THREE.DoubleSide }));
    this.strap.frustumCulled = false;
    this.group.add(this.strap);
    this.hub = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.04, 0.025), new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.5 }));
    this.group.add(this.hub);
  }

  buildXray() {
    this.xrayGroup = new THREE.Group();
    this.segLines = [];
    for (let i = 0; i < 3; i++) {
      const c = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1, 8), new THREE.MeshBasicMaterial({ color: SLOT_COLORS[i], depthTest: false }));
      c.renderOrder = 9;
      this.xrayGroup.add(c);
      this.segLines.push(c);
    }
    this.vertebrae = [];
    for (let i = 0; i < 4; i++) {
      const v = new THREE.Mesh(new THREE.SphereGeometry(0.022, 12, 8), new THREE.MeshBasicMaterial({ color: SLOT_COLORS[i], depthTest: false }));
      v.renderOrder = 9;
      const ax = new THREE.AxesHelper(0.09);
      ax.renderOrder = 9;
      ax.material.depthTest = false;
      v.add(ax);
      this.xrayGroup.add(v);
      this.vertebrae.push(v);
    }
    const curve = new THREE.BufferGeometry();
    curve.setAttribute("position", new THREE.BufferAttribute(new Float32Array(40 * 3), 3));
    this.curveLine = new THREE.Line(curve, new THREE.LineBasicMaterial({ color: 0xffffff, depthTest: false }));
    this.curveLine.renderOrder = 9;
    this.curveLine.frustumCulled = false;
    this.xrayGroup.add(this.curveLine);
    this.group.add(this.xrayGroup);
  }

  buildTrail(payload) {
    const sp = payload.frames.spine;
    const n = sp.length / 12;
    const pts = [];
    for (let i = 0; i < n; i += 2) pts.push(new THREE.Vector3(sp[i * 12], sp[i * 12 + 1], sp[i * 12 + 2]));
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    this.trail = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0xff8a1f, transparent: true, opacity: 0.8 }));
    this.trail.visible = false;
    this.trail.frustumCulled = false;
    this.group.add(this.trail);
  }

  setXray(on) {
    this.xray = on;
    for (const mm of this.materials || []) {
      mm.transparent = on;
      mm.opacity = on ? 0.16 : 1;
      mm.depthWrite = !on;
      mm.needsUpdate = true;
    }
    if (this.xrayGroup) this.xrayGroup.visible = on;
    for (const p of this.pods || []) p.axes.visible = on;
  }

  // f: fractional frame index; sample: current device sample (or null)
  update(f, t, sample) {
    const P = this.payload;
    const fr = P.frames;
    const b = this.dims;
    const tmp = new Array(72);
    // spine line points and segment frames
    lerpFrame(fr.spine, 12, f, tmp);
    const sp = [0, 1, 2, 3].map((i) => new THREE.Vector3(tmp[i * 3], tmp[i * 3 + 1], tmp[i * 3 + 2]));
    const q = [0, 1, 2, 3].map((i) => quatFrame(fr.segq, 4, f, i, new THREE.Quaternion()));
    const ax = q.map((qq) => ({
      x: new THREE.Vector3(1, 0, 0).applyQuaternion(qq), y: new THREE.Vector3(0, 1, 0).applyQuaternion(qq), z: new THREE.Vector3(0, 0, 1).applyQuaternion(qq),
    }));
    lerpFrame(fr.joints, 54, f, tmp);
    const J = [];
    for (let k = 0; k < 18; k++) J.push(new THREE.Vector3(tmp[k * 3], tmp[k * 3 + 1], tmp[k * 3 + 2]));
    this.J = J;
    this.updateTorso(sp, q, ax, J);
    // limbs
    for (const l of this.limbs) {
      const a = J[l.a], bb = J[l.b];
      if (l.trim) {
        const end = a.clone().lerp(bb, 0.86);
        placeBetween(l.mesh, end, a, false);
      } else {
        placeBetween(l.mesh, bb, a, false);
      }
    }
    for (const j of this.joints) j.mesh.position.copy(J[j.j]);
    // boots: heel (6/7) to toe (8/9), up toward the ankle (4/5)
    for (let side = 0; side < 2; side++) {
      const heel = J[6 + side], toe = J[8 + side], ankle = J[4 + side];
      const fwdV = toe.clone().sub(heel);
      const len = Math.max(fwdV.length(), 0.05);
      fwdV.normalize();
      const mid = heel.clone().add(toe).multiplyScalar(0.5);
      const upV = ankle.clone().sub(mid);
      upV.addScaledVector(fwdV, -upV.dot(fwdV)).normalize();
      const rightV = new THREE.Vector3().crossVectors(fwdV, upV).normalize();
      const m = new THREE.Matrix4().makeBasis(rightV, upV, fwdV.clone().negate());
      const g = this.boots[side].group;
      g.quaternion.setFromRotationMatrix(m);
      g.position.copy(mid);
      this.boots[side].upper.scale.z = len + 0.02;
      this.boots[side].sole.scale.z = len + 0.03;
      placeBetween(this.boots[side].cuff, ankle.clone().addScaledVector(upV, -0.02), ankle.clone().addScaledVector(upV, 0.06), false);
    }
    // neck and head
    const hq = quatFrame(fr.headq, 1, f, 0, new THREE.Quaternion());
    const headUp = new THREE.Vector3(0, 1, 0).applyQuaternion(hq);
    this.head.position.copy(J[17]);
    this.head.quaternion.copy(hq);
    const neckTop = J[17].clone().addScaledVector(headUp, -0.06 * b.s);
    const neckBase = J[16].clone().addScaledVector(ax[3].y, -0.02);
    placeBetween(this.neck, neckBase, neckTop);
    this.neck.scale.x = this.neck.scale.z = 1;
    this.updateHat(f);
    this.updateTools(f, t, J, ax);
    this.updateStrip(f, sample, ax);
    if (this.xray) this.updateXray(sp, ax);
  }

  updateTorso(sp, q, ax, J) {
    const b = this.dims, s = b.s, w = b.w;
    const bottom = sp[0].clone().addScaledVector(ax[0].y, -0.085 * b.H);
    const neck = J[16];
    const pts = [bottom, ...sp, neck];
    const tangents = [ax[0].y, ax[0].y, ax[1].y, ax[2].y, ax[3].y, ax[3].y];
    hermite(pts, tangents, RINGS, this.ringCenters);
    const pos = this.torso.geometry.attributes.position;
    const qa = new THREE.Quaternion();
    const xr = new THREE.Vector3(), zr = new THREE.Vector3();
    // segment weights along the curve: s in [0,1] maps to segments by arc proportion
    for (let r = 0; r < RINGS; r++) {
      const sr = r / (RINGS - 1);
      // which segment frame: bottom..P0 -> 0, P0..P1 -> 0->1, P1..P2 -> 1->2, P2..P3 -> 2->3, P3..neck -> 3
      const knots = [0, 0.2, 0.4, 0.62, 0.9, 1.0];
      let k = 0;
      while (k < knots.length - 2 && sr > knots[k + 1]) k++;
      const u = (sr - knots[k]) / (knots[k + 1] - knots[k]);
      const segA = [0, 0, 1, 2, 3][k], segB = [0, 1, 2, 3, 3][k];
      qa.copy(q[segA]).slerp(q[segB], u);
      xr.set(1, 0, 0).applyQuaternion(qa);
      zr.set(0, 0, 1).applyQuaternion(qa);
      const [a0, b0] = profileAt(sr);
      const a = a0 * s * w, bd = b0 * s * Math.sqrt(w);
      const c = this.ringCenters[r];
      const cx = c.x - zr.x * (bd - 0.04), cy = c.y - zr.y * (bd - 0.04), cz = c.z - zr.z * (bd - 0.04);
      for (let kk = 0; kk < RADIAL; kk++) {
        const ang = (kk / RADIAL) * Math.PI * 2;
        const ca = Math.cos(ang), sa = Math.sin(ang);
        // flatten the back a little (super-ellipse) so the strip lies on a plausible surface
        const back = sa > 0 ? 0.92 : 1.0;
        pos.setXYZ(r * RADIAL + kk, cx + xr.x * a * ca + zr.x * bd * sa * back, cy + xr.y * a * ca + zr.y * bd * sa * back, cz + xr.z * a * ca + zr.z * bd * sa * back);
      }
    }
    pos.needsUpdate = true;
    this.torso.geometry.computeVertexNormals();
  }

  updateHat(f) {
    const dyn = this.payload.dyn || {};
    if (!dyn.hat_on) return;
    const on = dyn.hat_on[Math.min(Math.round(f), dyn.hat_on.length - 1)] > 0.5;
    if (on) {
      if (this.hat.parent !== this.head) { this.head.add(this.hat); this.hat.position.copy(this.hatOffset); this.hat.quaternion.identity(); }
      this.hatCable.visible = false;
      return;
    }
    if (this.hat.parent !== this.group) this.group.add(this.hat);
    const tmp = lerpFrame(dyn.hat, 7, f, new Array(7));
    this.hat.position.set(tmp[0], tmp[1], tmp[2]);
    this.hat.quaternion.set(tmp[3], tmp[4], tmp[5], tmp[6]).normalize();
    const a = lerpFrame(this.payload.frames.anchors, 12, f, new Array(12));
    const pos = this.hatCable.geometry.attributes.position;
    pos.setXYZ(0, tmp[0], tmp[1], tmp[2]); pos.setXYZ(1, a[9], a[10], a[11]); pos.needsUpdate = true;
    this.hatCable.visible = true;
  }

  updateTools(f, t, J, ax) {
    const tool = this.payload.frames.tool[Math.min(Math.round(f), this.payload.frames.tool.length - 1)];
    this.hammer.visible = tool === 1;
    this.drill.visible = tool === 2;
    const rest = this.payload.dyn && this.payload.dyn.jack_rest;
    this.jack.visible = tool === 3 || !!rest;
    this.dust.visible = false;
    const handR = J[15], elbowR = J[13];
    if (tool === 1) {
      const fore = handR.clone().sub(elbowR).normalize();
      const dir = new THREE.Vector3().crossVectors(ax[3].x, fore).normalize();
      this.hammer.position.copy(handR).addScaledVector(dir, -0.04);
      this.hammer.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
    } else if (tool === 2) {
      this.drill.position.copy(handR);
      const yaw = Math.atan2(-ax[3].z.x, -ax[3].z.z);
      this.drill.rotation.set(Math.PI / 2, 0, 0);
      this.drill.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), yaw + Math.PI);
      this.dust.visible = true;
      const tip = handR.clone().add(new THREE.Vector3(0, 0.25, 0));
      const pos = this.dust.geometry.attributes.position;
      for (let i = 0; i < pos.count; i++) {
        const ph = (t * 1.7 + i * 0.137) % 1;
        pos.setXYZ(i, tip.x + Math.sin(i * 12.9) * 0.25 * ph, tip.y - ph * 1.4, tip.z + Math.cos(i * 7.1) * 0.25 * ph);
      }
      pos.needsUpdate = true;
    }
    if (tool === 3) {
      const mid = J[14].clone().add(J[15]).multiplyScalar(0.5);
      const ground = Math.min(J[8].y, J[9].y) - 0.0;
      const bottom = new THREE.Vector3(mid.x, ground + 0.02, mid.z);
      const top = mid.clone().add(new THREE.Vector3(0, -0.06, 0));
      this.jack.position.set(0, 0, 0);
      this.jack.quaternion.identity();
      const jb = this.jackParts.jb;
      placeBetween(jb, bottom.clone().add(new THREE.Vector3(0, 0.25, 0)), top);
      jb.scale.x = jb.scale.z = 1;
      this.jackParts.chisel.position.copy(bottom).add(new THREE.Vector3(0, 0.12, 0));
      this.jackParts.chisel.quaternion.identity();
      placeBetween(this.jackParts.bar, J[14], J[15], false);
      const shake = Math.sin(t * 160) * 0.004;
      this.jack.position.x = shake;
    } else if (rest) {
      const p = new THREE.Vector3(rest[0], rest[1] + 0.09, rest[2]);
      const dir = new THREE.Vector3(Math.cos(rest[3]), 0, -Math.sin(rest[3]));
      const jb = this.jackParts.jb;
      this.jack.position.set(0, 0, 0); this.jack.quaternion.identity();
      placeBetween(jb, p.clone().addScaledVector(dir, -0.35), p.clone().addScaledVector(dir, 0.35));
      this.jackParts.chisel.position.copy(p).addScaledVector(dir, -0.45);
      this.jackParts.chisel.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0), dir.clone().negate());
      placeBetween(this.jackParts.bar, p.clone().addScaledVector(dir, 0.4).add(new THREE.Vector3(0.2, 0, 0)), p.clone().addScaledVector(dir, 0.4).add(new THREE.Vector3(-0.2, 0, 0)), false);
    }
  }

  updateStrip(f, sample, ax) {
    const fr = this.payload.frames;
    const a = lerpFrame(fr.anchors, 12, f, new Array(12));
    const anchors = [0, 1, 2, 3].map((i) => new THREE.Vector3(a[i * 3], a[i * 3 + 1], a[i * 3 + 2]));
    const chip = [0, 1, 2, 3].map((i) => quatFrame(fr.chipq, 4, f, i, new THREE.Quaternion()));
    const dead = this.payload.meta.dead_slot;
    const blink = (performance.now() % 1000) < 120;
    this.anchorsNow = anchors;
    for (let i = 0; i < 4; i++) {
      const p = this.pods[i];
      const zc = new THREE.Vector3(0, 0, 1).applyQuaternion(chip[i]);
      p.group.position.copy(anchors[i]).addScaledVector(zc, 0.01);
      p.group.quaternion.copy(chip[i]);
      const isDead = dead === i;
      p.shell.material.color.set(isDead ? 0x333333 : SLOT_COLORS[i]);
      p.shell.material.emissiveIntensity = isDead ? 0 : (sample && sample.sat[i] ? 1.2 : 0.25);
      if (sample && sample.sat[i]) p.shell.material.emissive.set(0xff0000); else p.shell.material.emissive.set(SLOT_COLORS[i]);
      p.led.material.color.set(isDead ? 0x550000 : (sample && !sample.ok ? 0x444444 : (blink ? 0xffffff : 0x7dff7d)));
      // vectors: specific force (g) and angular rate (rad/s), world frame, from the device sample
      const showF = this.showForces && sample && !isDead;
      if (showF) {
        const v = new THREE.Vector3(sample.fw[i * 3], sample.fw[i * 3 + 1], sample.fw[i * 3 + 2]);
        const len = v.length();
        p.force.visible = len > 0.04;
        if (p.force.visible) {
          p.force.position.copy(p.group.position).addScaledVector(zc, 0.012);
          p.force.setDirection(v.clone().normalize());
          const L = Math.min(len * 0.13, 1.6);
          p.force.setLength(L, Math.min(0.04, L * 0.35), Math.min(0.025, L * 0.25));
        }
      } else p.force.visible = false;
      const showG = this.showGyro && sample && !isDead;
      if (showG) {
        const v = new THREE.Vector3(sample.ww[i * 3], sample.ww[i * 3 + 1], sample.ww[i * 3 + 2]);
        const len = v.length();
        p.gyro.visible = len > 0.05;
        if (p.gyro.visible) {
          p.gyro.position.copy(p.group.position);
          p.gyro.setDirection(v.clone().normalize());
          const L = Math.min(len * 0.05, 1.2);
          p.gyro.setLength(L, Math.min(0.03, L * 0.35), Math.min(0.02, L * 0.25));
        }
      } else p.gyro.visible = false;
    }
    // strap: Hermite through the anchors with the chip y axes as tangents, extended at both ends
    const ys = chip.map((qq) => new THREE.Vector3(0, 1, 0).applyQuaternion(qq));
    const xs = chip.map((qq) => new THREE.Vector3(1, 0, 0).applyQuaternion(qq));
    const lo = anchors[0].clone().addScaledVector(ys[0], -0.05), hi = anchors[3].clone().addScaledVector(ys[3], 0.05);
    hermite([lo, ...anchors, hi], [ys[0], ...ys, ys[3]], this.strapPts.length, this.strapPts);
    const pos = this.strap.geometry.attributes.position;
    const n = this.strapPts.length;
    for (let k = 0; k < n; k++) {
      const seg = Math.min(3, Math.floor((k / (n - 1)) * 4));
      const x = xs[seg];
      const z = new THREE.Vector3(0, 0, 1).applyQuaternion(chip[seg]);
      const c = this.strapPts[k].clone().addScaledVector(z, 0.002);
      pos.setXYZ(2 * k, c.x - x.x * 0.012, c.y - x.y * 0.012, c.z - x.z * 0.012);
      pos.setXYZ(2 * k + 1, c.x + x.x * 0.012, c.y + x.y * 0.012, c.z + x.z * 0.012);
    }
    pos.needsUpdate = true;
    this.strap.geometry.computeVertexNormals();
    this.hub.position.copy(anchors[0]).addScaledVector(ys[0], -0.075).addScaledVector(new THREE.Vector3(0, 0, 1).applyQuaternion(chip[0]), 0.01);
    this.hub.quaternion.copy(chip[0]);
  }

  updateXray(sp, ax) {
    for (let i = 0; i < 3; i++) {
      placeBetween(this.segLines[i], sp[i], sp[i + 1]);
      this.segLines[i].scale.x = this.segLines[i].scale.z = 1;
    }
    for (let i = 0; i < 4; i++) {
      this.vertebrae[i].position.copy(sp[i]);
      const m = new THREE.Matrix4().makeBasis(ax[i].x, ax[i].y, ax[i].z);
      this.vertebrae[i].quaternion.setFromRotationMatrix(m);
    }
    const pts = Array.from({ length: 40 }, () => new THREE.Vector3());
    hermite(this.anchorsNow, this.pods.map((p) => new THREE.Vector3(0, 1, 0).applyQuaternion(p.group.quaternion)), 40, pts);
    const pos = this.curveLine.geometry.attributes.position;
    pts.forEach((p, k) => pos.setXYZ(k, p.x, p.y, p.z));
    pos.needsUpdate = true;
  }

  pelvis(f, out = new THREE.Vector3()) {
    const tmp = lerpFrame(this.payload.frames.spine, 12, f, new Array(12));
    return out.set(tmp[0], tmp[1], tmp[2]);
  }
}
