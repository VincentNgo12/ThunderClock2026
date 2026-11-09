// File: world.js
// Purpose: Builds the physical scene for one run (ground, site props, ladders, scaffold,
//          roof, trench, stairs, net, harness anchor, crane load, vehicle...) from the
//          payload's scene description, and animates the moving props each frame (the
//          lanyard rope and the safety-net cloth come as keyframes from scene.py's physics).
import * as THREE from "three";
import { surfaceTexture, radialTexture, crackTexture, gridTexture, chainLinkTexture, lerpFrame, placeBetween } from "./util.js";

const MATS = {};
function mat(name) {
  if (MATS[name]) return MATS[name];
  const spec = {
    concrete: { color: 0x9c9a95, roughness: 0.92 },
    plank: { color: 0xb08850, roughness: 0.85 },
    plywood: { color: 0xc49a5c, roughness: 0.8 },
    lumber: { color: 0xc7a066, roughness: 0.8 },
    soil: { color: 0x7a5c40, roughness: 1 },
    steel: { color: 0x7d858e, roughness: 0.45, metalness: 0.6 },
    scaffold: { color: 0xa9b0b7, roughness: 0.4, metalness: 0.7 },
    rail: { color: 0xffc21a, roughness: 0.5, metalness: 0.2 },
    aluminium: { color: 0xc9ced3, roughness: 0.35, metalness: 0.75 },
    brick: { color: 0x9b5a43, roughness: 0.95 },
    wall: { color: 0xd8d2c4, roughness: 0.95 },
    roof: { color: 0x3b3f45, roughness: 0.9 },
    roofdeck: { color: 0x5a5f66, roughness: 0.9 },
    yellow: { color: 0xf2b705, roughness: 0.55, metalness: 0.1 },
    white: { color: 0xeeeeee, roughness: 0.6 },
    hazard: { color: 0xffd400, roughness: 0.7 },
    block: { color: 0xa8a59e, roughness: 0.95 },
    bucket: { color: 0xff7a1a, roughness: 0.6 },
    cardboard: { color: 0xb08a5a, roughness: 1 },
    tarp: { color: 0x2f6fd0, roughness: 0.8 },
    tile: { color: 0xd9d4ca, roughness: 0.4 },
    black: { color: 0x1d1f22, roughness: 0.7 },
    glass: { color: 0x9fc6e0, roughness: 0.05, metalness: 0.1, transparent: true, opacity: 0.25 },
    rubber: { color: 0x222222, roughness: 0.9 },
    orange: { color: 0xff6a00, roughness: 0.6 },
    red: { color: 0xc8302c, roughness: 0.6 },
    blue: { color: 0x2b5fb3, roughness: 0.6 },
    green: { color: 0x3b7d3b, roughness: 0.7 },
    rebar: { color: 0x7a4a2e, roughness: 0.7, metalness: 0.4 },
  }[name] || { color: 0x888888, roughness: 0.8 };
  MATS[name] = new THREE.MeshStandardMaterial(spec);
  return MATS[name];
}

function box(size, material, castShadow = true) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(size[0], size[1], size[2]), typeof material === "string" ? mat(material) : material);
  m.castShadow = castShadow; m.receiveShadow = true;
  return m;
}

function cyl(rTop, rBottom, h, material, seg = 16) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(rTop, rBottom, h, seg), typeof material === "string" ? mat(material) : material);
  m.castShadow = true; m.receiveShadow = true;
  return m;
}

export class World {
  constructor(scene) {
    this.scene = scene;
    this.root = new THREE.Group();
    scene.add(this.root);
    this.dynamic = {};
  }

  clear() {
    this.root.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
    this.scene.remove(this.root);
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.dynamic = {};
  }

  build(payload) {
    this.clear();
    this.payload = payload;
    const sc = payload.scene;
    this.addGround(sc);
    for (const p of sc.props) this.addProp(p);
  }

  // Top of the highest solid (deck, slab, soil block) under (x, z); the camera stays above it.
  solidTop(x, z) {
    let top = this.payload.scene.ground_y;
    for (const s of this.payload.scene.solids || []) {
      const dx = x - s.c[0], dz = z - s.c[1];
      const a = dx * Math.cos(s.yaw) - dz * Math.sin(s.yaw);
      const b = -dx * Math.sin(s.yaw) - dz * Math.cos(s.yaw);
      if (Math.abs(a) <= s.h[0] + 0.3 && Math.abs(b) <= s.h[1] + 0.3 && s.y > top) top = s.y;
    }
    return top;
  }

  addGround(sc) {
    const { tex, tile, roughness } = surfaceTexture(sc.surface, sc.condition);
    const size = 1200;
    const t = tex.clone();
    t.needsUpdate = true;
    t.repeat.set(size / tile, size / tile);
    const g = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshStandardMaterial({ map: t, roughness, metalness: 0 }));
    g.rotation.x = -Math.PI / 2;
    g.position.set(sc.focus[0], sc.ground_y, sc.focus[2]);
    g.receiveShadow = true;
    this.root.add(g);
    // faint 1 m grid near the action, for scale
    const gt = gridTexture(10, "rgba(255,255,255,.55)", 1.5).clone();
    gt.needsUpdate = true;
    gt.repeat.set(4, 4);
    const grid = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshBasicMaterial({ map: gt, transparent: true, opacity: 0.07, depthWrite: false }));
    grid.rotation.x = -Math.PI / 2;
    grid.position.set(Math.round(sc.focus[0] / 4) * 4, sc.ground_y + 0.003, Math.round(sc.focus[2] / 4) * 4);
    grid.renderOrder = 1;
    this.grid = grid;
    this.root.add(grid);
  }

  addProp(p) {
    const fn = {
      box: () => this.propBox(p), cyl: () => this.propCyl(p), tube: () => this.propTube(p), ladder: () => this.propLadder(p),
      wedge: () => this.propWedge(p), patch: () => this.propPatch(p), net: () => this.propNet(p), rebar: () => this.propRebar(p),
      decor: () => this.propDecor(p), fence: () => this.propFence(p), dyn: () => this.propDyn(p),
    }[p.t];
    if (fn) fn();
  }

  propBox(p) {
    let m;
    if (p.m === "brick" || p.m === "wall" || p.m === "concrete" || p.m === "soil" || p.m === "plank" || p.m === "plywood" || p.m === "block") {
      m = box(p.s, texturedMat(p.m, p.s));
    } else {
      m = box(p.s, p.m);
    }
    m.position.set(...p.p);
    m.rotation.y = p.yaw;
    this.root.add(m);
  }

  propCyl(p) {
    const m = cyl(p.rt, p.r, p.h, p.m, 24);
    m.position.set(p.p[0], p.p[1] + p.h / 2, p.p[2]);
    this.root.add(m);
    if (p.m === "bucket") {
      const rim = new THREE.Mesh(new THREE.TorusGeometry(p.rt, 0.012, 6, 24), mat("bucket"));
      rim.rotation.x = Math.PI / 2; rim.position.set(p.p[0], p.p[1] + p.h, p.p[2]);
      this.root.add(rim);
    }
  }

  propTube(p) {
    const a = new THREE.Vector3(...p.a), b = new THREE.Vector3(...p.b);
    const m = cyl(p.r, p.r, 1, p.m, 10);
    placeBetween(m, a, b);
    this.root.add(m);
  }

  propLadder(p) {
    const g = ladderMesh(p.top_y - p.base[1], p.w, p.m, 0.3);
    g.position.set(...p.base);
    g.rotation.y = p.yaw;
    this.root.add(g);
  }

  propWedge(p) {
    // Sloped slab (roof or trench bank): top edge at p.p, descending along yaw by `slope` deg for
    // `len` metres (horizontal), `w` wide, solid down to `base`.
    const s = Math.tan(p.slope * Math.PI / 180) * p.len;
    const y0 = p.p[1], y1 = y0 - s, yb = Math.min(p.base, y1 - 0.15);
    const hw = p.w / 2;
    // local: x across, z forward = -fwd? use fwd along -Z like three convention
    const v = [
      [-hw, y0 - yb, 0], [hw, y0 - yb, 0], [hw, y1 - yb, -p.len], [-hw, y1 - yb, -p.len],
      [-hw, 0, 0], [hw, 0, 0], [hw, 0, -p.len], [-hw, 0, -p.len],
    ];
    const idx = [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7, 0, 3, 7, 0, 7, 4, 1, 5, 6, 1, 6, 2];
    const pos = [];
    for (let i = 0; i < idx.length; i++) pos.push(...v[idx[i]]);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.computeVertexNormals();
    const m = new THREE.Mesh(geo, p.m === "roof" ? roofMat() : texturedMat(p.m, [p.w, 1, p.len]));
    m.material.side = THREE.DoubleSide;
    m.castShadow = true; m.receiveShadow = true;
    m.position.set(p.p[0], yb, p.p[2]);
    m.rotation.y = p.yaw;
    this.root.add(m);
  }

  propPatch(p) {
    let material;
    if (p.k === "broken") {
      material = new THREE.MeshStandardMaterial({ map: crackTexture(), transparent: true, roughness: 1, depthWrite: false });
    } else {
      const colors = { water: ["rgba(40,60,80,.75)", "rgba(40,60,80,0)"], ice: ["rgba(215,235,255,.85)", "rgba(215,235,255,0)"],
        oil: ["rgba(15,12,10,.85)", "rgba(15,12,10,0)"], mud: ["rgba(70,45,25,.95)", "rgba(70,45,25,0)"], dust: ["rgba(205,185,145,.6)", "rgba(205,185,145,0)"] }[p.k];
      material = new THREE.MeshStandardMaterial({ map: radialTexture(colors[0], colors[1]), transparent: true, depthWrite: false,
        roughness: p.k === "water" || p.k === "ice" || p.k === "oil" ? 0.08 : 0.9, metalness: p.k === "oil" ? 0.4 : 0 });
    }
    const m = new THREE.Mesh(new THREE.CircleGeometry(p.r, 40), material);
    m.rotation.x = -Math.PI / 2;
    m.scale.set(1, 0.75, 1);
    m.position.set(p.p[0], p.p[1] + 0.004, p.p[2]);
    m.receiveShadow = true;
    m.renderOrder = 2;
    this.root.add(m);
  }

  propNet(p) {
    const segs = 48;
    const geo = new THREE.PlaneGeometry(p.w, p.l, segs, segs);
    geo.rotateX(-Math.PI / 2);
    const t = gridTexture(12, "rgba(30,30,30,1)", 5).clone();
    t.needsUpdate = true;
    t.repeat.set(p.w / 1.2, p.l / 1.2);
    const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ map: t, transparent: true, alphaTest: 0.3, side: THREE.DoubleSide, roughness: 1 }));
    m.castShadow = true;
    m.position.set(p.p[0], p.p[1], p.p[2]);
    m.rotation.y = p.yaw;
    this.root.add(m);
    const border = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(p.w, 0.001, p.l)), new THREE.LineBasicMaterial({ color: 0x111111 }));
    border.position.copy(m.position); border.rotation.y = p.yaw;
    this.root.add(border);
    // Flat stand-in; buildDynamics() replaces it with the simulated cloth (dyn.net_cloth).
    this.dynamic.net = { mesh: m };
  }

  propRebar(p) {
    const group = new THREE.Group();
    const r = 0.008;
    const nx = Math.floor(p.w / p.sp), nz = Math.floor(p.l / p.sp);
    const geo = new THREE.CylinderGeometry(r, r, 1, 6);
    const m = mat("rebar");
    const inst = new THREE.InstancedMesh(geo, m, nx + 1 + nz + 1);
    const dummy = new THREE.Object3D();
    let k = 0;
    for (let i = 0; i <= nx; i++) {
      dummy.position.set(-p.w / 2 + i * p.sp, 0.03, 0); dummy.rotation.set(Math.PI / 2, 0, 0); dummy.scale.set(1, p.l, 1);
      dummy.updateMatrix(); inst.setMatrixAt(k++, dummy.matrix);
    }
    for (let j = 0; j <= nz; j++) {
      dummy.position.set(0, 0.045, -p.l / 2 + j * p.sp); dummy.rotation.set(0, 0, Math.PI / 2); dummy.scale.set(1, p.w, 1);
      dummy.updateMatrix(); inst.setMatrixAt(k++, dummy.matrix);
    }
    inst.castShadow = true; inst.receiveShadow = true;
    group.add(inst);
    group.position.set(...p.p);
    group.rotation.y = p.yaw;
    this.root.add(group);
  }

  propFence(p) {
    const r = p.r, h = 2.0;
    const t = chainLinkTexture().clone();
    t.needsUpdate = true;
    t.repeat.set(Math.round(2 * Math.PI * r / 0.6), Math.round(h / 0.6));
    const mesh = new THREE.Mesh(new THREE.CylinderGeometry(r, r, h, 96, 1, true),
      new THREE.MeshStandardMaterial({ map: t, transparent: true, alphaTest: 0.25, side: THREE.DoubleSide, roughness: 0.6, metalness: 0.5 }));
    mesh.position.set(p.c[0], p.c[1] + h / 2, p.c[2]);
    this.root.add(mesh);
    const n = Math.round(2 * Math.PI * r / 3);
    const posts = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.03, 0.03, h + 0.1, 6), mat("scaffold"), n);
    const feet = new THREE.InstancedMesh(new THREE.BoxGeometry(0.6, 0.12, 0.25), mat("concrete"), n);
    const d = new THREE.Object3D();
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      d.position.set(p.c[0] + Math.cos(a) * r, p.c[1] + (h + 0.1) / 2, p.c[2] + Math.sin(a) * r); d.rotation.set(0, -a, 0); d.updateMatrix();
      posts.setMatrixAt(i, d.matrix);
      d.position.y = p.c[1] + 0.06; d.updateMatrix(); feet.setMatrixAt(i, d.matrix);
    }
    this.root.add(posts, feet);
  }

  propDecor(p) {
    const g = decorMesh(p.k, p);
    if (!g) return;
    g.position.set(...p.p);
    g.rotation.y = p.yaw;
    this.root.add(g);
  }

  propDyn(p) {
    if (p.k === "crane_load") {
      // Beam + slings + hook block, hung from the rope; origin at the beam centre (the pendulum
      // bob). Tilted about the beam axis by the swing angle so the slings stay in line with the rope.
      const g = new THREE.Group();
      g.add(beamMesh(2.6, "red"));
      const slingMat = new THREE.LineBasicMaterial({ color: 0x222222 });
      const sl = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-1.1, 0.14, 0), new THREE.Vector3(0, 1.25, 0), new THREE.Vector3(1.1, 0.14, 0)]);
      g.add(new THREE.Line(sl, slingMat));
      const hook = box([0.18, 0.3, 0.12], "yellow"); hook.position.y = 1.43; g.add(hook);
      g.rotation.order = "YXZ";
      g.rotation.y = p.yaw;
      this.root.add(g);
      const cableGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
      const cable = new THREE.Line(cableGeo, new THREE.LineBasicMaterial({ color: 0x111111 }));
      this.root.add(cable);
      this.dynamic.crane = { group: g, cable, yaw: p.yaw, pivot: null, hookTop: 1.6 };
    } else if (p.k === "bags") {
      this.bagSize = p.size;
    } else if (p.k === "lanyard") {
      // Anchor eye; the webbing, absorber pack and snap hook are built in buildDynamics().
      const eye = new THREE.Mesh(new THREE.TorusGeometry(0.05, 0.012, 8, 16), mat("steel"));
      eye.position.set(...p.anchor); this.root.add(eye);
    } else if (p.k === "vehicle") {
      const g = loaderMesh(p.floor ?? 0.9, p.seat ?? 1.35);
      this.root.add(g);
      this.dynamic.vehicle = { group: g };
    } else if (p.k === "stepladder") {
      const g = stepladderMesh(p.h);
      this.root.add(g);
      this.dynamic.stepladder = { group: g, h: p.h };
    } else if (p.k === "creeper") {
      // Low-profile creeper: 25 mm board on 50 mm casters, top at p.top (6.5 cm).
      const top = p.top ?? 0.065;
      const g = new THREE.Group();
      const board = box([0.42, 0.025, 1.0], "black"); board.position.y = top - 0.0125; g.add(board);
      const pad = box([0.34, 0.012, 0.3], "blue"); pad.position.set(0, top + 0.002, -0.32); g.add(pad);
      for (const x of [-0.17, 0.17]) for (const z of [-0.42, 0.42]) { const w = cyl(0.025, 0.025, 0.025, "rubber", 10); w.rotation.z = Math.PI / 2; w.position.set(x, 0.025, z); g.add(w); }
      this.root.add(g);
      this.dynamic.creeper = { group: g };
    }
  }

  // Build moving props that need data (called after build when payload.dyn exists).
  buildDynamics(payload) {
    const dyn = payload.dyn || {};
    const frames = payload.meta ? payload.meta.frames : 0;
    if (dyn.bag0) {
      const size = this.bagSize || [0.32, 0.55, 0.16];
      this.dynamic.bags = [];
      for (let k = 0; dyn[`bag${k}`]; k++) {
        // Standing sack (across, height, depth) with a printed label on its front.
        const m = box(size, new THREE.MeshStandardMaterial({ color: 0xd9cfb8, roughness: 0.95 }));
        const label = box([size[0] * 0.7, size[1] * 0.45, size[2] + 0.004], new THREE.MeshStandardMaterial({ color: 0x2d5fa8, roughness: 0.9 }), false);
        m.add(label);
        this.root.add(m);
        this.dynamic.bags.push({ mesh: m, data: dyn[`bag${k}`] });
      }
    }
    if (dyn.jack_rest) this.dynamic.jackRest = dyn.jack_rest;
    if (dyn.crane_rig && this.dynamic.crane) {
      const rig = dyn.crane_rig;
      this.dynamic.crane.pivot = new THREE.Vector3(...rig.pivot);
      this.dynamic.crane.hookTop = rig.hook_top;
      this.dynamic.crane.stride = frames ? Math.round(dyn.crane.length / frames) : 4;
      this.root.add(cribMesh(rig.crib, rig.yaw));
      this.root.add(mobileCraneMesh(rig.boom));
    }
    if (dyn.vehicle && this.dynamic.vehicle) this.dynamic.vehicle.stride = frames ? Math.round(dyn.vehicle.length / frames) : 6;
    if (dyn.lanyard && dyn.lanyard.keys) {
      // Webbing as a tube through the simulated rope points; absorber pack at the D-ring end,
      // snap hook at the other.
      const L = dyn.lanyard;
      const rope = ropeMesh(L.n, 0.012, 8, new THREE.MeshStandardMaterial({ color: 0xff7a00, roughness: 0.7 }));
      const pack = box([0.07, 0.15, 0.045], new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.8 }));
      const hook = box([0.03, 0.09, 0.015], mat("steel"));
      this.root.add(rope, pack, hook);
      this.dynamic.lanyard = { rope, pack, hook, keys: L.keys, x: L.x, stride: L.n * 3, pts: new Float32Array(L.n * 3) };
    }
    if (dyn.net_cloth && this.dynamic.net) {
      // Safety net cloth: rest grid + each key's node offsets (mm), decoded from the sparse
      // changes between keys (scene.py sparse_steps: (zeros skipped, change) pairs).
      const C = dyn.net_cloth, stride = C.nx * C.nz * 3;
      const offsets = new Int32Array(C.keys.length * stride);
      let at = 0;
      for (let k = 0; k < C.keys.length; k++) {
        const base = k * stride;
        if (k) offsets.copyWithin(base, base - stride, base);
        for (let c = 0, i = -1; c < C.nnz[k]; c++, at += 2) {
          i += C.d[at] + 1;
          offsets[base + i] += C.d[at + 1];
        }
      }
      const old = this.dynamic.net.mesh;
      old.visible = false;
      const mesh = clothMesh(C.nx, C.nz, C.rest, old.material);
      this.root.add(mesh);
      this.dynamic.cloth = { mesh, rest: Float32Array.from(C.rest), keys: C.keys, offsets, stride };
    }
  }

  update(f, payload, body) {
    const dyn = payload.dyn || {};
    const tmp = new Array(8);
    const D = this.dynamic;
    if (D.crane && dyn.crane && D.crane.pivot) {
      lerpFrame(dyn.crane, D.crane.stride, f, tmp);
      const g = D.crane.group;
      g.position.set(tmp[0], tmp[1], tmp[2]);
      g.rotation.set(tmp[3] || 0, D.crane.yaw, 0);
      // Rope from the top of the hook block to the boom-tip sheave (the pendulum pivot).
      const dir = D.crane.pivot.clone().sub(g.position).normalize();
      const start = g.position.clone().addScaledVector(dir, D.crane.hookTop);
      const pos = D.crane.cable.geometry.attributes.position;
      pos.setXYZ(0, start.x, start.y, start.z); pos.setXYZ(1, D.crane.pivot.x, D.crane.pivot.y, D.crane.pivot.z); pos.needsUpdate = true;
    }
    if (D.lanyard) {
      const L = D.lanyard, p = L.pts;
      keyedFrame(L.keys, L.x, L.stride, f, p);
      setRope(L.rope, p);
      // absorber pack centred 9 cm along the rope from the D-ring, snap hook on the last point
      _a.set(p[0], p[1], p[2]); _b.set(p[3], p[4], p[5]);
      placeBetween(L.pack, _a, _b, false);
      L.pack.position.lerpVectors(_a, _b, Math.min(0.09 / Math.max(_a.distanceTo(_b), 1e-6), 1));
      const n = p.length / 3;
      _a.set(p[3 * n - 6], p[3 * n - 5], p[3 * n - 4]); _b.set(p[3 * n - 3], p[3 * n - 2], p[3 * n - 1]);
      placeBetween(L.hook, _a, _b, false);
      L.hook.position.copy(_b);
    }
    if (D.cloth) {
      const C = D.cloth, pos = C.mesh.geometry.attributes.position;
      const [i0, i1, a] = keyBracket(C.keys, f);
      const o0 = i0 * C.stride, o1 = i1 * C.stride;
      for (let i = 0; i < C.stride; i++) pos.array[i] = C.rest[i] + (C.offsets[o0 + i] + (C.offsets[o1 + i] - C.offsets[o0 + i]) * a) / 1000;
      pos.needsUpdate = true;
      C.mesh.geometry.computeVertexNormals();
    }
    if (D.vehicle && dyn.vehicle) {
      // Chassis on the ground; the suspension seat carries the operator's bounce; wheels turn by
      // the odometer channel (deterministic when scrubbing).
      const stride = D.vehicle.stride || 6;
      lerpFrame(dyn.vehicle, stride, f, tmp);
      const g = D.vehicle.group;
      g.position.set(tmp[0], tmp[1], tmp[2]); g.rotation.y = tmp[3];
      if (stride >= 6) {
        g.userData.setSeat(tmp[4]);
        for (const w of g.userData.wheels) w.rotation.x = -tmp[5] / 0.6;
      }
    }
    if (D.stepladder && dyn.stepladder) {
      lerpFrame(dyn.stepladder, 5, f, tmp);
      const g = D.stepladder.group;
      const open = tmp[4];
      g.position.set(tmp[0], tmp[1], tmp[2]);
      g.rotation.set(0, tmp[3], 0);
      g.userData.setOpen(open);
    }
    if (D.creeper && dyn.creeper) {
      lerpFrame(dyn.creeper, 4, f, tmp);
      D.creeper.group.position.set(tmp[0], tmp[1], tmp[2]);
      D.creeper.group.rotation.y = tmp[3];
    }
    if (D.bags) {
      for (const b of D.bags) {
        lerpFrame(b.data, 4, f, tmp);
        b.mesh.position.set(tmp[0], tmp[1], tmp[2]);
        b.mesh.rotation.y = tmp[3];
      }
    }
  }
}

// --- keyframed channels (scene.py keyframes(): linear interpolation between key frames) ----------
// Bracketing keys of frame f: [index before, index after, blend 0..1].
function keyBracket(keys, f) {
  const last = keys.length - 1;
  if (f <= keys[0]) return [0, 0, 0];
  if (f >= keys[last]) return [last, last, 0];
  let lo = 0, hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (keys[mid] <= f) lo = mid; else hi = mid;
  }
  return [lo, hi, (f - keys[lo]) / (keys[hi] - keys[lo])];
}

function keyedFrame(keys, data, stride, f, out) {
  const [i0, i1, a] = keyBracket(keys, f);
  const o0 = i0 * stride, o1 = i1 * stride;
  for (let k = 0; k < stride; k++) out[k] = data[o0 + k] + (data[o1 + k] - data[o0 + k]) * a;
  return out;
}

// --- rope tube (lanyard webbing) -------------------------------------------------------------
const _a = new THREE.Vector3(), _b = new THREE.Vector3();
const _t = new THREE.Vector3(), _nrm = new THREE.Vector3(), _bin = new THREE.Vector3();

function ropeMesh(n, radius, radial, material) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(n * radial * 3), 3));
  const idx = [];
  for (let i = 0; i < n - 1; i++) {
    for (let k = 0; k < radial; k++) {
      const a = i * radial + k, b = i * radial + (k + 1) % radial;
      idx.push(a, b, a + radial, b, b + radial, a + radial);
    }
  }
  geo.setIndex(idx);
  const mesh = new THREE.Mesh(geo, material);
  mesh.castShadow = true;
  mesh.frustumCulled = false;
  mesh.userData.rope = { n, radius, radial };
  return mesh;
}

// Rings around each point, oriented by parallel transport along the rope (no twisting flips).
function setRope(mesh, pts) {
  const { n, radius, radial } = mesh.userData.rope;
  const pos = mesh.geometry.attributes.position;
  for (let i = 0; i < n; i++) {
    const a = Math.max(i - 1, 0), b = Math.min(i + 1, n - 1);
    _t.set(pts[3 * b] - pts[3 * a], pts[3 * b + 1] - pts[3 * a + 1], pts[3 * b + 2] - pts[3 * a + 2]);
    if (_t.lengthSq() < 1e-12) _t.set(0, 1, 0);
    _t.normalize();
    if (i === 0) _nrm.set(Math.abs(_t.y) < 0.9 ? 0 : 1, Math.abs(_t.y) < 0.9 ? 1 : 0, 0);
    _nrm.addScaledVector(_t, -_nrm.dot(_t));
    if (_nrm.lengthSq() < 1e-12) _nrm.set(_t.y, -_t.x, 0).addScaledVector(_t, -_t.y * _t.y);
    _nrm.normalize();
    _bin.crossVectors(_t, _nrm);
    for (let k = 0; k < radial; k++) {
      const ang = (k / radial) * Math.PI * 2, c = Math.cos(ang) * radius, s = Math.sin(ang) * radius;
      pos.setXYZ(i * radial + k, pts[3 * i] + _nrm.x * c + _bin.x * s, pts[3 * i + 1] + _nrm.y * c + _bin.y * s,
        pts[3 * i + 2] + _nrm.z * c + _bin.z * s);
    }
  }
  pos.needsUpdate = true;
  mesh.geometry.computeVertexNormals();
}

// --- safety-net cloth --------------------------------------------------------------------------
// Grid of nx * nz vertices (row-major along the net's length) in world coordinates, with the
// net texture mapped over the rest grid.
function clothMesh(nx, nz, rest, material) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(rest), 3));
  const uv = new Float32Array(nx * nz * 2);
  for (let j = 0; j < nz; j++) {
    for (let i = 0; i < nx; i++) {
      uv[(j * nx + i) * 2] = i / (nx - 1);
      uv[(j * nx + i) * 2 + 1] = j / (nz - 1);
    }
  }
  geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  const idx = [];
  for (let j = 0; j < nz - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = j * nx + i, b = a + 1, c = a + nx, d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
  }
  geo.setIndex(idx);
  geo.computeVertexNormals();
  const mesh = new THREE.Mesh(geo, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.frustumCulled = false;
  return mesh;
}

// --- textured materials ------------------------------------------------------------------
const TEX_MATS = new Map();
function texturedMat(name, size) {
  const surface = { concrete: "concrete", soil: "packed_soil", plank: "plywood_deck", plywood: "plywood_deck", block: "concrete" }[name];
  if (!surface) {
    if (name === "brick") return brickMat(size);
    return mat(name);
  }
  const key = `${name}|${size.map((x) => x.toFixed(1)).join(",")}`;
  if (TEX_MATS.has(key)) return TEX_MATS.get(key);
  const { tex, tile } = surfaceTexture(surface, "dry");
  const t = tex.clone(); t.needsUpdate = true;
  t.repeat.set(Math.max(size[0], size[2]) / tile, Math.max(size[1], 0.5) / tile);
  const m = new THREE.MeshStandardMaterial({ map: t, roughness: 0.95, color: name === "block" ? 0xcfcac0 : 0xffffff });
  TEX_MATS.set(key, m);
  return m;
}

let BRICK = null;
function brickMat() {
  if (BRICK) return BRICK;
  const c = document.createElement("canvas"); c.width = c.height = 256;
  const g = c.getContext("2d");
  g.fillStyle = "#cfc6b8"; g.fillRect(0, 0, 256, 256);
  for (let row = 0; row < 8; row++) {
    for (let col = -1; col < 5; col++) {
      const x = col * 64 + (row % 2) * 32, y = row * 32;
      const l = 120 + Math.random() * 40;
      g.fillStyle = `rgb(${l + 30},${l * 0.55},${l * 0.42})`;
      g.fillRect(x + 2, y + 2, 60, 28);
    }
  }
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace; t.repeat.set(4, 4);
  BRICK = new THREE.MeshStandardMaterial({ map: t, roughness: 0.95 });
  return BRICK;
}

let ROOF = null;
function roofMat() {
  if (ROOF) return ROOF;
  const c = document.createElement("canvas"); c.width = c.height = 256;
  const g = c.getContext("2d");
  g.fillStyle = "#3a3e44"; g.fillRect(0, 0, 256, 256);
  for (let row = 0; row < 16; row++) for (let col = -1; col < 9; col++) {
    const l = 45 + Math.random() * 25;
    g.fillStyle = `rgb(${l},${l + 3},${l + 8})`;
    g.fillRect(col * 32 + (row % 2) * 16 + 1, row * 16 + 1, 30, 14);
  }
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.colorSpace = THREE.SRGBColorSpace; t.repeat.set(6, 3);
  ROOF = new THREE.MeshStandardMaterial({ map: t, roughness: 0.9 });
  return ROOF;
}

// --- prop meshes -----------------------------------------------------------------------------
// Ladder along local -Z (three.js forward) leaning 1:4, rails from y=0 to `height`.
export function ladderMesh(height, width, material, spacing) {
  const g = new THREE.Group();
  const len = height * Math.sqrt(1 + 1 / 16);
  const angle = Math.atan2(1, 4);
  const inner = new THREE.Group();
  for (const x of [-width / 2, width / 2]) {
    const rail = box([0.04, len, 0.07], material);
    rail.position.set(x, len / 2, 0); inner.add(rail);
  }
  for (let y = spacing; y < height; y += spacing) {
    const rung = cyl(0.016, 0.016, width, material, 8);
    rung.rotation.z = Math.PI / 2;
    rung.position.set(0, y / Math.cos(angle), 0);
    inner.add(rung);
  }
  inner.rotation.x = -angle;
  g.add(inner);
  return g;
}

function stepladderMesh(height) {
  const g = new THREE.Group();
  const front = new THREE.Group();
  const back = new THREE.Group();
  const len = height / Math.cos(Math.atan2(1, 4));
  const fiber = new THREE.MeshStandardMaterial({ color: 0xffb000, roughness: 0.55 });
  for (const x of [-0.24, 0.24]) {
    const rail = box([0.035, len, 0.08], fiber); rail.position.set(x, len / 2, 0); front.add(rail);
    const leg = box([0.03, len, 0.05], fiber); leg.position.set(x * 0.9, -len / 2, 0); back.add(leg);
  }
  for (let y = 0.3; y < height - 0.1; y += 0.3) {
    const step = box([0.46, 0.025, 0.09], mat("aluminium")); step.position.set(0, y / Math.cos(Math.atan2(1, 4)), -0.02); front.add(step);
  }
  const cap = box([0.56, 0.05, 0.22], new THREE.MeshStandardMaterial({ color: 0x333333 })); cap.position.y = len + 0.02; front.add(cap);
  const hinge = new THREE.Group();
  hinge.position.y = len;
  back.position.y = 0;
  hinge.add(back);
  front.add(hinge);
  const roll = new THREE.Group();  // carried on edge (rails one above the other), rolled about its length
  roll.add(front);
  g.add(roll);
  g.userData.setOpen = (open) => {
    const a = Math.atan2(1, 4);
    roll.rotation.z = (1 - open) * Math.PI / 2;
    front.rotation.x = -a * open + (1 - open) * (-Math.PI / 2 + 0.06);
    front.position.set(0, 0, (1 - open) * (len / 2));
    hinge.rotation.x = (2 * a + 0.12) * open + (1 - open) * 0.02;
  };
  g.userData.setOpen(1);
  return g;
}

// Wheel loader; origin on the ground under the seat. Cab floor at `floor`, seat top at `seat`
// (the simulated ladder climb and seat height). setSeat(dy) moves the suspension seat.
function loaderMesh(floor = 0.9, seat = 1.35) {
  const g = new THREE.Group();
  const yellow = mat("yellow");
  const engine = box([1.6, 1.0, 1.4], yellow); engine.position.set(0, 1.05, 1.15); g.add(engine);
  const grille = box([1.2, 0.5, 0.02], mat("black")); grille.position.set(0, 1.05, 1.86); g.add(grille);
  const deck = box([1.7, 0.08, 1.2], mat("steel")); deck.position.set(0, floor - 0.04, -0.25); g.add(deck);
  const frame = box([1.3, 0.45, 3.4], mat("black")); frame.position.set(0, 0.6, 0.1); g.add(frame);
  const seatGroup = new THREE.Group();
  const cushion = box([0.48, 0.12, 0.45], mat("black")); cushion.position.set(0, seat - 0.06, 0.02); seatGroup.add(cushion);
  const back = box([0.48, 0.55, 0.1], mat("black")); back.position.set(0, seat + 0.24, 0.25); seatGroup.add(back);
  g.add(seatGroup);
  const ped = box([0.2, 1, 0.2], mat("steel")); g.add(ped);    // suspension column, stretched to the seat
  g.userData.setSeat = (dy) => {
    seatGroup.position.y = dy;
    const h = Math.max(seat - 0.12 + dy - floor, 0.04);
    ped.scale.y = h; ped.position.set(0, floor + h / 2, 0.05);
  };
  g.userData.setSeat(0);
  const column = cyl(0.04, 0.04, 0.55, "black"); column.position.set(0, 1.2, -0.5); column.rotation.x = 0.4; g.add(column);
  const wheel = new THREE.Mesh(new THREE.TorusGeometry(0.17, 0.02, 8, 24), mat("black")); wheel.position.set(0, 1.47, -0.6); wheel.rotation.x = Math.PI / 2 - 0.4; g.add(wheel);
  for (const [x, z] of [[-0.88, -0.65], [0.88, -0.65], [-0.88, 0.55], [0.88, 0.55]]) {
    const post = box([0.08, 1.75, 0.08], yellow); post.position.set(x, 1.75, z); g.add(post);
  }
  const roof = box([1.9, 0.08, 1.4], yellow); roof.position.set(0, 2.65, -0.05); g.add(roof);
  const glass = box([1.7, 1.2, 0.02], mat("glass")); glass.position.set(0, 1.95, -0.66); g.add(glass);
  // loader arms and bucket in front
  for (const x of [-0.6, 0.6]) {
    const arm = box([0.12, 0.18, 2.2], yellow); arm.position.set(x, 1.0, -1.6); arm.rotation.x = 0.25; g.add(arm);
  }
  const bucket = box([2.2, 0.7, 0.8], mat("steel")); bucket.position.set(0, 0.45, -2.75); g.add(bucket);
  const wheels = [];
  for (const [x, z] of [[-0.95, -0.95], [0.95, -0.95], [-0.95, 1.15], [0.95, 1.15]]) {
    const w = new THREE.Group();
    const tyre = cyl(0.6, 0.6, 0.5, "rubber", 24); tyre.rotation.z = Math.PI / 2; w.add(tyre);
    const rim = cyl(0.32, 0.32, 0.52, "yellow", 16); rim.rotation.z = Math.PI / 2; w.add(rim);
    w.position.set(x, 0.6, z);
    g.add(w); wheels.push(w);
  }
  // access steps where the simulated climb puts the feet (rung line 0.12 m ahead of the start, 1:4)
  for (let k = 1; k <= 2; k++) { const s = box([0.5, 0.04, 0.2], mat("steel")); s.position.set(0, 0.3 * k, 0.105 - 0.075 * k); g.add(s); }
  g.userData.wheels = wheels;
  g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return g;
}

// Wide-flange beam along local X: 150 x 12 mm flanges, 8 mm web, 274 mm deep (scene.BEAM_*).
function beamMesh(length, material) {
  const g = new THREE.Group();
  for (const y of [0.131, -0.131]) { const fl = box([length, 0.012, 0.15], material); fl.position.y = y; g.add(fl); }
  g.add(box([length, 0.25, 0.008], material));
  return g;
}

// Timber crib the beam rests on before the lift: two piles of crossed timbers up to crib.top,
// with spare beams lying behind the lifted one. crib.top = [x, top y, z] at the crib centre.
function cribMesh(crib, yaw) {
  const g = new THREE.Group();
  const top = crib.top[1], ground = crib.ground, [len, depth] = crib.size;
  const layer = 0.15, n = Math.max(Math.round((top - ground - 0.03) / layer), 1);
  const h = (top - ground - 0.03) / n;
  for (const x of [-len / 2 + 0.2, len / 2 - 0.2]) {
    for (let i = 0; i < n; i++) {
      const across = i % 2 === 0;
      const t = box(across ? [0.36, h, 0.14] : [0.14, h, depth], "lumber");
      for (const off of across ? [-depth / 2 + 0.07, depth / 2 - 0.07] : [-0.11, 0.11]) {
        const c = t.clone();
        if (across) c.position.set(x, ground + h * (i + 0.5), off); else c.position.set(x + off, ground + h * (i + 0.5), 0);
        g.add(c);
      }
    }
  }
  // local -Z faces the worker: the lifted beam rested on the front dunnage, spares lie behind it
  const dunnage = box([len, 0.03, 0.08], "lumber"); dunnage.position.set(0, top - 0.015, -depth / 2 + 0.04); g.add(dunnage);
  for (let k = 1; k <= 2; k++) {
    const b = beamMesh(2.6, "red"); b.position.set(0, top + 0.137, -depth / 2 + 0.1 + 0.24 * k); g.add(b);
  }
  g.position.set(crib.top[0], 0, crib.top[2]);
  g.rotation.y = yaw;
  g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return g;
}

// Mobile crane: carrier and turntable on the ground, telescopic boom from boom.foot to boom.tip,
// sheave head at the tip (the rope's pivot).
function mobileCraneMesh(boom) {
  const g = new THREE.Group();
  const foot = new THREE.Vector3(...boom.foot), tip = new THREE.Vector3(...boom.tip);
  const flat = new THREE.Vector3(foot.x - tip.x, 0, foot.z - tip.z).normalize();   // tip -> carrier
  const yaw = Math.atan2(-flat.x, -flat.z);   // carrier local -Z points away from the boom tip
  const carrier = new THREE.Group();
  const part = (mesh, x, y, z) => { mesh.position.set(x, y, z); carrier.add(mesh); return mesh; };
  part(box([2.5, 1.2, 9.0], mat("yellow")), 0, 1.3, -2.5);
  part(box([2.6, 0.5, 9.2], mat("black")), 0, 0.55, -2.5);
  part(box([2.3, 1.4, 1.8], mat("yellow")), 0, 2.4, 0.4);        // turntable under the boom foot
  part(box([1.0, 1.1, 1.4], mat("glass")), -1.1, 2.4, -0.6);      // operator cab
  for (const z of [-6.0, -4.3, -1.0, 0.7]) for (const x of [-1.15, 1.15]) {
    const w = part(cyl(0.55, 0.55, 0.45, "rubber", 18), x, 0.55, z); w.rotation.z = Math.PI / 2;
  }
  for (const z of [-6.8, 1.6]) for (const x of [-1.0, 1.0]) {      // outriggers on pads
    part(box([1.4, 0.2, 0.25], mat("black")), x * 1.6, 0.5, z);
    part(cyl(0.3, 0.3, 0.06, "steel", 12), x * 2.3, 0.03, z);
  }
  carrier.position.set(foot.x, 0, foot.z);
  carrier.rotation.y = yaw;
  g.add(carrier);
  const boomMesh = box([0.55, 1, 0.55], mat("yellow"));
  placeBetween(boomMesh, foot, tip);
  g.add(boomMesh);
  const head = cyl(0.35, 0.35, 0.3, "steel", 16); head.position.copy(tip); head.rotation.z = Math.PI / 2; head.rotation.y = yaw; g.add(head);
  g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return g;
}

function decorMesh(kind, p) {
  const g = new THREE.Group();
  const add = (m, x = 0, y = 0, z = 0) => { m.position.set(x, y, z); g.add(m); return m; };
  switch (kind) {
    case "cone": {
      add(cyl(0.035, 0.17, 0.7, "orange", 20), 0, 0.37);
      add(cyl(0.07, 0.11, 0.12, "white", 20), 0, 0.45);
      add(box([0.38, 0.03, 0.38], "orange"), 0, 0.015);
      break;
    }
    case "barrier": {
      add(box([1.8, 0.8, 0.45], new THREE.MeshStandardMaterial({ color: 0xff6a00, roughness: 0.5 })), 0, 0.4);
      add(box([1.82, 0.18, 0.47], "white"), 0, 0.62);
      break;
    }
    case "pallet": {
      for (const x of [-0.5, 0, 0.5]) add(box([0.1, 0.1, 1.2], "lumber"), x, 0.05);
      for (let z = -0.55; z <= 0.56; z += 0.18) add(box([1.2, 0.025, 0.12], "lumber"), 0, 0.115, z);
      break;
    }
    case "lumber_stack": {
      for (let layer = 0; layer < 5; layer++) for (let k = 0; k < 6; k++) add(box([0.09, 0.045, 3.6], "lumber"), -0.3 + k * 0.12, 0.1 + layer * 0.05);
      add(box([0.09, 0.09, 0.9], "lumber"), 0, 0.045, -1.2); add(box([0.09, 0.09, 0.9], "lumber"), 0, 0.045, 1.2);
      g.children.forEach((c, i) => { if (i >= 30) c.rotation.y = Math.PI / 2; });
      break;
    }
    case "drum": {
      add(cyl(0.29, 0.29, 0.88, new THREE.MeshStandardMaterial({ color: 0x2457a8, roughness: 0.5, metalness: 0.3 }), 24), 0, 0.44);
      break;
    }
    case "pipes": {
      for (let i = 0; i < 6; i++) {
        const row = i < 3 ? 0 : 1;
        const m = cyl(0.11, 0.11, 4.0, new THREE.MeshStandardMaterial({ color: 0x5a6068, metalness: 0.6, roughness: 0.5 }), 14);
        m.rotation.z = Math.PI / 2;
        add(m, 0, 0.11 + row * 0.19, -0.24 + (i % 3) * 0.23 + row * 0.11);
      }
      break;
    }
    case "block_stack": {
      for (let i = 0; i < 3; i++) for (let j = 0; j < 2; j++) for (let k = 0; k < 3; k++) add(box([0.39, 0.19, 0.19], "block"), -0.2 + j * 0.4, 0.22 + k * 0.2, -0.4 + i * 0.4);
      add(box([1.0, 0.12, 1.2], "lumber"), 0, 0.06);
      break;
    }
    case "wheelbarrow": {
      add(box([0.6, 0.3, 0.8], new THREE.MeshStandardMaterial({ color: 0x2e7d32, roughness: 0.5 })), 0, 0.5);
      const w = add(cyl(0.18, 0.18, 0.08, "rubber", 16), 0, 0.18, -0.5); w.rotation.z = Math.PI / 2;
      add(box([0.04, 0.04, 1.2], "steel"), -0.25, 0.4, 0.3); add(box([0.04, 0.04, 1.2], "steel"), 0.25, 0.4, 0.3);
      break;
    }
    case "trailer": {
      add(box([7.0, 2.6, 2.6], new THREE.MeshStandardMaterial({ color: 0xe6e3dc, roughness: 0.8 })), 0, 1.6);
      for (const x of [-2.2, 0, 2.2]) add(box([1.0, 0.8, 0.05], mat("glass")), x, 1.9, 1.31);
      add(box([0.9, 2.0, 0.05], mat("blue")), 3.0, 1.3, 1.31);
      add(box([6.6, 0.3, 2.4], "black"), 0, 0.15);
      break;
    }
    case "toilet": {
      add(box([1.1, 2.3, 1.1], new THREE.MeshStandardMaterial({ color: 0x1f5fbf, roughness: 0.6 })), 0, 1.15);
      add(box([1.15, 0.12, 1.15], "white"), 0, 2.35);
      break;
    }
    case "container": {
      add(box([6.1, 2.6, 2.44], new THREE.MeshStandardMaterial({ color: 0x8c2f22, roughness: 0.7, metalness: 0.3 })), 0, 1.3);
      for (let x = -2.9; x <= 2.9; x += 0.3) add(box([0.06, 2.5, 2.46], new THREE.MeshStandardMaterial({ color: 0x7a2a1f, roughness: 0.7 })), x, 1.3);
      break;
    }
    case "tower_crane": {
      const y = new THREE.MeshStandardMaterial({ color: 0xf2b705, roughness: 0.6, metalness: 0.2 });
      add(box([1.8, 42, 1.8], new THREE.MeshStandardMaterial({ color: 0xf2b705, roughness: 0.6, wireframe: true })), 0, 21);
      add(box([1.6, 42, 1.6], y), 0, 21).scale.set(0.35, 1, 0.35);
      add(box([1.4, 1.2, 50], y), 0, 43, -12);
      add(box([2.4, 2.0, 2.4], y), 0, 44.5);
      add(box([2.5, 2.5, 4], mat("concrete")), 0, 42.5, 10);
      add(new THREE.Mesh(new THREE.BoxGeometry(0.05, 14, 0.05), mat("black")), 0, 36, -28);
      break;
    }
    case "spoil": {
      const m = add(new THREE.Mesh(new THREE.SphereGeometry(1.6, 18, 12, 0, Math.PI * 2, 0, Math.PI / 2), mat("soil")), 0, 0);
      m.scale.set(1.6, 0.6, 1); m.castShadow = true; m.receiveShadow = true;
      break;
    }
    case "rubble": {
      const s = p.s || 0.12;
      const m = add(new THREE.Mesh(new THREE.DodecahedronGeometry(s, 0), mat("concrete")), 0, s * 0.6);
      m.rotation.set(Math.random() * 3, Math.random() * 3, 0); m.castShadow = true;
      break;
    }
    case "toolbox": {
      add(box([0.55, 0.25, 0.25], new THREE.MeshStandardMaterial({ color: 0xc62828, roughness: 0.4, metalness: 0.3 })), 0, 0.125);
      add(box([0.3, 0.03, 0.03], "black"), 0, 0.27);
      break;
    }
    case "bucket": {
      add(cyl(0.15, 0.13, 0.36, "bucket", 20), 0, 0.18);
      break;
    }
    case "cooler": {
      add(box([0.5, 0.35, 0.32], new THREE.MeshStandardMaterial({ color: 0x1e88e5, roughness: 0.5 })), 0, 0.175);
      add(box([0.52, 0.06, 0.34], "white"), 0, 0.37);
      break;
    }
    case "rebar_bundle": {
      for (let i = 0; i < 12; i++) { const m = cyl(0.01, 0.01, 6, "rebar", 6); m.rotation.x = Math.PI / 2; add(m, (i % 4) * 0.03, 0.12 + Math.floor(i / 4) * 0.03); }
      add(box([0.1, 0.1, 0.4], "lumber"), 0, 0.05, -2); add(box([0.1, 0.1, 0.4], "lumber"), 0, 0.05, 2);
      g.children.slice(-2).forEach((c) => (c.rotation.y = Math.PI / 2));
      break;
    }
    case "vehicle_side": {
      const v = loaderMesh(); v.rotation.y = Math.PI / 2; g.add(v);
      break;
    }
    default:
      return null;
  }
  g.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  return g;
}
