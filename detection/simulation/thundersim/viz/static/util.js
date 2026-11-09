// File: util.js
// Purpose: Shared helpers for the ThunderSim 3D viewer: colours, procedural textures,
//          frame interpolation and small three.js utilities.
import * as THREE from "three";

export const SLOT_NAMES = ["pelvis", "lumbar", "thoracic", "upper thoracic"];
export const SLOT_COLORS = ["#3d8bfd", "#3ccf7a", "#ff9f1c", "#ff4d4d"];
export const PHASE_COLORS = {
  activity: "#4a5563", calibration_stand: "#5b6f8f", fall_descent: "#ff8a1f", impact: "#ff3b3b",
  lying: "#a12a2a", suspended: "#9b5cff", recovery: "#3ccf7a", near_fall: "#ffd23f", lying_work: "#2bb5a8",
  device_off: "#2c3138", free_fall: "#ff5a1f", collapse: "#c0392b",
};
export const UP = new THREE.Vector3(0, 1, 0);

const textureCache = new Map();

function noiseCanvas(size, draw) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d");
  draw(g, size);
  return c;
}

function speckle(g, size, base, spread, count, alpha = 1, rmax = 1.6) {
  g.fillStyle = base;
  g.fillRect(0, 0, size, size);
  for (let i = 0; i < count; i++) {
    const v = (Math.random() - 0.5) * spread;
    const l = Math.max(0, Math.min(255, 128 + v));
    g.fillStyle = `rgba(${l},${l},${l},${0.06 * alpha})`;
    const r = Math.random() * rmax + 0.3;
    g.fillRect(Math.random() * size, Math.random() * size, r, r);
  }
}

// Ground textures per simulator surface. One texture tile covers `tile` metres.
export function surfaceTexture(surface, condition) {
  const key = `${surface}|${condition}`;
  if (textureCache.has(key)) return textureCache.get(key);
  const size = 512;
  let tile = 4;
  const canvas = noiseCanvas(size, (g, s) => {
    if (surface === "plywood_deck" || surface === "scaffold_plank") {
      tile = 2.44;
      g.fillStyle = "#b8925a"; g.fillRect(0, 0, s, s);
      for (let i = 0; i < 260; i++) {
        g.strokeStyle = `rgba(90,60,25,${0.05 + Math.random() * 0.08})`;
        g.lineWidth = 1 + Math.random() * 2;
        const y = Math.random() * s;
        g.beginPath(); g.moveTo(0, y);
        for (let x = 0; x <= s; x += 32) g.lineTo(x, y + Math.sin(x / 40 + i) * 3);
        g.stroke();
      }
      g.strokeStyle = "rgba(60,40,15,.7)"; g.lineWidth = 3;
      g.strokeRect(0, 0, s, s / 2); g.strokeRect(0, s / 2, s, s / 2);
    } else if (surface === "packed_soil" || surface === "mud") {
      tile = 3;
      speckle(g, s, surface === "mud" ? "#4a3626" : "#7b6046", 260, 26000, 1.5, 2.2);
      for (let i = 0; i < 40; i++) {
        g.fillStyle = `rgba(40,28,18,${0.08 + Math.random() * 0.1})`;
        g.beginPath(); g.ellipse(Math.random() * s, Math.random() * s, 20 + Math.random() * 50, 10 + Math.random() * 30, Math.random() * 3, 0, 7); g.fill();
      }
    } else if (surface === "gravel" || surface === "rebar_mat") {
      tile = 2;
      g.fillStyle = "#7d7a74"; g.fillRect(0, 0, s, s);
      for (let i = 0; i < 9000; i++) {
        const l = 80 + Math.random() * 120;
        g.fillStyle = `rgb(${l},${l * 0.97},${l * 0.92})`;
        g.beginPath(); g.ellipse(Math.random() * s, Math.random() * s, 1.5 + Math.random() * 3.5, 1 + Math.random() * 2.5, Math.random() * 3, 0, 7); g.fill();
      }
    } else if (surface === "crash_mat") {
      tile = 2;
      g.fillStyle = "#2f5fa8"; g.fillRect(0, 0, s, s);
      speckle(g, s, "rgba(0,0,0,0)", 80, 6000, 0.6);
      g.strokeStyle = "rgba(15,30,70,.9)"; g.lineWidth = 6; g.strokeRect(3, 3, s - 6, s - 6);
    } else if (surface === "steel_deck") {
      tile = 1.2;
      g.fillStyle = "#6f777f"; g.fillRect(0, 0, s, s);
      for (let x = 0; x < s; x += 32) { g.fillStyle = "rgba(255,255,255,.08)"; g.fillRect(x, 0, 10, s); g.fillStyle = "rgba(0,0,0,.18)"; g.fillRect(x + 12, 0, 6, s); }
    } else {
      tile = 6;
      speckle(g, s, "#8e8c87", 200, 30000, 1.2);
      g.strokeStyle = "rgba(40,40,40,.55)"; g.lineWidth = 2;
      g.strokeRect(0, 0, s, s);
      for (let i = 0; i < 6; i++) {
        g.strokeStyle = `rgba(50,50,50,${0.12 + Math.random() * 0.15})`; g.lineWidth = 1;
        g.beginPath(); let x = Math.random() * s, y = Math.random() * s; g.moveTo(x, y);
        for (let k = 0; k < 8; k++) { x += (Math.random() - 0.5) * 60; y += (Math.random() - 0.5) * 60; g.lineTo(x, y); }
        g.stroke();
      }
    }
    if (condition === "wet" || condition === "icy" || condition === "oily" || condition === "muddy") {
      const tint = { wet: "rgba(20,30,40,.28)", icy: "rgba(200,225,255,.35)", oily: "rgba(20,15,10,.25)", muddy: "rgba(70,45,25,.35)" }[condition];
      g.fillStyle = tint; g.fillRect(0, 0, s, s);
    } else if (condition === "dusty") {
      g.fillStyle = "rgba(210,190,150,.18)"; g.fillRect(0, 0, s, s);
    }
  });
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  const out = { tex, tile, roughness: { wet: 0.35, icy: 0.15, oily: 0.3 }[condition] ?? 0.95 };
  textureCache.set(key, out);
  return out;
}

export function radialTexture(inner, outer) {
  const key = `radial|${inner}|${outer}`;
  if (textureCache.has(key)) return textureCache.get(key);
  const canvas = noiseCanvas(256, (g, s) => {
    const grad = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    grad.addColorStop(0, inner); grad.addColorStop(0.65, inner); grad.addColorStop(1, outer);
    g.fillStyle = grad; g.fillRect(0, 0, s, s);
    for (let i = 0; i < 18; i++) {
      g.fillStyle = "rgba(0,0,0,0)";
    }
  });
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  textureCache.set(key, tex);
  return tex;
}

export function crackTexture() {
  if (textureCache.has("crack")) return textureCache.get("crack");
  const canvas = noiseCanvas(256, (g, s) => {
    g.clearRect(0, 0, s, s);
    const grad = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    grad.addColorStop(0, "rgba(90,88,84,.95)"); grad.addColorStop(0.75, "rgba(110,108,104,.7)"); grad.addColorStop(1, "rgba(110,108,104,0)");
    g.fillStyle = grad; g.fillRect(0, 0, s, s);
    g.strokeStyle = "rgba(25,25,25,.85)";
    for (let i = 0; i < 14; i++) {
      g.lineWidth = 1 + Math.random() * 2;
      g.beginPath(); let x = s / 2, y = s / 2; g.moveTo(x, y);
      const a = Math.random() * Math.PI * 2;
      for (let k = 0; k < 7; k++) { x += Math.cos(a + (Math.random() - 0.5)) * 14; y += Math.sin(a + (Math.random() - 0.5)) * 14; g.lineTo(x, y); }
      g.stroke();
    }
  });
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  textureCache.set("crack", tex);
  return tex;
}

export function gridTexture(cells = 16, line = "rgba(255,255,255,.9)", width = 2) {
  const key = `grid|${cells}|${line}|${width}`;
  if (textureCache.has(key)) return textureCache.get(key);
  const canvas = noiseCanvas(256, (g, s) => {
    g.clearRect(0, 0, s, s);
    g.strokeStyle = line; g.lineWidth = width;
    const step = s / cells;
    for (let i = 0; i <= cells; i++) {
      g.beginPath(); g.moveTo(i * step, 0); g.lineTo(i * step, s); g.stroke();
      g.beginPath(); g.moveTo(0, i * step); g.lineTo(s, i * step); g.stroke();
    }
  });
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  textureCache.set(key, tex);
  return tex;
}

export function chainLinkTexture() {
  if (textureCache.has("chain")) return textureCache.get("chain");
  const canvas = noiseCanvas(128, (g, s) => {
    g.clearRect(0, 0, s, s);
    g.strokeStyle = "rgba(190,195,200,.9)"; g.lineWidth = 2;
    for (let i = -s; i < 2 * s; i += 16) {
      g.beginPath(); g.moveTo(i, 0); g.lineTo(i + s, s); g.stroke();
      g.beginPath(); g.moveTo(i + s, 0); g.lineTo(i, s); g.stroke();
    }
  });
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  textureCache.set("chain", tex);
  return tex;
}

// --- frame interpolation ---------------------------------------------------------------
// Flat arrays hold `stride` numbers per frame. Returns values interpolated at fractional frame f.
export function lerpFrame(arr, stride, f, out) {
  const n = arr.length / stride;
  const i0 = Math.max(0, Math.min(n - 1, Math.floor(f)));
  const i1 = Math.min(n - 1, i0 + 1);
  const a = Math.max(0, Math.min(1, f - i0));
  const b0 = i0 * stride, b1 = i1 * stride;
  for (let k = 0; k < stride; k++) out[k] = arr[b0 + k] + (arr[b1 + k] - arr[b0 + k]) * a;
  return out;
}

// Quaternions packed as xyzw, `count` per frame; nlerp with hemisphere fix.
export function quatFrame(arr, count, f, index, q) {
  const stride = count * 4;
  const n = arr.length / stride;
  const i0 = Math.max(0, Math.min(n - 1, Math.floor(f)));
  const i1 = Math.min(n - 1, i0 + 1);
  const a = Math.max(0, Math.min(1, f - i0));
  const b0 = i0 * stride + index * 4, b1 = i1 * stride + index * 4;
  let dot = 0;
  for (let k = 0; k < 4; k++) dot += arr[b0 + k] * arr[b1 + k];
  const s = dot < 0 ? -1 : 1;
  q.set(arr[b0] + (s * arr[b1] - arr[b0]) * a, arr[b0 + 1] + (s * arr[b1 + 1] - arr[b0 + 1]) * a,
    arr[b0 + 2] + (s * arr[b1 + 2] - arr[b0 + 2]) * a, arr[b0 + 3] + (s * arr[b1 + 3] - arr[b0 + 3]) * a).normalize();
  return q;
}

export function vec(arr, i) { return new THREE.Vector3(arr[i], arr[i + 1], arr[i + 2]); }

// Cylinder mesh oriented between two points (geometry along +Y, unit length, scaled).
const _y = new THREE.Vector3(0, 1, 0);
const _d = new THREE.Vector3();
export function placeBetween(mesh, a, b, scaleLength = true) {
  _d.subVectors(b, a);
  const len = _d.length();
  mesh.position.copy(a).addScaledVector(_d, 0.5);
  if (len > 1e-6) mesh.quaternion.setFromUnitVectors(_y, _d.multiplyScalar(1 / len));
  if (scaleLength) mesh.scale.set(1, len, 1);
}

export function fmt(x, digits = 2) {
  return (x === null || x === undefined || Number.isNaN(x)) ? "—" : Number(x).toFixed(digits);
}

export function smoothstep(x) { x = Math.max(0, Math.min(1, x)); return x * x * (3 - 2 * x); }
