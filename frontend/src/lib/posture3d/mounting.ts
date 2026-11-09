// Per-sensor mounting correction for the spine view, plus the solver the calibration wizard uses.
//
// Each IMU runs its own 6-axis fusion, so its "world" shares the vertical with the others but has its
// own arbitrary heading (and, if a board or firmware axis convention is off, may even be mirrored).
// A forward bend therefore shows up about a different horizontal axis for every sensor.
//
// Model, per sensor:
//   D    = q · q0⁻¹                      rotation since the captured neutral, in the sensor's world
//   C    = R(yaw, pitch, roll) · P        P: signed axis remap (may be a mirror), R: rotational offset
//   body = C · D · Cᵀ                     the same rotation expressed in the wearer frame
// Wearer frame: +X right, +Y up, +Z posterior. A forward bend is a negative rotation about +X, a bend to
// the right is a negative rotation about +Z. The board's mounting on the skin cancels out of D, so only
// the frame offset C (mostly heading) needs to be known.

import * as THREE from "three";
import { REGIONS, type Region } from "../../telemetry/types";

export interface Mounting {
  remap: string; // body axes from sensor axes, e.g. "x,y,z" or "-y,z,-x"
  yaw: number; // degrees about +Y (up)
  pitch: number; // degrees about +X (right)
  roll: number; // degrees about +Z (posterior)
}

export type MountingSet = Record<Region, Mounting>;

export const IDENTITY_MOUNTING: Mounting = { remap: "x,y,z", yaw: 0, pitch: 0, roll: 0 };

export function defaultMountings(): MountingSet {
  return Object.fromEntries(REGIONS.map((r) => [r, { ...IDENTITY_MOUNTING }])) as MountingSet;
}

// All 48 signed axis permutations, identity first.
export const REMAP_OPTIONS: string[] = (() => {
  const perms = [
    ["x", "y", "z"], ["x", "z", "y"], ["y", "x", "z"], ["y", "z", "x"], ["z", "x", "y"], ["z", "y", "x"],
  ];
  const out: string[] = [];
  for (const perm of perms) {
    for (let signs = 0; signs < 8; signs += 1) {
      out.push(perm.map((axis, i) => ((signs >> i) & 1 ? `-${axis}` : axis)).join(","));
    }
  }
  return out;
})();

const AXIS_INDEX: Record<string, number> = { x: 0, y: 1, z: 2 };

/** P with body[i] = sign · sensor[axis]. Returns null for a malformed remap. */
export function remapMatrix(remap: string): THREE.Matrix3 | null {
  const parts = remap.split(",").map((part) => part.trim());
  if (parts.length !== 3) {
    return null;
  }
  const e = new Array(9).fill(0);
  const used = new Set<number>();
  for (let row = 0; row < 3; row += 1) {
    const negative = parts[row].startsWith("-");
    const axis = AXIS_INDEX[parts[row].replace("-", "")];
    if (axis === undefined || used.has(axis)) {
      return null;
    }
    used.add(axis);
    e[row * 3 + axis] = negative ? -1 : 1;
  }
  return new THREE.Matrix3().set(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8]);
}

function offsetMatrix(m: Mounting): THREE.Matrix3 {
  const d = THREE.MathUtils.degToRad;
  const q = new THREE.Quaternion()
    .setFromAxisAngle(new THREE.Vector3(0, 1, 0), d(m.yaw))
    .multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), d(m.pitch)))
    .multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), d(m.roll)));
  return new THREE.Matrix3().setFromMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(q));
}

/** C = R(yaw, pitch, roll) · P */
export function frameMatrix(m: Mounting): THREE.Matrix3 {
  const p = remapMatrix(m.remap) ?? new THREE.Matrix3();
  return offsetMatrix(m).multiply(p);
}

function quatToMatrix3(q: THREE.Quaternion): THREE.Matrix3 {
  return new THREE.Matrix3().setFromMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(q));
}

function matrix3ToQuat(m: THREE.Matrix3): THREE.Quaternion {
  const e = m.elements; // column-major
  const m4 = new THREE.Matrix4().set(e[0], e[3], e[6], 0, e[1], e[4], e[7], 0, e[2], e[5], e[8], 0, 0, 0, 0, 1);
  return new THREE.Quaternion().setFromRotationMatrix(m4).normalize();
}

/** body = C · D · Cᵀ. C is orthogonal, so this is a proper rotation even when P is a mirror. */
export function toBodyFrame(delta: THREE.Quaternion, mounting: Mounting): THREE.Quaternion {
  const c = frameMatrix(mounting);
  const ct = c.clone().transpose();
  return matrix3ToQuat(c.clone().multiply(quatToMatrix3(delta)).multiply(ct));
}

/** Rotation vector (axis · angle, radians), shorter path. */
export function rotationVector(q: THREE.Quaternion): THREE.Vector3 {
  const sign = q.w < 0 ? -1 : 1;
  const v = new THREE.Vector3(q.x * sign, q.y * sign, q.z * sign);
  const s = v.length();
  if (s < 1e-9) {
    return v.multiplyScalar(2);
  }
  return v.multiplyScalar((2 * Math.atan2(s, q.w * sign)) / s);
}

/** Mean of nearby unit quaternions (sign-aligned to the first), for averaging a held pose. */
export function averageQuaternion(samples: THREE.Quaternion[]): THREE.Quaternion | null {
  if (samples.length === 0) {
    return null;
  }
  const ref = samples[0];
  const sum = new THREE.Vector4();
  for (const q of samples) {
    const s = q.x * ref.x + q.y * ref.y + q.z * ref.z + q.w * ref.w < 0 ? -1 : 1;
    sum.x += s * q.x;
    sum.y += s * q.y;
    sum.z += s * q.z;
    sum.w += s * q.w;
  }
  return new THREE.Quaternion(sum.x, sum.y, sum.z, sum.w).normalize();
}

// ------------------------------------------------------------------ calibration solver

export type MovementName = "forward" | "backward" | "right" | "left";

// Expected rotation axis of each movement in the wearer frame.
export const MOVEMENT_AXES: Record<MovementName, THREE.Vector3> = {
  forward: new THREE.Vector3(-1, 0, 0),
  backward: new THREE.Vector3(1, 0, 0),
  right: new THREE.Vector3(0, 0, -1),
  left: new THREE.Vector3(0, 0, 1),
};

export const MIN_BEND_DEG = 8; // smaller movements are too noisy to fix a heading

export interface MovementFit {
  movement: MovementName;
  bendDeg: number; // how far this sensor rotated
  errorDeg: number | null; // angle between the corrected axis and the expected one (null: too small)
}

export interface SensorFit {
  mounting: Mounting | null; // null: not enough usable movement, keep the old one
  mirrored: boolean;
  movements: MovementFit[];
  meanErrorDeg: number | null;
}

/**
 * Heading (and mirror) that best maps the measured bend axes onto the expected ones.
 * deltas: q_step · q0⁻¹ for each held movement, in the sensor's world frame.
 * Least squares over yaw: maximise Σ (R_y(ψ) v)·t → ψ = atan2(Σ (v_z t_x − v_x t_z), Σ (v_x t_x + v_z t_z)).
 */
export function solveMounting(deltas: Partial<Record<MovementName, THREE.Quaternion>>): SensorFit {
  const used = (Object.keys(deltas) as MovementName[])
    .map((movement) => ({ movement, v: rotationVector(deltas[movement]!) }))
    .filter(({ v }) => THREE.MathUtils.radToDeg(v.length()) >= MIN_BEND_DEG);

  const candidates = [
    { remap: "x,y,z", mirrored: false, axial: (v: THREE.Vector3) => v.clone() },
    // Mirror P = diag(−1, 1, 1): a rotation vector is axial, so it maps to −P·v = (x, −y, −z).
    { remap: "-x,y,z", mirrored: true, axial: (v: THREE.Vector3) => new THREE.Vector3(v.x, -v.y, -v.z) },
  ];

  let best: { remap: string; mirrored: boolean; yaw: number; score: number } | null = null;
  if (used.length > 0) {
    for (const candidate of candidates) {
      let sin = 0;
      let cos = 0;
      for (const { movement, v } of used) {
        const w = candidate.axial(v);
        const t = MOVEMENT_AXES[movement];
        sin += w.z * t.x - w.x * t.z;
        cos += w.x * t.x + w.z * t.z;
      }
      const yaw = Math.atan2(sin, cos);
      const score = Math.hypot(sin, cos);
      if (!best || score > best.score + 1e-9) {
        best = { remap: candidate.remap, mirrored: candidate.mirrored, yaw, score };
      }
    }
  }

  const mounting = best
    ? { remap: best.remap, yaw: Math.round(THREE.MathUtils.radToDeg(best.yaw) * 10) / 10, pitch: 0, roll: 0 }
    : null;

  const movements: MovementFit[] = (Object.keys(deltas) as MovementName[]).map((movement) => {
    const delta = deltas[movement]!;
    const bendDeg = THREE.MathUtils.radToDeg(rotationVector(delta).length());
    if (!mounting || bendDeg < MIN_BEND_DEG) {
      return { movement, bendDeg, errorDeg: null };
    }
    const axis = rotationVector(toBodyFrame(delta, mounting)).normalize();
    const errorDeg = THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(axis.dot(MOVEMENT_AXES[movement]), -1, 1)));
    return { movement, bendDeg, errorDeg };
  });
  const errors = movements.map((m) => m.errorDeg).filter((e): e is number => e !== null);
  return {
    mounting,
    mirrored: best?.mirrored ?? false,
    movements,
    meanErrorDeg: errors.length ? errors.reduce((a, b) => a + b, 0) / errors.length : null,
  };
}

// ------------------------------------------------------------------ persistence (per browser)

const STORAGE_KEY = "spine.calibration.v1";

export interface StoredCalibration {
  neutral: Partial<Record<Region, [number, number, number, number]>>; // x, y, z, w
  mountings: MountingSet;
}

export function loadCalibration(): StoredCalibration | null {
  try {
    const text = window.localStorage.getItem(STORAGE_KEY);
    if (!text) {
      return null;
    }
    const parsed = JSON.parse(text) as StoredCalibration;
    const mountings = defaultMountings();
    for (const region of REGIONS) {
      const m = parsed.mountings?.[region];
      if (m && remapMatrix(m.remap) && [m.yaw, m.pitch, m.roll].every(Number.isFinite)) {
        mountings[region] = { remap: m.remap, yaw: m.yaw, pitch: m.pitch, roll: m.roll };
      }
    }
    return { neutral: parsed.neutral ?? {}, mountings };
  } catch {
    return null;
  }
}

export function saveCalibration(value: StoredCalibration): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
  } catch {
    // Storage blocked (private window): calibration still works for this session.
  }
}
