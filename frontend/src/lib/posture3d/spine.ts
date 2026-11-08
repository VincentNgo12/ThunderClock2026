// Spine pose math: turns four measured segment orientations into a chain of vertebra
// transforms. Vertebrae between two sensors are visual interpolation (a smooth spline), not
// measurements. Frame: +X wearer-right, +Y up, +Z posterior (same as three.js y-up).

import * as THREE from "three";
import type { Region } from "../../telemetry/types";

export interface VertebraSpec {
  name: string;
  height: number; // metres along the local up axis, body plus the disc above it
  width: number; // body width (X), metres
  depth: number; // body depth (Z), metres
  restTiltDeg: number; // neutral sagittal tilt; positive leans the top posterior (+Z)
}

// Bottom to top. Rest tilts give a mild lumbar lordosis and thoracic kyphosis.
export const VERTEBRAE: VertebraSpec[] = [
  { name: "Sacrum", height: 0.1, width: 0.075, depth: 0.04, restTiltDeg: -32 },
  { name: "L5", height: 0.036, width: 0.05, depth: 0.035, restTiltDeg: -12 },
  { name: "L4", height: 0.036, width: 0.05, depth: 0.035, restTiltDeg: -6 },
  { name: "L3", height: 0.035, width: 0.048, depth: 0.034, restTiltDeg: 0 },
  { name: "L2", height: 0.034, width: 0.046, depth: 0.033, restTiltDeg: 5 },
  { name: "L1", height: 0.033, width: 0.044, depth: 0.032, restTiltDeg: 9 },
  { name: "T12", height: 0.03, width: 0.04, depth: 0.03, restTiltDeg: 10 },
  { name: "T11", height: 0.029, width: 0.038, depth: 0.029, restTiltDeg: 10 },
  { name: "T10", height: 0.028, width: 0.036, depth: 0.028, restTiltDeg: 9 },
  { name: "T9", height: 0.027, width: 0.034, depth: 0.027, restTiltDeg: 7 },
  { name: "T8", height: 0.026, width: 0.033, depth: 0.026, restTiltDeg: 5 },
  { name: "T7", height: 0.025, width: 0.032, depth: 0.025, restTiltDeg: 3 },
  { name: "T6", height: 0.024, width: 0.031, depth: 0.024, restTiltDeg: 1 },
  { name: "T5", height: 0.023, width: 0.03, depth: 0.023, restTiltDeg: -2 },
  { name: "T4", height: 0.022, width: 0.029, depth: 0.022, restTiltDeg: -5 },
  { name: "T3", height: 0.021, width: 0.028, depth: 0.021, restTiltDeg: -8 },
  { name: "T2", height: 0.02, width: 0.027, depth: 0.02, restTiltDeg: -11 },
  { name: "T1", height: 0.02, width: 0.027, depth: 0.02, restTiltDeg: -14 },
];

// Index into VERTEBRAE where each sensor sits. Confirm against the physical mounting.
export const SENSOR_ANCHORS: { region: Region; index: number }[] = [
  { region: "pelvis", index: 0 }, // sacrum
  { region: "lumbar", index: 3 }, // L3
  { region: "thoracic", index: 10 }, // T8
  { region: "upper_thoracic", index: 16 }, // T2
];

const UP = new THREE.Vector3(0, 1, 0);
const X_AXIS = new THREE.Vector3(1, 0, 0);

const REST_ROTATIONS = VERTEBRAE.map((v) =>
  new THREE.Quaternion().setFromAxisAngle(X_AXIS, THREE.MathUtils.degToRad(v.restTiltDeg)),
);

export interface VertebraPose {
  position: THREE.Vector3; // base of the vertebra
  rotation: THREE.Quaternion;
}

// "spline": cubic Hermite through the sensors, so the bend changes smoothly along the column.
// "linear": per-segment slerp, which keeps the bend uniform between two sensors and kinks at each one.
export type InterpolationMode = "spline" | "linear";

// Rotation vector (axis × angle, radians) of a unit quaternion, taking the shorter path.
function quatLog(q: THREE.Quaternion): THREE.Vector3 {
  const sign = q.w < 0 ? -1 : 1;
  const v = new THREE.Vector3(q.x * sign, q.y * sign, q.z * sign);
  const sinHalf = v.length();
  if (sinHalf < 1e-9) {
    return v.multiplyScalar(2);
  }
  const angle = 2 * Math.atan2(sinHalf, q.w * sign);
  return v.multiplyScalar(angle / sinHalf);
}

function quatExp(v: THREE.Vector3): THREE.Quaternion {
  const angle = v.length();
  if (angle < 1e-9) {
    return new THREE.Quaternion(v.x / 2, v.y / 2, v.z / 2, 1).normalize();
  }
  return new THREE.Quaternion().setFromAxisAngle(v.clone().divideScalar(angle), angle);
}

/**
 * Segment rotation for every vertebra. Sensors are knots at their vertebra index. Each span
 * is a cubic Hermite curve in rotation-vector space, relative to its lower sensor:
 *   q(t) = exp(h10·h·ω_k + h01·Δ_k + h11·h·ω_{k+1}) · q_k
 * Here Δ_k = log(q_{k+1} q_k⁻¹) and ω_k is the bend rate per vertebra at sensor k, taken from
 * a non-uniform Catmull–Rom estimate inside the column and natural-spline ends. The curve passes
 * exactly through every sensor. The bend rate is continuous across sensors. Vertebrae outside
 * the end sensors copy the nearest sensor.
 */
function segmentRotations(anchors: THREE.Quaternion[], mode: InterpolationMode): THREE.Quaternion[] {
  const knots = SENSOR_ANCHORS.map((anchor) => anchor.index);
  const spans = knots.length - 1;
  const deltas: THREE.Vector3[] = [];
  const widths: number[] = [];
  for (let k = 0; k < spans; k += 1) {
    deltas.push(quatLog(anchors[k + 1].clone().multiply(anchors[k].clone().invert())));
    widths.push(knots[k + 1] - knots[k]);
  }
  const slopes = deltas.map((delta, k) => delta.clone().divideScalar(widths[k]));

  const rates: THREE.Vector3[] = new Array(knots.length);
  for (let k = 1; k < spans; k += 1) {
    const before = widths[k - 1];
    const after = widths[k];
    rates[k] = slopes[k - 1].clone().multiplyScalar(after).addScaledVector(slopes[k], before).divideScalar(before + after);
  }
  if (spans === 1) {
    rates[0] = slopes[0].clone();
    rates[1] = slopes[0].clone();
  } else {
    rates[0] = slopes[0].clone().multiplyScalar(3).sub(rates[1]).multiplyScalar(0.5);
    rates[spans] = slopes[spans - 1].clone().multiplyScalar(3).sub(rates[spans - 1]).multiplyScalar(0.5);
  }

  return VERTEBRAE.map((_, index) => {
    if (index <= knots[0]) {
      return anchors[0].clone();
    }
    if (index >= knots[spans]) {
      return anchors[spans].clone();
    }
    const k = knots.findIndex((knot, i) => i < spans && index >= knot && index <= knots[i + 1]);
    const width = widths[k];
    const t = (index - knots[k]) / width;
    if (mode === "linear") {
      return anchors[k].clone().slerp(anchors[k + 1], t);
    }
    const t2 = t * t;
    const t3 = t2 * t;
    const v = rates[k].clone().multiplyScalar((t3 - 2 * t2 + t) * width)
      .addScaledVector(deltas[k], -2 * t3 + 3 * t2)
      .addScaledVector(rates[k + 1], (t3 - t2) * width);
    return quatExp(v).multiply(anchors[k]);
  });
}

/**
 * anchors: one segment rotation per SENSOR_ANCHORS entry, in the same order.
 * Returns a base position and world rotation per vertebra, with the sacrum base at the origin.
 */
export function computeSpinePose(anchors: THREE.Quaternion[], mode: InterpolationMode = "spline"): VertebraPose[] {
  const poses: VertebraPose[] = [];
  const cursor = new THREE.Vector3(0, 0, 0);
  const segments = segmentRotations(anchors, mode);
  VERTEBRAE.forEach((spec, index) => {
    const rotation = segments[index].multiply(REST_ROTATIONS[index]);
    poses.push({ position: cursor.clone(), rotation });
    cursor.add(UP.clone().applyQuaternion(rotation).multiplyScalar(spec.height));
  });
  return poses;
}

// Vertical midpoint of the neutral spine, used to keep the column centred on screen.
export function neutralCenter(): THREE.Vector3 {
  const identity = SENSOR_ANCHORS.map(() => new THREE.Quaternion());
  const poses = computeSpinePose(identity);
  const top = poses[poses.length - 1];
  const topEnd = top.position
    .clone()
    .add(UP.clone().applyQuaternion(top.rotation).multiplyScalar(VERTEBRAE[VERTEBRAE.length - 1].height));
  return topEnd.multiplyScalar(0.5);
}
