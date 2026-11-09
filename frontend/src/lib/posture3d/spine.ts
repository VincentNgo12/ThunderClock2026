// Spine pose math: turns four measured segment orientations into a chain of vertebra
// transforms. Vertebrae between two sensors are visual interpolation (a smooth spline), not
// measurements. Frame: +X wearer-right, +Y up, +Z posterior (same as three.js y-up).

import * as THREE from "three";
import type { Region } from "../../telemetry/types";

import column from "./vertebrae.json";

export interface VertebraSpec {
  name: string;
  height: number; // metres along the local up axis: pivot (disc below) to the next pivot
  width: number; // bone width (X), metres
  depth: number; // bone depth (Z), metres
  restTiltDeg: number; // neutral sagittal tilt; positive leans the top posterior (+Z)
  pod?: [number, number, number]; // sensor pod point on the skin, local frame (anatomical model)
}

// Bottom to top (sacrum .. T1). Measured from BodyParts3D by scripts/build_spine_mesh.py: segment
// lengths and rest tilts between neighbouring vertebral-body centres, so the neutral column has the
// real lordosis and kyphosis. BodyParts3D, (c) The Database Center for Life Science, CC BY 4.0.
export const VERTEBRAE: VertebraSpec[] = column.vertebrae.map((v) => ({
  name: v.name,
  height: v.height,
  width: v.width,
  depth: v.depth,
  restTiltDeg: v.restTiltDeg,
  pod: [v.pod[0], v.pod[1], v.pod[2]],
}));

// index: vertebra where each sensor's pod is drawn. bendIndex (rendering only, defaults to index): vertebra
// where that sensor's rotation takes full effect in the drawn curve; vertebrae above the top one copy it.
// The upper thoracic sensor sits lower on the vest than T2, so its bend is applied from T4 up: the upper back
// shows the measured bend instead of spreading it thinly over T4-T2. Its pod is still drawn on T2.
export const SENSOR_ANCHORS: { region: Region; index: number; bendIndex?: number }[] = [
  { region: "pelvis", index: 0 }, // sacrum
  { region: "lumbar", index: 3 }, // L3
  { region: "thoracic", index: 10 }, // T8
  { region: "upper_thoracic", index: 16, bendIndex: 14 }, // pod on T2, bend from T4
];

const bendIndexOf = (anchor: { index: number; bendIndex?: number }) => anchor.bendIndex ?? anchor.index;

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
  const knots = SENSOR_ANCHORS.map(bendIndexOf);
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

/**
 * Fill sensors that have no data from the ones that do, so a missing sensor follows the body instead
 * of staying pinned at its last pose. Between two measured sensors: slerp by vertebra index. Beyond
 * the outermost measured sensor: copy it. Returns null when no sensor has data (keep the last pose).
 * The second array marks which entries were inferred.
 */
export function fillMissingAnchors(anchors: (THREE.Quaternion | null)[]): { rotations: THREE.Quaternion[]; inferred: boolean[] } | null {
  const known = anchors.map((q, k) => (q ? k : -1)).filter((k) => k >= 0);
  if (known.length === 0) {
    return null;
  }
  const at = (k: number) => bendIndexOf(SENSOR_ANCHORS[k]);
  const rotations = anchors.map((q, k) => {
    if (q) {
      return q.clone();
    }
    const below = known.filter((j) => j < k).pop();
    const above = known.find((j) => j > k);
    if (below === undefined) {
      return anchors[above!]!.clone();
    }
    if (above === undefined) {
      return anchors[below]!.clone();
    }
    const t = (at(k) - at(below)) / (at(above) - at(below));
    return anchors[below]!.clone().slerp(anchors[above]!, t);
  });
  return { rotations, inferred: anchors.map((q) => !q) };
}
