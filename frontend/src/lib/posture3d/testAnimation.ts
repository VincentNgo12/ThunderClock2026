// Client-side test motion: a looping sequence of spine movements used in place of live
// sensor data. It produces one rotation per sensor in the same frame as real telemetry
// (+X wearer-right, +Y up, +Z posterior), so the whole render path is exercised.

import * as THREE from "three";
import type { Region } from "../../telemetry/types";

interface Keyframe {
  at: number; // seconds into the loop
  label: string; // movement that ends at this keyframe
  flexionDeg: number; // forward bend (+) / extension (−) of the top of the spine
  lateralDeg: number; // bend toward the wearer's left (+) / right (−)
  twistDeg: number; // rotation to the wearer's left (+) / right (−)
}

const KEYFRAMES: Keyframe[] = [
  { at: 0, label: "Upright", flexionDeg: 0, lateralDeg: 0, twistDeg: 0 },
  { at: 2.2, label: "Forward bend", flexionDeg: 75, lateralDeg: 0, twistDeg: 0 },
  { at: 3.4, label: "Forward bend", flexionDeg: 75, lateralDeg: 0, twistDeg: 0 },
  { at: 5.4, label: "Back to upright", flexionDeg: 0, lateralDeg: 0, twistDeg: 0 },
  { at: 6.8, label: "Extension", flexionDeg: -25, lateralDeg: 0, twistDeg: 0 },
  { at: 8.2, label: "Back to upright", flexionDeg: 0, lateralDeg: 0, twistDeg: 0 },
  { at: 9.6, label: "Side bend left", flexionDeg: 0, lateralDeg: 35, twistDeg: 0 },
  { at: 11.0, label: "Back to upright", flexionDeg: 0, lateralDeg: 0, twistDeg: 0 },
  { at: 12.4, label: "Side bend right", flexionDeg: 0, lateralDeg: -35, twistDeg: 0 },
  { at: 13.8, label: "Back to upright", flexionDeg: 0, lateralDeg: 0, twistDeg: 0 },
  { at: 15.2, label: "Twist left", flexionDeg: 0, lateralDeg: 0, twistDeg: 35 },
  { at: 16.6, label: "Twist right", flexionDeg: 0, lateralDeg: 0, twistDeg: -35 },
  { at: 18.0, label: "Back to upright", flexionDeg: 0, lateralDeg: 0, twistDeg: 0 },
  { at: 19.5, label: "Bend and twist", flexionDeg: 55, lateralDeg: 15, twistDeg: 25 },
  { at: 21.5, label: "Back to upright", flexionDeg: 0, lateralDeg: 0, twistDeg: 0 },
];

export const TEST_LOOP_SECONDS = KEYFRAMES[KEYFRAMES.length - 1].at;

// Share of the total movement each sensor carries, so the spine curves instead of
// tipping as one rigid block.
const SHARE: Record<Region, number> = {
  pelvis: 0.12,
  lumbar: 0.45,
  thoracic: 0.8,
  upper_thoracic: 1,
};

const X = new THREE.Vector3(1, 0, 0);
const Y = new THREE.Vector3(0, 1, 0);
const Z = new THREE.Vector3(0, 0, 1);

function smoothstep(t: number): number {
  return t * t * (3 - 2 * t);
}

export interface TestFrame {
  label: string;
  rotations: Record<Region, THREE.Quaternion>;
}

export function testFrameAt(seconds: number): TestFrame {
  const time = ((seconds % TEST_LOOP_SECONDS) + TEST_LOOP_SECONDS) % TEST_LOOP_SECONDS;
  let next = KEYFRAMES.findIndex((frame) => frame.at > time);
  if (next <= 0) {
    next = KEYFRAMES.length - 1;
  }
  const from = KEYFRAMES[next - 1];
  const to = KEYFRAMES[next];
  const t = smoothstep(Math.min(1, Math.max(0, (time - from.at) / (to.at - from.at))));
  const mix = (a: number, b: number) => THREE.MathUtils.degToRad(a + (b - a) * t);
  const flexion = mix(from.flexionDeg, to.flexionDeg);
  const lateral = mix(from.lateralDeg, to.lateralDeg);
  const twist = mix(from.twistDeg, to.twistDeg);

  const rotations = {} as Record<Region, THREE.Quaternion>;
  for (const region of Object.keys(SHARE) as Region[]) {
    const share = SHARE[region];
    // Forward bend is a negative rotation about +X, matching the backend simulation.
    rotations[region] = new THREE.Quaternion()
      .setFromAxisAngle(Y, twist * share)
      .multiply(new THREE.Quaternion().setFromAxisAngle(Z, lateral * share))
      .multiply(new THREE.Quaternion().setFromAxisAngle(X, -flexion * share));
  }
  return { label: to.label, rotations };
}
