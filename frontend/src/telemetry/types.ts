// The subset of the backend's WorkerTelemetry 2.0 snapshot that the spine view reads.
// Full contract: docs/TELEMETRY_PROTOCOL.md.

export type Region = "pelvis" | "lumbar" | "thoracic" | "upper_thoracic";

export const REGIONS: Region[] = ["pelvis", "lumbar", "thoracic", "upper_thoracic"];

export const REGION_LABELS: Record<Region, string> = {
  pelvis: "Pelvis",
  lumbar: "Lumbar",
  thoracic: "Thoracic",
  upper_thoracic: "Upper thoracic",
};

export type FreshnessStatus = "valid" | "stale" | "unavailable" | "invalid";

// Quaternion in the backend's [w, x, y, z] convention, rotating segment-local vectors into
// the application frame (+X wearer-right, +Y up, +Z posterior).
export interface Quaternion {
  w: number;
  x: number;
  y: number;
  z: number;
}

export interface ImuSlot {
  region: Region;
  orientation: Quaternion | null;
  freshness: { status: FreshnessStatus; age_ms: number | null };
}

// Detection hub fields (detection/hub/demo_app.py). Absent when the plain backend serves the page.
export type FallStateName =
  | "normal"
  | "suspected"
  | "near_fall"
  | "fall_confirmed"
  | "dismissed"
  | "man_down_warning"
  | "man_down_critical";

export interface DetectionEvent {
  id: number;
  kind: string;
  confidence: number;
  episode_id: number | null;
  path: string | null;
  rule: string | null;
  playback_s: number | null;
}

export interface FallState {
  state: FallStateName;
  confidence: number | null;
  episode_id: number | null;
  dismissable: boolean;
  since_s: number | null;
  events: DetectionEvent[];
  label: string;
}

export interface DetectorStatus {
  status: string; // ok | uncalibrated | insufficient_rate | stale | error
  raw_rate_hz_5s: number | null;
  intake_dropped: number;
  update_ms_max: number;
}

export interface DemoBlock {
  live: boolean;
  phase: "building" | "playing" | "holding" | "error" | "live";
  error: string | null;
  building: string | null;
  scenario: string | null;
  description: string | null;
  expected: string | null;
  playback_s: number | null;
  duration_s: number | null;
  truth: { t: number; name: string }[];
}

export interface SegmentPose {
  rotation: Quaternion; // body frame, no twist: +X right, +Y up, +Z posterior
  flexion_deg: number; // + forward
  lateral_deg: number; // + right
  inclination_deg: number; // from vertical
}

export interface TrunkPosture {
  trunk: { flexion_deg: number; lateral_deg: number; inclination_deg: number };
  pelvis_inclination_deg: number | null;
  zone: "neutral" | "moderate" | "high" | "lying";
  lying: boolean;
  joints: Record<string, { flexion_deg: number; lateral_deg: number }>;
}

export interface WorkerTelemetry {
  schema_version: string;
  sequence: number;
  connection: { source: string; esp32: string };
  imus: ImuSlot[];
  alerts?: { severity: string; category: string; region: string; summary: string; sustained_seconds: number }[];
  mounting?: {
    calibrated: boolean;
    source: string | null;
    regions: Region[];
    auto_zero_pending: boolean;
    capturing: boolean;
    captured_poses: string[];
  };
  trunk_posture?: TrunkPosture | null;
  posture_source?: "gravity";
  // Heading-free segment poses from the backend (backend/app/segments.py), when a mounting is known.
  segments?: Partial<Record<Region, SegmentPose>> | null;
  segment_model?: { source: string; regions: Region[] };
  replay?: {
    file: string;
    t: number;
    duration_s: number;
    label: string | null;
    speed: number;
    pass: number;
    paused: boolean;
    loop: boolean;
    finished: boolean;
    steps: [number, string][];
  };
  fall_state?: FallState | null;
  detector?: DetectorStatus;
  demo?: DemoBlock;
}
