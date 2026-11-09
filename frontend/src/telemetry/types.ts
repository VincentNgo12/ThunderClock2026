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

export interface WorkerTelemetry {
  schema_version: string;
  sequence: number;
  connection: { source: string; esp32: string };
  imus: ImuSlot[];
  alerts?: { category: string; region: string; summary: string; sustained_seconds: number }[];
  fall_state?: FallState | null;
  detector?: DetectorStatus;
  demo?: DemoBlock;
}
