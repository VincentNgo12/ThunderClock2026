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

export interface WorkerTelemetry {
  schema_version: string;
  sequence: number;
  connection: { source: string; esp32: string };
  imus: ImuSlot[];
}
