// Trunk posture from the backend's gravity model (backend/app/trunk_posture.py): trunk inclination against
// gravity, its ergonomic zone, lying, and active sustained-bend warnings.

import type { WorkerTelemetry } from "../../telemetry/types";

const ZONE_TEXT: Record<string, string> = {
  neutral: "Neutral",
  moderate: "Bent",
  high: "Deep bend (> 60°)",
  lying: "Lying",
};

function direction(flexion: number, lateral: number): string {
  const parts: string[] = [];
  if (Math.abs(flexion) >= 5) {
    parts.push(`${Math.abs(flexion).toFixed(0)}° ${flexion > 0 ? "forward" : "back"}`);
  }
  if (Math.abs(lateral) >= 5) {
    parts.push(`${Math.abs(lateral).toFixed(0)}° ${lateral > 0 ? "right" : "left"}`);
  }
  return parts.length ? parts.join(", ") : "upright";
}

export function PosturePanel({ latest }: { latest: WorkerTelemetry | null }) {
  const posture = latest?.trunk_posture;
  const mounting = latest?.mounting;
  if (!latest) {
    return null;
  }
  if (!mounting?.calibrated) {
    const receiving = latest.imus.some((imu) => imu.orientation);
    return (
      <aside className="posture-panel zone-moderate">
        <div className="posture-title">Posture</div>
        <div className="posture-hint">
          {receiving
            ? "Receiving data from the strip. The spine stays upright until this strip is calibrated once: press Calibrate… (about 1 minute)."
            : "Not calibrated: connect the strip, then press Calibrate…."}
        </div>
      </aside>
    );
  }
  const alerts = (latest.alerts ?? []).filter((alert) => alert.category === "posture");
  const zone = posture?.zone ?? "neutral";
  return (
    <aside className={`posture-panel zone-${zone}`}>
      <div className="posture-title">
        Posture <span className="posture-zone">{posture ? ZONE_TEXT[zone] ?? zone : "no data"}</span>
      </div>
      {posture && !posture.lying && (
        <div className="posture-line">
          Trunk {direction(posture.trunk.flexion_deg, posture.trunk.lateral_deg)}
          <span className="posture-muted"> · {posture.trunk.inclination_deg.toFixed(0)}° from vertical</span>
        </div>
      )}
      {posture?.lying && <div className="posture-line">Trunk horizontal ({posture.trunk.inclination_deg.toFixed(0)}°)</div>}
      {alerts.map((alert) => (
        <div key={alert.summary} className={`posture-alert ${alert.severity}`}>
          {alert.summary} · {Math.round(alert.sustained_seconds)}s
        </div>
      ))}
      {mounting.auto_zero_pending && <div className="posture-hint">Stand still to re-zero upright…</div>}
    </aside>
  );
}
