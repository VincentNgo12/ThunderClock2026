// Spine sensor page: a centred 3D spine driven by the four spine IMUs.
// The backend turns each sensor's orientation into a heading-free segment pose (gravity model,
// backend/app/segments.py) using the strip's saved mounting; this page only draws those poses. The 6-axis
// sensors' heading drifts tens of degrees per minute, so nothing here uses heading.

import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { CalibrationWizard } from "./components/calibration/CalibrationWizard";
import { DataCollectionWizard } from "./components/calibration/DataCollectionWizard";
import { DeviceConnect } from "./components/device/DeviceConnect";
import { DetectionCard } from "./components/detection/DetectionCard";
import { PosturePanel } from "./components/posture/PosturePanel";
import { ReplayPanel } from "./components/replay/ReplayPanel";
import { SpineView, type SensorInput, type ViewName } from "./components/spine/SpineView";
import type { InterpolationMode } from "./lib/posture3d/spine";
import { telemetryUrl, useTelemetry } from "./hooks/useTelemetry";
import { REGION_LABELS, REGIONS, type FreshnessStatus, type Region } from "./telemetry/types";

const SILENT_AFTER_MS = 1000; // no message for this long → treat everything as stale

export default function App() {
  const url = useMemo(telemetryUrl, []);
  // Plain-HTTP base of the same backend, for the POST calls.
  const apiBase = useMemo(() => {
    try {
      const parsed = new URL(url);
      return `${parsed.protocol === "wss:" ? "https:" : "http:"}//${parsed.host}`;
    } catch {
      return "http://127.0.0.1:8000";
    }
  }, [url]);
  const { socketState, latest, rateHz, lastMessageAt } = useTelemetry(url);
  const [calibrating, setCalibrating] = useState(false);
  const [collecting, setCollecting] = useState(false);
  const [replaying, setReplaying] = useState(false);
  const [zeroing, setZeroing] = useState<string | null>(null);
  const [view, setView] = useState<{ name: ViewName; nonce: number }>({ name: "oblique", nonce: 0 });
  const [showAxes, setShowAxes] = useState(false);
  const [interpolation, setInterpolation] = useState<InterpolationMode>("spline");
  const [testAnimation, setTestAnimation] = useState(false);
  const [testLabel, setTestLabel] = useState("");
  const [now, setNow] = useState(performance.now());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(performance.now()), 250);
    return () => window.clearInterval(timer);
  }, []);

  const silent = socketState !== "open" || lastMessageAt === null || now - lastMessageAt > SILENT_AFTER_MS;
  const calibrated = Boolean(latest?.mounting?.calibrated ?? latest?.segment_model);
  const streaming = Boolean(latest && latest.connection.esp32 !== "disconnected");

  // Segment poses come from the backend; without a mounting the spine stays upright.
  const sensors = useMemo(() => {
    const result = {} as Record<Region, SensorInput>;
    for (const region of REGIONS) {
      const slot = latest?.imus.find((imu) => imu.region === region);
      let status: FreshnessStatus = slot?.freshness.status ?? "unavailable";
      if (silent && status === "valid") {
        status = "stale";
      }
      const pose = latest?.segments?.[region];
      const rotation = pose ? new THREE.Quaternion(pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w) : null;
      result[region] = { status, rotation };
    }
    return result;
  }, [latest, silent]);

  const connectionLabel =
    socketState === "open" ? (silent ? "Connected · no data" : "Live") : socketState === "connecting" ? "Connecting…" : "Disconnected";
  const connectionClass = socketState === "open" && !silent ? "ok" : socketState === "connecting" ? "wait" : "bad";

  const rezero = async () => {
    setZeroing("Stand still…");
    try {
      const response = await fetch(`${apiBase}/mounting/zero?seconds=2`, { method: "POST" });
      const body = await response.json().catch(() => ({}));
      setZeroing(response.ok ? "Re-zeroed" : body.detail ?? "Re-zero failed");
    } catch {
      setZeroing("Backend unreachable");
    }
    window.setTimeout(() => setZeroing(null), 2500);
  };

  return (
    <main className="app">
      <SpineView
        sensors={sensors}
        view={view}
        showAxes={showAxes}
        interpolation={interpolation}
        testAnimation={testAnimation}
        onTestLabel={setTestLabel}
      />

      <header className="topbar">
        <div className="title">Spine sensors</div>
        <div className={`pill ${connectionClass}`}>
          <span className="pill-dot" />
          {connectionLabel}
        </div>
        <DeviceConnect latest={latest} apiBase={apiBase} />
        {calibrated ? (
          <div className="pill model" title={latest?.mounting?.source ?? latest?.segment_model?.source ?? ""}>
            Gravity model
          </div>
        ) : (
          latest && <div className="pill uncalibrated">Not calibrated</div>
        )}
        {latest?.replay && (
          <div className="pill replay">
            Replay {latest.replay.t.toFixed(0)}s / {latest.replay.duration_s.toFixed(0)}s · {latest.replay.label ?? "—"}
          </div>
        )}
        {zeroing && <div className="pill wait">{zeroing}</div>}
        {testAnimation && <div className="pill test">Test animation · {testLabel || "Starting"}</div>}
        <div className="meta">
          {rateHz} msg/s
          {latest ? ` · seq ${latest.sequence} · ${latest.connection.source} · ESP32 ${latest.connection.esp32}` : ""}
        </div>
      </header>

      <DetectionCard latest={latest} apiBase={apiBase} />

      <div className="right-column">
      <aside className="sensor-list">
        {REGIONS.map((region) => {
          const status = sensors[region].status;
          const pose = latest?.segments?.[region];
          const q = latest?.imus.find((imu) => imu.region === region)?.orientation;
          return (
            <div className="sensor-row" key={region}>
              <span className={`status-dot ${status}`} />
              <span className="sensor-name">{REGION_LABELS[region]}</span>
              <span className="sensor-status">{status}</span>
              <span className="sensor-quat" title="Sensor quaternion as received (sensor frame, heading drifts)">
                {q ? `w ${q.w.toFixed(2)}  x ${q.x.toFixed(2)}  y ${q.y.toFixed(2)}  z ${q.z.toFixed(2)}` : "—"}
              </span>
              <span />
              <span />
              <span />
              <span className="sensor-angles" title="Segment tilt from the gravity model">
                {!pose
                  ? ""
                  : pose.inclination_deg > 60
                    ? `tilted ${pose.inclination_deg.toFixed(0)}° from vertical` // near horizontal: fwd/side split is meaningless
                    : `${pose.flexion_deg >= 0 ? "fwd" : "back"} ${Math.abs(pose.flexion_deg).toFixed(0)}°  ${pose.lateral_deg >= 0 ? "right" : "left"} ${Math.abs(pose.lateral_deg).toFixed(0)}°`}
              </span>
            </div>
          );
        })}
      </aside>

      <PosturePanel latest={latest} />
      </div>

      <footer className="controls">
        {(["oblique", "side", "back"] as ViewName[]).map((name) => (
          <button
            key={name}
            className={view.name === name ? "active" : ""}
            onClick={() => setView({ name, nonce: view.nonce + 1 })}
          >
            {name === "oblique" ? "3/4" : name === "side" ? "Side" : "Back"}
          </button>
        ))}
        <span className="divider" />
        {(["spline", "linear"] as InterpolationMode[]).map((mode) => (
          <button
            key={mode}
            className={interpolation === mode ? "active" : ""}
            title={mode === "spline" ? "Smooth curve through the sensors" : "Straight blend between neighbouring sensors"}
            onClick={() => setInterpolation(mode)}
          >
            {mode === "spline" ? "Spline" : "Linear"}
          </button>
        ))}
        <span className="divider" />
        <button
          className={testAnimation ? "active test" : ""}
          title="Play a built-in bending sequence instead of the live sensor data"
          onClick={() => setTestAnimation(!testAnimation)}
        >
          {testAnimation ? "Stop test" : "Test bend"}
        </button>
        <button className={showAxes ? "active" : ""} onClick={() => setShowAxes(!showAxes)}>
          Sensor axes
        </button>
        <span className="divider" />
        <button
          disabled={!calibrated || !streaming || zeroing !== null}
          title="Stand still and upright: takes a new upright, keeps each sensor's calibrated forward direction"
          onClick={rezero}
        >
          Re-zero upright
        </button>
        <button disabled={!streaming} onClick={() => setCalibrating(true)}>
          Calibrate…
        </button>
        <button
          disabled={!streaming}
          title="Record a labelled session (bends, twists, lying) for later analysis"
          onClick={() => setCollecting(true)}
        >
          Collect data
        </button>
        <button className={latest?.replay ? "active" : ""} onClick={() => setReplaying(!replaying)}>
          Replay
        </button>
      </footer>

      {replaying && <ReplayPanel latest={latest} apiBase={apiBase} onClose={() => setReplaying(false)} />}

      {calibrating && <CalibrationWizard apiBase={apiBase} onClose={() => setCalibrating(false)} />}

      {collecting && (
        <DataCollectionWizard
          apiBase={apiBase}
          clientMeta={() => ({
            view: { mounting: latest?.mounting ?? null, interpolation },
            user_agent: navigator.userAgent,
            page_rate_hz: rateHz,
          })}
          onClose={() => setCollecting(false)}
        />
      )}

      <div className="endpoint">
        {url} · Vertebrae: BodyParts3D © The Database Center for Life Science, CC BY 4.0
      </div>
    </main>
  );
}
