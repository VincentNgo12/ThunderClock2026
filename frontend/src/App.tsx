// Minimal sensor test page: a centred 3D spine driven live by the four spine IMUs,
// plus just enough status to confirm the backend → browser link is working.

import { useEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { SpineView, type SensorInput, type ViewName } from "./components/spine/SpineView";
import type { InterpolationMode } from "./lib/posture3d/spine";
import { telemetryUrl, useTelemetry } from "./hooks/useTelemetry";
import { REGION_LABELS, REGIONS, type FreshnessStatus, type Region } from "./telemetry/types";

const SILENT_AFTER_MS = 1000; // no message for this long → treat everything as stale

type Neutral = Partial<Record<Region, THREE.Quaternion>>;

export default function App() {
  const url = useMemo(telemetryUrl, []);
  const { socketState, latest, rateHz, lastMessageAt } = useTelemetry(url);
  const [neutral, setNeutral] = useState<Neutral>({});
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

  // Raw backend quaternions ([w,x,y,z]) → three.js quaternions, keyed by region.
  const raw = useMemo(() => {
    const result: Partial<Record<Region, THREE.Quaternion>> = {};
    for (const imu of latest?.imus ?? []) {
      if (imu.orientation) {
        const { w, x, y, z } = imu.orientation;
        result[imu.region] = new THREE.Quaternion(x, y, z, w).normalize();
      }
    }
    return result;
  }, [latest]);

  const sensors = useMemo(() => {
    const result = {} as Record<Region, SensorInput>;
    for (const region of REGIONS) {
      const slot = latest?.imus.find((imu) => imu.region === region);
      let status: FreshnessStatus = slot?.freshness.status ?? "unavailable";
      if (silent && status === "valid") {
        status = "stale";
      }
      let rotation = raw[region] ?? null;
      const reference = neutral[region];
      if (rotation && reference) {
        // Rotation relative to the captured neutral, expressed in the world frame.
        rotation = rotation.clone().multiply(reference.clone().invert());
      }
      result[region] = { status, rotation };
    }
    return result;
  }, [latest, raw, neutral, silent]);

  const allValid = REGIONS.every((region) => raw[region]);
  const connectionLabel =
    socketState === "open" ? (silent ? "Connected · no data" : "Live") : socketState === "connecting" ? "Connecting…" : "Disconnected";
  const connectionClass = socketState === "open" && !silent ? "ok" : socketState === "connecting" ? "wait" : "bad";

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
        {testAnimation && <div className="pill test">Test animation · {testLabel || "Starting"}</div>}
        <div className="meta">
          {rateHz} msg/s
          {latest ? ` · seq ${latest.sequence} · ${latest.connection.source} · ESP32 ${latest.connection.esp32}` : ""}
        </div>
      </header>

      <aside className="sensor-list">
        {REGIONS.map((region) => {
          const slot = latest?.imus.find((imu) => imu.region === region);
          const status = sensors[region].status;
          const q = slot?.orientation;
          return (
            <div className="sensor-row" key={region}>
              <span className={`status-dot ${status}`} />
              <span className="sensor-name">{REGION_LABELS[region]}</span>
              <span className="sensor-status">{status}</span>
              <span className="sensor-quat">
                {q ? `w ${q.w.toFixed(2)}  x ${q.x.toFixed(2)}  y ${q.y.toFixed(2)}  z ${q.z.toFixed(2)}` : "—"}
              </span>
            </div>
          );
        })}
      </aside>

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
        <button
          disabled={!allValid}
          title="Treat the current orientations as upright (client-side only)"
          onClick={() => {
            const captured: Neutral = {};
            for (const region of REGIONS) {
              captured[region] = raw[region]?.clone();
            }
            setNeutral(captured);
          }}
        >
          Set neutral
        </button>
        <button disabled={Object.keys(neutral).length === 0} onClick={() => setNeutral({})}>
          Clear neutral
        </button>
      </footer>

      <div className="endpoint">{url}</div>
    </main>
  );
}
