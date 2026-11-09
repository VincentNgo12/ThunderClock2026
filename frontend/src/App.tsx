// Minimal sensor test page: a centred 3D spine driven live by the four spine IMUs,
// plus just enough status to confirm the backend → browser link is working.

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { CalibrationWizard } from "./components/calibration/CalibrationWizard";
import { MountingPanel } from "./components/calibration/MountingPanel";
import { DeviceConnect } from "./components/device/DeviceConnect";
import { DetectionCard } from "./components/detection/DetectionCard";
import { SpineView, type SensorInput, type ViewName } from "./components/spine/SpineView";
import { shareTwist, type InterpolationMode } from "./lib/posture3d/spine";
import {
  defaultMountings,
  loadCalibration,
  saveCalibration,
  toBodyFrame,
  type MountingSet,
} from "./lib/posture3d/mounting";
import { telemetryUrl, useTelemetry } from "./hooks/useTelemetry";
import { REGION_LABELS, REGIONS, type FreshnessStatus, type Region } from "./telemetry/types";

const SILENT_AFTER_MS = 1000; // no message for this long → treat everything as stale

type Neutral = Partial<Record<Region, THREE.Quaternion>>;

// Neutral and per-sensor mounting survive a page reload (per browser).
const STORED = loadCalibration();
const STORED_NEUTRAL: Neutral = Object.fromEntries(
  Object.entries(STORED?.neutral ?? {}).map(([region, q]) => [region, new THREE.Quaternion(...q!)]),
);

export default function App() {
  const url = useMemo(telemetryUrl, []);
  // Plain-HTTP base of the same backend, for the detection card's two POST calls.
  const apiBase = useMemo(() => {
    try {
      const parsed = new URL(url);
      return `${parsed.protocol === "wss:" ? "https:" : "http:"}//${parsed.host}`;
    } catch {
      return "http://127.0.0.1:8000";
    }
  }, [url]);
  const { socketState, latest, rateHz, lastMessageAt } = useTelemetry(url);
  const [neutral, setNeutral] = useState<Neutral>(STORED_NEUTRAL);
  const [mountings, setMountings] = useState<MountingSet>(STORED?.mountings ?? defaultMountings());
  const [showMounting, setShowMounting] = useState(false);
  // 6-axis sensors cannot hold heading; by default show only each segment's tilt (no axial twist).
  const [twistLock, setTwistLock] = useState(true);
  const [calibrating, setCalibrating] = useState(false);

  useEffect(() => {
    saveCalibration({
      neutral: Object.fromEntries(
        Object.entries(neutral).filter(([, q]) => q).map(([r, q]) => [r, [q!.x, q!.y, q!.z, q!.w]]),
      ),
      mountings,
    });
  }, [neutral, mountings]);
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
  const rawRef = useRef(raw);
  rawRef.current = raw;

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
        // Rotation relative to the captured neutral, expressed in the sensor's world frame.
        rotation = rotation.clone().multiply(reference.clone().invert());
      }
      if (rotation) {
        // Sensor frame -> wearer frame (heading, axis remap and offsets; see lib/posture3d/mounting.ts).
        rotation = toBodyFrame(rotation, mountings[region]);
      }
      result[region] = { status, rotation };
    }
    if (twistLock) {
      // Keep the body's shared heading (a turn before lying down), drop each sensor's own heading drift.
      const shared = shareTwist(REGIONS.map((region) => result[region].rotation));
      REGIONS.forEach((region, k) => {
        result[region] = { ...result[region], rotation: shared[k] };
      });
    }
    return result;
  }, [latest, raw, neutral, mountings, twistLock, silent]);

  const allValid = REGIONS.every((region) => raw[region]);
  const anyValid = REGIONS.some((region) => raw[region]);
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
        <DeviceConnect latest={latest} apiBase={apiBase} />
        {testAnimation && <div className="pill test">Test animation · {testLabel || "Starting"}</div>}
        <div className="meta">
          {rateHz} msg/s
          {latest ? ` · seq ${latest.sequence} · ${latest.connection.source} · ESP32 ${latest.connection.esp32}` : ""}
        </div>
      </header>

      <DetectionCard latest={latest} apiBase={apiBase} />

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
          disabled={!anyValid}
          title={
            allValid
              ? "Treat the current orientations as upright (client-side only)"
              : "Treat the current orientations of the connected sensors as upright; the others keep their previous neutral"
          }
          onClick={() => {
            // Capture every sensor that is reporting now; a missing one keeps its previous neutral (if any).
            const captured: Neutral = { ...neutral };
            for (const region of REGIONS) {
              if (raw[region]) {
                captured[region] = raw[region]!.clone();
              }
            }
            setNeutral(captured);
          }}
        >
          Set neutral
        </button>
        <button disabled={Object.keys(neutral).length === 0} onClick={() => setNeutral({})}>
          Clear neutral
        </button>
        <span className="divider" />
        <button
          className={twistLock ? "active" : ""}
          title="Give all segments one shared twist about the spine (keeps whole-body turns, removes per-sensor heading drift). Turn off to show raw axial twist."
          onClick={() => setTwistLock(!twistLock)}
        >
          No twist
        </button>
        <button className={showMounting ? "active" : ""} onClick={() => setShowMounting(!showMounting)}>
          Axes &amp; offsets
        </button>
        <button disabled={!REGIONS.some((r) => raw[r])} onClick={() => setCalibrating(true)}>
          Calibrate…
        </button>
      </footer>

      {showMounting && (
        <MountingPanel
          mountings={mountings}
          onChange={setMountings}
          onCalibrate={() => setCalibrating(true)}
          onClose={() => setShowMounting(false)}
        />
      )}

      {calibrating && (
        <CalibrationWizard
          getRaw={() => rawRef.current}
          current={mountings}
          onApply={(captured, solved) => {
            setNeutral(captured);
            setMountings(solved);
          }}
          onClose={() => setCalibrating(false)}
        />
      )}

      <div className="endpoint">
        {url} · Vertebrae: BodyParts3D © The Database Center for Life Science, CC BY 4.0
      </div>
    </main>
  );
}
