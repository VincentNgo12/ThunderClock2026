// Connect / disconnect the IMU strip from the dashboard. The backend owns BLE: this only calls
// POST /device/connect (scan for the strip and start streaming) and POST /device/disconnect, then
// shows the stream state from GET /diagnostics. Hidden when the backend runs a simulated source.

import { useEffect, useState } from "react";
import type { WorkerTelemetry } from "../../telemetry/types";

interface Props {
  latest: WorkerTelemetry | null;
  apiBase: string;
}

type Format = "quaternion" | "both" | "raw_fusion";

const FORMATS: { value: Format; label: string }[] = [
  { value: "quaternion", label: "Quaternions" },
  { value: "both", label: "Quaternions + raw" },
  { value: "raw_fusion", label: "Raw, fused on laptop" },
];

interface Diagnostics {
  state: string;
  last_error: string | null;
}

export function DeviceConnect({ latest, apiBase }: Props) {
  const [format, setFormat] = useState<Format>("quaternion");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [diag, setDiag] = useState<Diagnostics | null>(null);
  const source = latest?.connection.source;
  const streaming = latest?.connection.esp32 === "connected";

  // Follow the BLE state (connecting, subscribed, error) between telemetry snapshots.
  useEffect(() => {
    if (!latest || source === "simulated") {
      return;
    }
    const poll = () =>
      fetch(`${apiBase}/diagnostics`)
        .then((response) => response.json())
        .then((body) => setDiag({ state: body.state, last_error: body.last_error ?? null }))
        .catch(() => setDiag(null));
    poll();
    const timer = window.setInterval(poll, 1000);
    return () => window.clearInterval(timer);
  }, [apiBase, source, latest === null]);

  if (!latest || source === "simulated") {
    return null;
  }

  const call = async (path: string) => {
    setBusy(true);
    setProblem(null);
    try {
      const response = await fetch(`${apiBase}${path}`, { method: "POST" });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        setProblem(body?.detail ?? `HTTP ${response.status}`);
      }
    } catch {
      setProblem("Backend unreachable");
    } finally {
      setBusy(false);
    }
  };

  const state = diag?.state ?? "disconnected";
  const active = streaming || ["connecting", "connected", "subscribed"].includes(state);
  const error = problem ?? (state === "error" ? diag?.last_error : null);

  return (
    <div className="device-connect">
      {active ? (
        <button disabled={busy} onClick={() => call("/device/disconnect")}>
          Disconnect strip
        </button>
      ) : (
        <>
          <select value={format} disabled={busy} onChange={(event) => setFormat(event.target.value as Format)}>
            {FORMATS.map((item) => (
              <option key={item.value} value={item.value}>{item.label}</option>
            ))}
          </select>
          <button className="primary" disabled={busy} onClick={() => call(`/device/connect?device_format=${format}`)}>
            {busy ? "Scanning…" : "Connect strip"}
          </button>
        </>
      )}
      <span className="device-state">{busy ? "scanning" : state}</span>
      {error && <span className="device-error" title={error}>{error}</span>}
    </div>
  );
}
