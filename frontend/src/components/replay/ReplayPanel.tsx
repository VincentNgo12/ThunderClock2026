// Replay a recorded "Collect data" session through the live view: the backend swaps its telemetry source for
// the recording (POST /replay/start), so the spine, posture and detection behave exactly as they would live.
// Stopping restores the previous source (a live strip must be reconnected with Connect strip).

import { useCallback, useEffect, useState } from "react";
import type { WorkerTelemetry } from "../../telemetry/types";

interface Props {
  latest: WorkerTelemetry | null;
  apiBase: string;
  onClose: () => void;
}

interface RecordingInfo {
  file: string;
  session: string;
  started_at: string | null;
  source: string | null;
  device_format: string | null;
  wearer: string | null;
  notes: string | null;
  duration_s: number | null;
  aborted: boolean | null;
  size_mb: number;
}

const SPEEDS = [0.5, 1, 2, 4];

async function post(url: string): Promise<string | null> {
  try {
    const response = await fetch(url, { method: "POST" });
    if (response.ok) {
      return null;
    }
    const body = await response.json().catch(() => null);
    return body?.detail ?? `HTTP ${response.status}`;
  } catch {
    return "Backend unreachable";
  }
}

function stepName(label: string): string {
  const [step, phase] = label.split(":");
  const [movement, rep] = step.split("#");
  return `${movement.replace(/_/g, " ")}${rep && rep !== "1" ? ` ${rep}` : ""}${phase && phase !== "hold" ? ` (${phase})` : ""}`;
}

export function ReplayPanel({ latest, apiBase, onClose }: Props) {
  const [recordings, setRecordings] = useState<RecordingInfo[]>([]);
  const [folder, setFolder] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [speed, setSpeed] = useState(1);
  const [loop, setLoop] = useState(false);
  const [useSessionCalibration, setUseSessionCalibration] = useState(true);
  const [problem, setProblem] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const replay = latest?.replay;

  const refresh = useCallback(() => {
    fetch(`${apiBase}/replay/recordings`)
      .then((r) => {
        if (r.status === 404) {
          throw new Error("outdated");
        }
        return r.json();
      })
      .then((body) => {
        setProblem(null);
        setRecordings(body.recordings ?? []);
        setFolder(body.folder ?? "");
        setSelected((current) => current ?? body.recordings?.[0]?.file ?? null);
      })
      .catch((error: Error) =>
        setProblem(
          error.message === "outdated"
            ? "This backend has no replay support: it is running older code. Restart the backend (Ctrl+C, then start it again)."
            : "Backend unreachable",
        ),
      );
  }, [apiBase]);

  useEffect(refresh, [refresh]);

  const start = async (file: string, startT = 0, speedValue = speed, loopValue = loop) => {
    setBusy(true);
    setProblem(
      await post(
        `${apiBase}/replay/start?file=${encodeURIComponent(file)}&speed=${speedValue}&loop=${loopValue}` +
          `&start_t=${startT.toFixed(2)}&model=${useSessionCalibration ? "recording" : "current"}`,
      ),
    );
    setBusy(false);
  };

  const active = replay ? recordings.find((r) => r.file === replay.file) : undefined;
  // Only the steps worth jumping to: the holds (and the first step).
  const steps = (replay?.steps ?? []).filter(([, name], i) => i === 0 || name.endsWith(":hold"));

  return (
    <aside className="replay-panel">
      <div className="mount-head">
        <span className="mount-title">Replay a recorded session</span>
        <button className="calib-close" onClick={onClose} aria-label="Close">×</button>
      </div>

      {replay ? (
        <>
          <div className="replay-now">
            <div className="replay-file">{active?.session ?? replay.file}</div>
            <div className="replay-step">
              {replay.finished ? "Finished" : replay.paused ? "Paused" : "Playing"} · {replay.label ? stepName(replay.label) : "—"} ·{" "}
              {replay.speed}× {replay.loop ? "· loop" : ""}
            </div>
          </div>
          <div
            className="replay-bar"
            title="Click to jump"
            onClick={(event) => {
              const box = event.currentTarget.getBoundingClientRect();
              const t = ((event.clientX - box.left) / box.width) * replay.duration_s;
              void start(replay.file, t, replay.speed, replay.loop);
            }}
          >
            <span style={{ width: `${(100 * replay.t) / Math.max(replay.duration_s, 1)}%` }} />
          </div>
          <div className="replay-time">
            {replay.t.toFixed(0)}s / {replay.duration_s.toFixed(0)}s
          </div>
          <div className="mount-actions replay-actions">
            <button
              disabled={busy || replay.finished}
              onClick={async () => setProblem(await post(`${apiBase}/replay/pause?paused=${!replay.paused}`))}
            >
              {replay.paused ? "Resume" : "Pause"}
            </button>
            <button disabled={busy} onClick={() => void start(replay.file, 0, replay.speed, replay.loop)}>
              Restart
            </button>
            <button
              className="primary"
              disabled={busy}
              onClick={async () => {
                setProblem(await post(`${apiBase}/replay/stop`));
                refresh();
              }}
            >
              Stop replay
            </button>
          </div>
          {steps.length > 0 && (
            <div className="replay-steps">
              {steps.map(([t, name]) => (
                <button
                  key={`${t}-${name}`}
                  className={replay.label === name ? "active" : ""}
                  onClick={() => void start(replay.file, Math.max(0, t - 1), replay.speed, replay.loop)}
                >
                  {stepName(name)}
                </button>
              ))}
            </div>
          )}
        </>
      ) : (
        <>
          {recordings.length === 0 ? (
            <p className="mount-help">No recordings yet in {folder || "backend/recordings"}. Use Collect data first.</p>
          ) : (
            <div className="replay-list">
              {recordings.map((r) => (
                <label key={r.file} className={`replay-item ${selected === r.file ? "selected" : ""}`}>
                  <input type="radio" name="recording" checked={selected === r.file} onChange={() => setSelected(r.file)} />
                  <span>
                    <span className="replay-file">{r.session}</span>
                    <span className="replay-meta">
                      {r.started_at ? new Date(r.started_at).toLocaleString() : ""} · {r.duration_s ? `${Math.round(r.duration_s)}s` : "?"}
                      {r.wearer ? ` · ${r.wearer}` : ""} · {r.source}/{r.device_format}
                      {r.aborted ? " · stopped early" : ""}
                    </span>
                    {r.notes && <span className="replay-meta">{r.notes}</span>}
                  </span>
                </label>
              ))}
            </div>
          )}
          <div className="replay-options">
            <label>
              Speed
              <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
                {SPEEDS.map((s) => (
                  <option key={s} value={s}>{s}×</option>
                ))}
              </select>
            </label>
            <label className="collect-check">
              <input type="checkbox" checked={loop} onChange={(e) => setLoop(e.target.checked)} />
              Loop
            </label>
            <label className="collect-check" title="Fit the sensor mounting from the session's own stand + bend steps">
              <input
                type="checkbox"
                checked={useSessionCalibration}
                onChange={(e) => setUseSessionCalibration(e.target.checked)}
              />
              Use the session's own calibration
            </label>
          </div>
          <p className="mount-help">Replaying disconnects a live strip; reconnect it after stopping.</p>
          <div className="mount-actions">
            <button onClick={refresh}>Refresh</button>
            <button className="primary" disabled={!selected || busy} onClick={() => selected && void start(selected)}>
              Play
            </button>
          </div>
        </>
      )}
      {problem && <p className="calib-error">{problem}</p>}
    </aside>
  );
}
