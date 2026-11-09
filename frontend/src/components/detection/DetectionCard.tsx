// Fall-detection card for the spine demo: the hub's state, a scenario picker (simulated source only)
// and the supervisor's dismiss button. Reads fall_state/detector/demo from the telemetry snapshot;
// the only writes are two small POSTs (query parameters, so no CORS preflight).

import { useEffect, useState } from "react";
import type { FallStateName, WorkerTelemetry } from "../../telemetry/types";

interface Props {
  latest: WorkerTelemetry | null;
  apiBase: string; // http://host:port of the backend
}

interface MenuItem {
  group: string;
  id: string;
  label: string;
}

interface ScenarioInfo {
  id: string;
  group: string;
  description: string;
  expected: string | null;
}

const STATE_TEXT: Record<FallStateName, string> = {
  normal: "Normal",
  suspected: "Checking…",
  near_fall: "Near fall",
  fall_confirmed: "Fall detected",
  dismissed: "Dismissed",
  man_down_warning: "Man down · warning",
  man_down_critical: "Man down · critical",
};

const STATE_TONE: Record<FallStateName, "ok" | "wait" | "bad"> = {
  normal: "ok",
  suspected: "wait",
  near_fall: "wait",
  fall_confirmed: "bad",
  dismissed: "wait",
  man_down_warning: "bad",
  man_down_critical: "bad",
};

const KIND_TEXT: Record<string, string> = {
  suspected_fall: "Impact trigger",
  near_fall: "Near fall",
  recovered: "Recovered",
  fall_confirmed: "Fall confirmed",
  man_down_warning: "Man-down warning",
  man_down_critical: "Man-down critical",
  cleared: "Cleared",
};

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

export function DetectionCard({ latest, apiBase }: Props) {
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [menu, setMenu] = useState<MenuItem[]>([]);
  const [problem, setProblem] = useState<string | null>(null);
  const fall = latest?.fall_state;
  const demo = latest?.demo;
  const detector = latest?.detector;
  const live = demo?.live ?? false;

  useEffect(() => {
    if (live || !demo) {
      return;
    }
    fetch(`${apiBase}/demo/scenarios`)
      .then((response) => response.json())
      .then((body) => {
        setScenarios(body.scenarios ?? []);
        setMenu(body.menu ?? []);
      })
      .catch(() => undefined);
  }, [apiBase, live, demo === undefined]);

  if (!fall || !demo) {
    return null; // plain backend: nothing to show
  }

  const tone = STATE_TONE[fall.state];
  const building = demo.phase === "building";
  const groups = Array.from(new Set(menu.map((item) => item.group)));
  const pickedLabel = menu.find((item) => item.id === demo.scenario)?.label;
  const posture = (latest?.alerts ?? []).filter((alert) => alert.category === "posture");
  const picked = demo.scenario ?? "";
  // Between loops of a replay no samples arrive, so "stale" there is expected, not a fault.
  const quietHold = demo.phase === "holding" || demo.phase === "building";
  const detectorNote =
    detector && detector.status !== "ok" && !(quietHold && detector.status === "stale")
      ? `Detector ${detector.status.replace("_", " ")}`
      : null;

  return (
    <aside className={`detect-card ${tone}`}>
      <div className="detect-head">
        <span className={`detect-dot ${tone}`} />
        <span className="detect-state">{STATE_TEXT[fall.state]}</span>
        {fall.dismissable && (
          <button
            className="detect-dismiss"
            onClick={async () => setProblem(await post(`${apiBase}/incident/dismiss`))}
          >
            Dismiss
          </button>
        )}
      </div>

      <div className="detect-sub">
        {live
          ? "Live strip"
          : building
            ? `Building ${demo.building ?? "scenario"}…`
            : `${pickedLabel ?? picked} · ${demo.playback_s !== null && demo.playback_s > 0 ? Math.floor(demo.playback_s) : 0}s / ${Math.round(demo.duration_s ?? 0)}s`}
        {detectorNote ? ` · ${detectorNote}` : ""}
      </div>

      {!live && (
        <div className="detect-pick">
          <select
            value={picked}
            disabled={building}
            onChange={async (event) =>
              setProblem(await post(`${apiBase}/demo/scenario?scenario=${encodeURIComponent(event.target.value)}`))
            }
          >
            {!scenarios.some((s) => s.id === picked) && <option value={picked}>{picked || "Scenario"}</option>}
            {groups.map((group) => (
              <optgroup key={group} label={group}>
                {menu
                  .filter((item) => item.group === group)
                  .map((item) => (
                    <option key={item.id} value={item.id}>{item.label}</option>
                  ))}
              </optgroup>
            ))}
            <optgroup label="All scenarios">
              {scenarios.map((s) => (
                <option key={s.id} value={s.id}>{s.id} · {s.description}</option>
              ))}
            </optgroup>
          </select>
          {demo.expected && <span className="detect-expected">expects {demo.expected.replace("_", " ")}</span>}
        </div>
      )}

      <div className={`detect-posture ${posture.length ? "warn" : ""}`}>
        Posture:{" "}
        {posture.length
          ? posture.map((alert) => `${alert.summary.replace("Sustained ", "")} ${Math.round(alert.sustained_seconds)}s`).join(", ")
          : "OK"}
      </div>

      {fall.events.length > 0 && (
        <ol className="detect-events">
          {fall.events.slice(-4).map((event) => (
            <li key={event.id}>
              <span className="detect-time">{event.playback_s !== null ? `${Math.max(0, event.playback_s).toFixed(0)}s` : ""}</span>
              {KIND_TEXT[event.kind] ?? event.kind}
            </li>
          ))}
        </ol>
      )}

      {(problem || demo.error) && <div className="detect-error">{problem ?? demo.error}</div>}
      <div className="detect-foot">{fall.label} · simulated data</div>
    </aside>
  );
}
