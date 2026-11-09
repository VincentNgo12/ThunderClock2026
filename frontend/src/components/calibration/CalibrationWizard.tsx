// Sensor mounting calibration, done by the backend (backend/app/mounting.py): the wizard prompts the wearer
// and asks the backend to capture each held pose from the live stream (POST /mounting/capture), then fits
// and saves each sensor's mounting for this strip (POST /mounting/fit). Mounting is physical, so it survives
// ESP32 reboots; re-run only when the sensors move on the vest.

import { useEffect, useState } from "react";

interface Props {
  apiBase: string;
  onClose: () => void;
}

type Pose = "stand" | "forward" | "backward" | "right" | "left";

interface Step {
  pose: Pose;
  title: string;
  prompt: string;
  optional: boolean;
}

const STEPS: Step[] = [
  { pose: "stand", title: "Stand upright", prompt: "Stand straight and relaxed, arms by your sides. Stay still.", optional: false },
  { pose: "forward", title: "Bend forward", prompt: "Bend forward from the waist (about 45°) and hold it still.", optional: false },
  { pose: "backward", title: "Lean backward", prompt: "Stand up, then lean back gently (about 20°) and hold it.", optional: true },
  { pose: "right", title: "Bend to your right", prompt: "Stand up, then bend sideways to your RIGHT (about 30°) and hold it.", optional: true },
  { pose: "left", title: "Bend to your left", prompt: "Stand up, then bend sideways to your LEFT (about 30°) and hold it.", optional: true },
];

const GET_READY_S = 4;
const HOLD_S = 2;
const MIN_BEND_DEG = 8;

interface SensorCapture {
  samples: number;
  spread_deg: number;
  still: boolean;
  tilt_from_stand_deg?: number;
}

interface FitReport {
  report: Record<string, { tilt_deg?: Record<string, number>; direction_error_deg?: number | null; error?: string }>;
  regions: string[];
  saved_path: string | null;
}

const REGION_NAMES: Record<string, string> = {
  pelvis: "Pelvis",
  lumbar: "Lumbar",
  thoracic: "Thoracic",
  upper_thoracic: "Upper thoracic",
};

async function post<T>(url: string): Promise<{ ok: boolean; data: T & { detail?: string } }> {
  try {
    const response = await fetch(url, { method: "POST" });
    return { ok: response.ok, data: await response.json().catch(() => ({})) };
  } catch {
    return { ok: false, data: { detail: "Backend unreachable" } as T & { detail?: string } };
  }
}

type Phase = "intro" | "ready" | "capturing" | "review" | "fitting" | "done";

export function CalibrationWizard({ apiBase, onClose }: Props) {
  const [phase, setPhase] = useState<Phase>("intro");
  const [index, setIndex] = useState(0);
  const [countdown, setCountdown] = useState(GET_READY_S);
  const [capture, setCapture] = useState<Record<string, SensorCapture> | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [fit, setFit] = useState<FitReport | null>(null);
  const step = STEPS[index];

  // Countdown, then the backend captures HOLD_S seconds of the held pose.
  useEffect(() => {
    if (phase !== "ready") {
      return;
    }
    setCountdown(GET_READY_S);
    const started = performance.now();
    const timer = window.setInterval(() => {
      const left = GET_READY_S - (performance.now() - started) / 1000;
      setCountdown(Math.max(0, Math.ceil(left)));
      if (left <= 0) {
        window.clearInterval(timer);
        void runCapture();
      }
    }, 100);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, index]);

  const runCapture = async () => {
    setPhase("capturing");
    setProblem(null);
    const response = await post<{ sensors: Record<string, SensorCapture> }>(
      `${apiBase}/mounting/capture?pose=${step.pose}&seconds=${HOLD_S}`,
    );
    if (!response.ok) {
      setProblem(response.data.detail ?? "Capture failed");
      setCapture(null);
    } else {
      setCapture(response.data.sensors);
    }
    setPhase("review");
  };

  const issues = (): string[] => {
    if (!capture) {
      return [];
    }
    const out: string[] = [];
    const moving = Object.entries(capture).filter(([, c]) => !c.still).map(([r]) => REGION_NAMES[r] ?? r);
    if (moving.length) {
      out.push(`Moved during the hold: ${moving.join(", ")}.`);
    }
    if (step.pose !== "stand") {
      const tilts = Object.values(capture).map((c) => c.tilt_from_stand_deg ?? 0);
      if (Math.max(...tilts) < MIN_BEND_DEG) {
        out.push("Hardly any bend was measured: bend further.");
      }
    }
    if (Object.keys(capture).length < 4) {
      out.push(`Only ${Object.keys(capture).length} of 4 sensors reported.`);
    }
    return out;
  };

  const next = () => {
    if (index + 1 < STEPS.length) {
      setIndex(index + 1);
      setCapture(null);
      setPhase("ready");
    } else {
      void runFit();
    }
  };

  const runFit = async () => {
    setPhase("fitting");
    const response = await post<FitReport>(`${apiBase}/mounting/fit`);
    if (!response.ok) {
      setProblem(response.data.detail ?? "Fit failed");
      setPhase("review");
      return;
    }
    setFit(response.data);
    setPhase("done");
  };

  const start = () => {
    setIndex(0);
    setCapture(null);
    setFit(null);
    setProblem(null);
    setPhase("ready");
  };

  const busy = phase === "capturing" || phase === "fitting";
  const stepIssues = issues();

  return (
    <div className="calib-backdrop">
      <div className="calib-dialog" role="dialog" aria-label="Sensor calibration">
        <div className="calib-head">
          <span className="calib-title">Sensor calibration</span>
          {!busy && <button className="calib-close" onClick={onClose} aria-label="Close">×</button>}
        </div>

        {phase === "intro" && (
          <>
            <p>
              Finds how each sensor sits on the body, from a few held poses: standing, then bending forward (needed),
              backward and to each side (recommended). Each pose has a {GET_READY_S} s countdown, then {HOLD_S} s of
              holding still.
            </p>
            <p className="calib-muted">
              The result is saved for this strip and survives restarts. Re-run it only if the sensors move on the vest;
              for small shifts use Re-zero upright.
            </p>
            <div className="calib-actions">
              <button onClick={onClose}>Cancel</button>
              <button className="primary" onClick={start}>Start</button>
            </div>
          </>
        )}

        {(phase === "ready" || phase === "capturing" || phase === "review") && (
          <>
            <div className="calib-progress">
              Step {index + 1} of {STEPS.length}
              {step.optional ? " · optional" : ""}
              <span className="calib-dots">
                {STEPS.map((s, i) => (
                  <span key={s.pose} className={i < index ? "done" : i === index ? "now" : ""} />
                ))}
              </span>
            </div>
            <div className="calib-step">{step.title}</div>
            <p>{step.prompt}</p>
            {phase === "ready" && <div className="calib-count ready">Get into position… {countdown}</div>}
            {phase === "capturing" && <div className="calib-count hold">Hold still… capturing</div>}
            {phase === "review" && (
              <>
                {capture && (
                  <table className="calib-table">
                    <thead>
                      <tr>
                        <th>Sensor</th>
                        <th>Still</th>
                        {step.pose !== "stand" && <th>Bend measured</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(capture).map(([region, c]) => (
                        <tr key={region}>
                          <td>{REGION_NAMES[region] ?? region}</td>
                          <td className={c.still ? "" : "calib-error"}>{c.still ? "yes" : `no (${c.spread_deg}°)`}</td>
                          {step.pose !== "stand" && <td>{c.tilt_from_stand_deg?.toFixed(0) ?? "—"}°</td>}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
                {stepIssues.map((issue) => (
                  <p key={issue} className="calib-error">{issue}</p>
                ))}
                {problem && <p className="calib-error">{problem}</p>}
                <div className="calib-actions">
                  <button onClick={() => setPhase("ready")}>Retry</button>
                  {step.optional && <button onClick={next}>Skip</button>}
                  <button className="primary" disabled={!capture} onClick={next}>
                    {index + 1 < STEPS.length ? "Next" : "Finish"}
                  </button>
                </div>
              </>
            )}
          </>
        )}

        {phase === "fitting" && <p>Fitting the sensor mountings…</p>}

        {phase === "done" && fit && (
          <>
            <p>Calibrated {fit.regions.length} of 4 sensors{fit.saved_path ? " and saved it for this strip" : ""}.</p>
            <table className="calib-table">
              <thead>
                <tr>
                  <th>Sensor</th>
                  <th>Direction fit</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(fit.report).map(([region, r]) => (
                  <tr key={region}>
                    <td>{REGION_NAMES[region] ?? region}</td>
                    <td className={r.error || (r.direction_error_deg ?? 0) > 25 ? "calib-error" : ""}>
                      {r.error ?? (r.direction_error_deg == null ? "—" : `${r.direction_error_deg.toFixed(0)}° off`)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="calib-muted">Under about 15° is good. A large value means a bend was mixed with twisting: repeat it.</p>
            <div className="calib-actions">
              <button onClick={start}>Repeat</button>
              <button className="primary" onClick={onClose}>Done</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
