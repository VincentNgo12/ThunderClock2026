// Guided mounting calibration: prompts the wearer through upright, forward, backward, right and left
// bends, averages each held pose per sensor, then solves every sensor's heading (and mirror) so the
// bends line up with the wearer frame. Nothing changes until the user presses Apply.

import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import {
  averageQuaternion,
  MIN_BEND_DEG,
  rotationVector,
  solveMounting,
  type MountingSet,
  type MovementName,
  type SensorFit,
} from "../../lib/posture3d/mounting";
import { REGION_LABELS, REGIONS, type Region } from "../../telemetry/types";

type Raw = Partial<Record<Region, THREE.Quaternion>>;

interface Props {
  getRaw: () => Raw; // latest raw sensor quaternions (sensor world frame)
  current: MountingSet;
  onApply: (neutral: Record<Region, THREE.Quaternion>, mountings: MountingSet) => void;
  onClose: () => void;
}

interface Step {
  key: "neutral" | MovementName;
  title: string;
  prompt: string;
}

const STEPS: Step[] = [
  { key: "neutral", title: "Stand upright", prompt: "Stand straight and relaxed, arms by your sides. Stay still." },
  { key: "forward", title: "Bend forward", prompt: "Bend forward from the waist (about 45°) and hold it." },
  { key: "backward", title: "Lean backward", prompt: "Stand up, then lean back gently (about 20°) and hold it." },
  { key: "right", title: "Bend to your right", prompt: "Stand up, then bend sideways to your RIGHT (about 30°) and hold it." },
  { key: "left", title: "Bend to your left", prompt: "Stand up, then bend sideways to your LEFT (about 30°) and hold it." },
];

const GET_READY_S = 4; // time to get into position
const HOLD_S = 2; // samples averaged while holding
const SAMPLE_MS = 50;

type Phase = "intro" | "ready" | "hold" | "done";

export function CalibrationWizard({ getRaw, current, onApply, onClose }: Props) {
  const [phase, setPhase] = useState<Phase>("intro");
  const [stepIndex, setStepIndex] = useState(0);
  const [countdown, setCountdown] = useState(GET_READY_S);
  const [liveBend, setLiveBend] = useState(0);
  const [problem, setProblem] = useState<string | null>(null);
  const captures = useRef<Partial<Record<Step["key"], Raw>>>({});
  const samples = useRef<Record<Region, THREE.Quaternion[]>>(emptySamples());
  const [fits, setFits] = useState<Record<Region, SensorFit> | null>(null);

  const step = STEPS[stepIndex];

  // Countdown, then hold-and-sample, then the next step.
  useEffect(() => {
    if (phase !== "ready" && phase !== "hold") {
      return;
    }
    const started = performance.now();
    const duration = (phase === "ready" ? GET_READY_S : HOLD_S) * 1000;
    if (phase === "hold") {
      samples.current = emptySamples();
    }
    const timer = window.setInterval(() => {
      const elapsed = performance.now() - started;
      const raw = getRaw();
      const neutral = captures.current.neutral;
      if (neutral) {
        // Live feedback: the largest rotation away from upright over all sensors.
        let max = 0;
        for (const region of REGIONS) {
          const q = raw[region];
          const q0 = neutral[region];
          if (q && q0) {
            max = Math.max(max, THREE.MathUtils.radToDeg(rotationVector(q.clone().multiply(q0.clone().invert())).length()));
          }
        }
        setLiveBend(max);
      }
      if (phase === "hold") {
        for (const region of REGIONS) {
          const q = raw[region];
          if (q) {
            samples.current[region].push(q.clone());
          }
        }
      }
      setCountdown(Math.max(0, Math.ceil((duration - elapsed) / 1000)));
      if (elapsed >= duration) {
        window.clearInterval(timer);
        if (phase === "ready") {
          setPhase("hold");
          return;
        }
        finishStep();
      }
    }, SAMPLE_MS);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, stepIndex]);

  const finishStep = () => {
    const held: Raw = {};
    for (const region of REGIONS) {
      const mean = averageQuaternion(samples.current[region]);
      if (mean) {
        held[region] = mean;
      }
    }
    if (step.key === "neutral" && Object.keys(held).length === 0) {
      setProblem("No sensor data while standing. Check that the strip is connected, then retry.");
      setPhase("intro");
      return;
    }
    captures.current[step.key] = held;
    if (stepIndex + 1 < STEPS.length) {
      setStepIndex(stepIndex + 1);
      setCountdown(GET_READY_S);
      setPhase("ready");
    } else {
      solve();
    }
  };

  const solve = () => {
    const neutral = captures.current.neutral!;
    const result = {} as Record<Region, SensorFit>;
    for (const region of REGIONS) {
      const q0 = neutral[region];
      const deltas: Partial<Record<MovementName, THREE.Quaternion>> = {};
      for (const key of ["forward", "backward", "right", "left"] as MovementName[]) {
        const q = captures.current[key]?.[region];
        if (q && q0) {
          deltas[key] = q.clone().multiply(q0.clone().invert());
        }
      }
      result[region] = solveMounting(deltas);
    }
    setFits(result);
    setPhase("done");
  };

  const start = () => {
    captures.current = {};
    setProblem(null);
    setFits(null);
    setStepIndex(0);
    setCountdown(GET_READY_S);
    setPhase("ready");
  };

  const apply = () => {
    if (!fits) {
      return;
    }
    const neutral = captures.current.neutral!;
    const mountings = { ...current };
    const neutralOut = {} as Record<Region, THREE.Quaternion>;
    for (const region of REGIONS) {
      const fit = fits[region];
      if (fit.mounting) {
        mountings[region] = fit.mounting;
      }
      if (neutral[region]) {
        neutralOut[region] = neutral[region]!;
      }
    }
    onApply(neutralOut, mountings);
    onClose();
  };

  return (
    <div className="calib-backdrop">
      <div className="calib-dialog" role="dialog" aria-label="Sensor calibration">
        <div className="calib-head">
          <span className="calib-title">Sensor calibration</span>
          <button className="calib-close" onClick={onClose} aria-label="Close">×</button>
        </div>

        {phase === "intro" && (
          <>
            <p>
              Wear the strip and connect it. You will be asked to hold five poses: upright, forward, backward,
              right and left. Each pose has a {GET_READY_S} s countdown to get into position, then {HOLD_S} s of holding still.
            </p>
            <p className="calib-muted">
              This finds each sensor's heading (and a mirrored axis, if any) so bends show up the right way.
              It also sets the neutral. Re-run it if the spine starts to drift.
            </p>
            {problem && <p className="calib-error">{problem}</p>}
            <div className="calib-actions">
              <button onClick={onClose}>Cancel</button>
              <button className="primary" onClick={start}>Start</button>
            </div>
          </>
        )}

        {(phase === "ready" || phase === "hold") && (
          <>
            <div className="calib-progress">
              Step {stepIndex + 1} of {STEPS.length}
              <span className="calib-dots">
                {STEPS.map((s, i) => (
                  <span key={s.key} className={i < stepIndex ? "done" : i === stepIndex ? "now" : ""} />
                ))}
              </span>
            </div>
            <div className="calib-step">{step.title}</div>
            <p>{step.prompt}</p>
            <div className={`calib-count ${phase}`}>
              {phase === "ready" ? `Get into position… ${countdown}` : `Hold still… ${countdown}`}
            </div>
            {step.key !== "neutral" && (
              <div className={`calib-bend ${liveBend >= MIN_BEND_DEG * 2 ? "ok" : ""}`}>
                Current bend: {Math.round(liveBend)}°
              </div>
            )}
            <div className="calib-actions">
              <button onClick={() => setPhase("intro")}>Restart</button>
            </div>
          </>
        )}

        {phase === "done" && fits && (
          <>
            <p className="calib-muted">
              Heading found per sensor. Fit error is the angle between the corrected bend axes and the expected
              ones (lower is better; under ~15° is good).
            </p>
            <table className="calib-table">
              <thead>
                <tr>
                  <th>Sensor</th>
                  <th>Yaw</th>
                  <th>Mirror</th>
                  <th>Fit error</th>
                </tr>
              </thead>
              <tbody>
                {REGIONS.map((region) => {
                  const fit = fits[region];
                  const bad = fit.meanErrorDeg !== null && fit.meanErrorDeg > 25;
                  return (
                    <tr key={region}>
                      <td>{REGION_LABELS[region]}</td>
                      <td>{fit.mounting ? `${fit.mounting.yaw.toFixed(0)}°` : "—"}</td>
                      <td>{fit.mounting ? (fit.mirrored ? "yes" : "no") : "—"}</td>
                      <td className={bad ? "calib-error" : ""}>
                        {fit.mounting
                          ? fit.meanErrorDeg !== null
                            ? `${fit.meanErrorDeg.toFixed(0)}°`
                            : "—"
                          : "Too little movement: kept old setting"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {REGIONS.some((r) => (fits[r].meanErrorDeg ?? 0) > 25) && (
              <p className="calib-error">
                A large fit error usually means the bends were mixed with twisting, or that sensor's axes are
                unusual. Repeat, or set its axis remap by hand.
              </p>
            )}
            <div className="calib-actions">
              <button onClick={start}>Repeat</button>
              <button className="primary" onClick={apply}>Apply</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function emptySamples(): Record<Region, THREE.Quaternion[]> {
  return Object.fromEntries(REGIONS.map((r) => [r, []])) as unknown as Record<Region, THREE.Quaternion[]>;
}
