// "Collect data": a guided, labelled recording session for later analysis. The backend records every
// sample the strip sends (quaternions and raw accel/gyro, at full rate); this wizard only prompts the
// wearer and tells the backend which movement is happening (label = "<step>#<rep>:<phase>").

import { useEffect, useMemo, useRef, useState } from "react";

interface Props {
  apiBase: string;
  clientMeta: () => Record<string, unknown>; // the view's calibration etc., stored in the file header
  onClose: () => void;
}

type PhaseName = "move" | "hold" | "return";

interface Movement {
  key: string;
  title: string;
  move: string; // prompt while getting into position
  hold: string; // prompt while holding
  lying?: boolean; // longer phases
}

const BENDS: Movement[] = [
  { key: "forward", title: "Bend forward", move: "Bend forward from the waist, as far as is comfortable.", hold: "Hold the forward bend." },
  { key: "backward", title: "Lean backward", move: "Lean back gently.", hold: "Hold the backward lean." },
  { key: "left", title: "Bend to your left", move: "Bend sideways to your LEFT.", hold: "Hold the left bend." },
  { key: "right", title: "Bend to your right", move: "Bend sideways to your RIGHT.", hold: "Hold the right bend." },
  { key: "twist_left", title: "Twist to your left", move: "Keep your hips still and turn your shoulders to the LEFT.", hold: "Hold the twist." },
  { key: "twist_right", title: "Twist to your right", move: "Keep your hips still and turn your shoulders to the RIGHT.", hold: "Hold the twist." },
  { key: "squat", title: "Squat", move: "Squat down with your back straight.", hold: "Hold the squat." },
];

const LYING: Movement[] = [
  { key: "lie_back", title: "Lie on your back", move: "Lie down on your back (face up).", hold: "Stay still, face up.", lying: true },
  { key: "lie_front", title: "Lie on your front", move: "Roll onto your front (face down).", hold: "Stay still, face down.", lying: true },
  { key: "lie_left", title: "Lie on your left side", move: "Roll onto your LEFT side.", hold: "Stay still on your left side.", lying: true },
  { key: "lie_right", title: "Lie on your right side", move: "Roll onto your RIGHT side.", hold: "Stay still on your right side.", lying: true },
];

const STAND: Movement = { key: "stand", title: "Stand upright", move: "Stand straight and relaxed, arms by your sides.", hold: "Stand still." };

// Seconds per phase.
const DURATION: Record<"bend" | "lying" | "stand", Record<PhaseName, number>> = {
  stand: { move: 3, hold: 6, return: 0 },
  bend: { move: 4, hold: 5, return: 4 },
  lying: { move: 8, hold: 8, return: 0 }, // the next lying step starts from lying; "get_up" ends the set
};

interface Step {
  movement: Movement;
  rep: number;
  phases: { name: PhaseName; seconds: number; prompt: string }[];
}

function buildSteps(reps: number, includeLying: boolean): Step[] {
  const step = (movement: Movement, rep: number, kind: keyof typeof DURATION): Step => ({
    movement,
    rep,
    phases: (["move", "hold", "return"] as PhaseName[])
      .filter((name) => DURATION[kind][name] > 0)
      .map((name) => ({
        name,
        seconds: DURATION[kind][name],
        prompt: name === "move" ? movement.move : name === "hold" ? movement.hold : "Come back to standing upright.",
      })),
  });
  const steps: Step[] = [step(STAND, 1, "stand")];
  for (let rep = 1; rep <= reps; rep += 1) {
    BENDS.forEach((m) => steps.push(step(m, rep, "bend")));
  }
  if (includeLying) {
    LYING.forEach((m) => steps.push(step(m, 1, "lying")));
    steps.push(
      step({ key: "get_up", title: "Get up", move: "Get up carefully and stand upright.", hold: "Stand still." }, 1, "bend"),
    );
  }
  steps.push(step({ ...STAND, key: "stand_end" }, 1, "stand"));
  return steps;
}

async function post(url: string, body?: string): Promise<{ ok: boolean; data: any }> {
  try {
    const response = await fetch(url, { method: "POST", body, headers: body ? { "Content-Type": "text/plain" } : undefined });
    const data = await response.json().catch(() => null);
    return { ok: response.ok, data };
  } catch {
    return { ok: false, data: { detail: "Backend unreachable" } };
  }
}

type Stage = "intro" | "running" | "saving" | "done";

export function DataCollectionWizard({ apiBase, clientMeta, onClose }: Props) {
  const [stage, setStage] = useState<Stage>("intro");
  const [name, setName] = useState("session");
  const [wearer, setWearer] = useState("");
  const [notes, setNotes] = useState("");
  const [reps, setReps] = useState(2);
  const [includeLying, setIncludeLying] = useState(true);
  const steps = useMemo(() => buildSteps(reps, includeLying), [reps, includeLying]);
  const [stepIndex, setStepIndex] = useState(0);
  const [phaseIndex, setPhaseIndex] = useState(0);
  const [remaining, setRemaining] = useState(0);
  const [samples, setSamples] = useState(0);
  const [problem, setProblem] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const advanceAt = useRef(0);

  const totalSeconds = steps.reduce((sum, s) => sum + s.phases.reduce((a, p) => a + p.seconds, 0), 0);
  const step = steps[stepIndex];
  const phase = step?.phases[phaseIndex];

  const label = (s: Step, p: { name: PhaseName }) => `${s.movement.key}#${s.rep}:${p.name}`;

  const start = async () => {
    setProblem(null);
    const meta = { ...clientMeta(), wearer, notes, reps, include_lying: includeLying, protocol: "collect-data-v1" };
    const response = await post(`${apiBase}/recording/start?name=${encodeURIComponent(name)}`, JSON.stringify(meta));
    if (!response.ok) {
      setProblem(response.data?.detail ?? "Could not start recording");
      return;
    }
    setStepIndex(0);
    setPhaseIndex(0);
    setStage("running");
  };

  const stop = async (aborted: boolean) => {
    setStage("saving");
    const response = await post(`${apiBase}/recording/stop?aborted=${aborted}`);
    if (!response.ok) {
      setProblem(response.data?.detail ?? "Could not save the recording");
    }
    setResult(response.ok ? response.data : null);
    setStage("done");
  };

  // Send the label when a phase starts and schedule the next phase.
  useEffect(() => {
    if (stage !== "running" || !step || !phase) {
      return;
    }
    void post(`${apiBase}/recording/label?name=${encodeURIComponent(label(step, phase))}`);
    advanceAt.current = performance.now() + phase.seconds * 1000;
    setRemaining(phase.seconds);
    const timer = window.setInterval(() => {
      const left = (advanceAt.current - performance.now()) / 1000;
      setRemaining(Math.max(0, Math.ceil(left)));
      if (left <= 0) {
        window.clearInterval(timer);
        next();
      }
    }, 100);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage, stepIndex, phaseIndex]);

  // Live sample count from the backend while recording.
  useEffect(() => {
    if (stage !== "running") {
      return;
    }
    const timer = window.setInterval(() => {
      fetch(`${apiBase}/recording/status`)
        .then((r) => r.json())
        .then((s) => setSamples(s.samples ?? 0))
        .catch(() => undefined);
    }, 1000);
    return () => window.clearInterval(timer);
  }, [stage, apiBase]);

  const next = () => {
    if (phaseIndex + 1 < step.phases.length) {
      setPhaseIndex(phaseIndex + 1);
    } else if (stepIndex + 1 < steps.length) {
      setStepIndex(stepIndex + 1);
      setPhaseIndex(0);
    } else {
      void stop(false);
    }
  };

  const skipStep = () => {
    void post(`${apiBase}/recording/label?name=${encodeURIComponent(`${step.movement.key}#${step.rep}:skipped`)}`);
    if (stepIndex + 1 < steps.length) {
      setStepIndex(stepIndex + 1);
      setPhaseIndex(0);
    } else {
      void stop(false);
    }
  };

  const phaseText = phase?.name === "move" ? "Get into position" : phase?.name === "hold" ? "Hold still" : "Return";

  return (
    <div className="calib-backdrop">
      <div className="calib-dialog" role="dialog" aria-label="Collect data">
        <div className="calib-head">
          <span className="calib-title">Collect data</span>
          {stage !== "running" && stage !== "saving" && (
            <button className="calib-close" onClick={onClose} aria-label="Close">×</button>
          )}
        </div>

        {stage === "intro" && (
          <>
            <p>
              Records everything the strip sends (orientations and raw accel/gyro) while the wearer follows the
              prompts: standing, bends in every direction, twists, a squat{includeLying ? ", and lying on the back, front and both sides" : ""}.
              The file is saved by the backend for later analysis.
            </p>
            <div className="collect-form">
              <label>Session name<input value={name} onChange={(e) => setName(e.target.value)} /></label>
              <label>Wearer (optional)<input value={wearer} placeholder="ID or initials" onChange={(e) => setWearer(e.target.value)} /></label>
              <label>Notes<input value={notes} placeholder="vest size, sensor placement, floor…" onChange={(e) => setNotes(e.target.value)} /></label>
              <label>
                Repeat each bend
                <select value={reps} onChange={(e) => setReps(Number(e.target.value))}>
                  {[1, 2, 3].map((n) => (
                    <option key={n} value={n}>{n}×</option>
                  ))}
                </select>
              </label>
              <label className="collect-check">
                <input type="checkbox" checked={includeLying} onChange={(e) => setIncludeLying(e.target.checked)} />
                Include lying down (needs a mat or clean floor)
              </label>
            </div>
            <p className="calib-muted">
              {steps.length} steps, about {Math.ceil(totalSeconds / 60)} min. Connect with "Quaternions + raw" to also
              record accel and gyro.
            </p>
            {problem && <p className="calib-error">{problem}</p>}
            <div className="calib-actions">
              <button onClick={onClose}>Cancel</button>
              <button className="primary" onClick={start}>Start recording</button>
            </div>
          </>
        )}

        {stage === "running" && step && phase && (
          <>
            <div className="calib-progress">
              <span className="collect-rec">● REC {samples.toLocaleString()} samples</span>
              <span>Step {stepIndex + 1} of {steps.length}</span>
            </div>
            <div className="calib-step">
              {step.movement.title}
              {step.rep > 1 || reps > 1 ? (step.movement.key === "stand" || step.movement.key === "stand_end" ? "" : ` · ${step.rep}/${reps}`) : ""}
            </div>
            <p>{phase.prompt}</p>
            <div className={`calib-count ${phase.name === "hold" ? "hold" : "ready"}`}>
              {phaseText}… {remaining}
            </div>
            <div className="collect-bar">
              <span style={{ width: `${(100 * (stepIndex + phaseIndex / step.phases.length)) / steps.length}%` }} />
            </div>
            <div className="calib-actions">
              <button onClick={() => void stop(true)}>Stop &amp; save</button>
              <button onClick={skipStep}>Skip step</button>
            </div>
          </>
        )}

        {stage === "saving" && <p>Saving…</p>}

        {stage === "done" && (
          <>
            {result ? (
              <>
                <p>
                  Saved {result.samples.toLocaleString()} samples ({Math.round(result.duration_s)} s)
                  {result.aborted ? ", stopped early" : ""}.
                </p>
                <p className="collect-path">{result.path}</p>
                {result.truncated && <p className="calib-error">The session hit the size limit and was cut short.</p>}
              </>
            ) : (
              <p className="calib-error">{problem ?? "Nothing was saved."}</p>
            )}
            <div className="calib-actions">
              <button
                onClick={() => {
                  setResult(null);
                  setStage("intro");
                }}
              >
                Record another
              </button>
              <button className="primary" onClick={onClose}>Done</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
