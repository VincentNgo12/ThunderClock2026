// Manual per-sensor frame offsets: an axis remap (which sensor axis is the wearer's right / up /
// back, with sign) and a rotational offset (yaw, pitch, roll in degrees). Changes apply live.

import { IDENTITY_MOUNTING, REMAP_OPTIONS, type Mounting, type MountingSet } from "../../lib/posture3d/mounting";
import { REGION_LABELS, REGIONS, type Region } from "../../telemetry/types";

interface Props {
  mountings: MountingSet;
  onChange: (mountings: MountingSet) => void;
  onCalibrate: () => void;
  onClose: () => void;
}

const ANGLES: { key: "yaw" | "pitch" | "roll"; label: string; title: string }[] = [
  { key: "yaw", label: "Yaw", title: "Degrees about up (+Y): fixes heading" },
  { key: "pitch", label: "Pitch", title: "Degrees about right (+X)" },
  { key: "roll", label: "Roll", title: "Degrees about back (+Z)" },
];

export function MountingPanel({ mountings, onChange, onCalibrate, onClose }: Props) {
  const update = (region: Region, patch: Partial<Mounting>) =>
    onChange({ ...mountings, [region]: { ...mountings[region], ...patch } });

  return (
    <aside className="mount-panel">
      <div className="mount-head">
        <span className="mount-title">Sensor axes and offsets</span>
        <button className="calib-close" onClick={onClose} aria-label="Close">×</button>
      </div>
      <p className="mount-help">
        Remap: wearer right, up, back as sensor axes (e.g. "-y,z,-x"). Offsets rotate the sensor's frame
        after the remap. Changes apply live.
      </p>
      <div className="mount-grid">
        <span />
        <span className="mount-col">Remap</span>
        {ANGLES.map((a) => (
          <span key={a.key} className="mount-col" title={a.title}>{a.label}°</span>
        ))}
        <span />
        {REGIONS.map((region) => {
          const m = mountings[region];
          return (
            <div className="mount-row" key={region}>
              <span className="mount-name">{REGION_LABELS[region]}</span>
              <select value={m.remap} onChange={(event) => update(region, { remap: event.target.value })}>
                {REMAP_OPTIONS.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
              {ANGLES.map((a) => (
                <input
                  key={a.key}
                  type="number"
                  step={1}
                  min={-180}
                  max={180}
                  value={m[a.key]}
                  title={a.title}
                  onChange={(event) => {
                    const value = Number(event.target.value);
                    if (Number.isFinite(value)) {
                      update(region, { [a.key]: value });
                    }
                  }}
                />
              ))}
              <button title="Reset this sensor" onClick={() => update(region, { ...IDENTITY_MOUNTING })}>↺</button>
            </div>
          );
        })}
      </div>
      <div className="mount-actions">
        <button
          onClick={() =>
            onChange(Object.fromEntries(REGIONS.map((r) => [r, { ...IDENTITY_MOUNTING }])) as MountingSet)
          }
        >
          Reset all
        </button>
        <button className="primary" onClick={onCalibrate}>Calibrate…</button>
      </div>
    </aside>
  );
}
