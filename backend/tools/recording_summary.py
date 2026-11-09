# File: recording_summary.py
# Purpose: Quick check of a "Collect data" recording: duration, sample rates per format, sensor validity,
#          and how many samples fall in each labelled step.
# Contributors: Unassigned
"""python tools/recording_summary.py recordings/<session>.jsonl"""
import collections
import json
import sys


def main(path):
    meta, end, labels, samples = None, None, [], []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            kind = row["type"]
            if kind == "meta":
                meta = row
            elif kind == "end":
                end = row
            elif kind == "label":
                labels.append((row["t"], row["name"]))
            else:
                samples.append(row)
    print(f"session {meta['session']}  source {meta['source']}  device format {meta['device_format']}")
    if meta.get("client"):
        client = meta["client"]
        print(f"wearer {client.get('wearer') or '-'}  notes {client.get('notes') or '-'}")
    duration = end["t"] if end else (samples[-1]["t"] if samples else 0)
    print(f"duration {duration:.1f} s  samples {len(samples)}  aborted {end and end['aborted']}  truncated {end and end['truncated']}")
    by_kind = collections.Counter((s["origin"], s["format"]) for s in samples)
    for (origin, fmt), count in sorted(by_kind.items()):
        valid = [sum(1 for s in samples if s["origin"] == origin and s["format"] == fmt and s["imus"][i]["valid"])
                 for i in range(4)]
        print(f"  {origin:13s} {fmt:10s} {count:7d} samples  {count / max(duration, 1e-9):6.1f} Hz  valid per sensor {valid}")
    # samples per label (label applies until the next label)
    counts = collections.Counter()
    times = [t for t, _ in labels]
    for s in samples:
        k = sum(1 for t in times if t <= s["t"]) - 1
        counts[labels[k][1] if k >= 0 else "(before first label)"] += 1
    print("samples per step:")
    for t, name in labels:
        print(f"  {t:7.1f} s  {name:28s} {counts[name]:6d}")


if __name__ == "__main__":
    main(sys.argv[1])
