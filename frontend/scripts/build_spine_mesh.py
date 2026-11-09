# File: build_spine_mesh.py
# Purpose: Build the spine view's anatomical vertebrae (sacrum..T1) from BodyParts3D into
#          public/models/spine/: one GLB with a mesh per vertebra in its own pivot frame, plus
#          spine.json with the column geometry (segment length, rest tilt, sensor pod point).
# Contributors: Unassigned
"""Offline, run once; the browser only loads the outputs (public/models/spine/vertebrae.glb and
src/lib/posture3d/vertebrae.json, which spine.ts imports).

    python scripts/build_spine_mesh.py [--cache DIR] [--faces 2400]

Needs numpy, trimesh and open3d (e.g. the sitescan venv). Downloads two BodyParts3D files
(PART-OF tree: partof_element_parts.txt and partof_BP3D_4.0_obj_99.zip, 62 MB) into --cache.

Source: BodyParts3D, (c) The Database Center for Life Science, CC BY 4.0
(https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html; older OBJ headers say CC BY-SA 2.1 JP).
Vertebra -> FMA concept IDs are taken from partof_parts_list_e.txt (listed in VERTEBRAE below).

Frames. BodyParts3D: millimetres, +X patient left, +Y posterior, +Z superior (checked: spinous
processes sit at high Y, lumbar bodies are anterior of the thoracic ones). Spine view: metres,
+X wearer right, +Y up, +Z posterior. Map: (x, y, z)_bp -> (-x, z, y) / 1000 (a proper rotation).

Column model (matches frontend/src/lib/posture3d/spine.ts): each vertebra pivots at the disc
below it. Pivot points are midpoints between neighbouring vertebral-body centres (the body is the
anterior part of the vertebra). Vertebra i spans pivot_i -> pivot_{i+1}; its length is that distance
and its rest tilt is the sagittal angle of that segment (positive leans the top posterior).
Each mesh is stored in its local frame: origin at pivot_i, segment along +Y.
"""
import argparse
import json
import math
import pathlib
import urllib.request
import zipfile

import numpy as np

BASE_URL = "https://dbarchive.biosciencedbc.jp/data/bodyparts3d/LATEST/"
ZIP_NAME = "partof_BP3D_4.0_obj_99.zip"
ELEMENTS_NAME = "partof_element_parts.txt"

# Bottom to top, same order and names as VERTEBRAE in spine.ts. FMA IDs from partof_parts_list_e.txt.
VERTEBRAE = [
    ("Sacrum", "FMA16202"), ("L5", "FMA13076"), ("L4", "FMA13075"), ("L3", "FMA13074"),
    ("L2", "FMA13073"), ("L1", "FMA13072"), ("T12", "FMA10081"), ("T11", "FMA10059"),
    ("T10", "FMA10037"), ("T9", "FMA10014"), ("T8", "FMA9991"), ("T7", "FMA9968"),
    ("T6", "FMA9945"), ("T5", "FMA9922"), ("T4", "FMA9248"), ("T3", "FMA9209"),
    ("T2", "FMA9187"), ("T1", "FMA9165"),
]
POD_GAP_M = 0.012       # sensor pod sits this far behind the most posterior bone (skin + strap)
ATTRIBUTION = "BodyParts3D, (c) The Database Center for Life Science licensed under CC Attribution 4.0 International"


def fetch(cache: pathlib.Path, name: str) -> pathlib.Path:
    path = cache / name
    if not path.exists():
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(BASE_URL + name, path)
    return path


def read_obj(text: str):
    verts, faces = [], []
    for line in text.splitlines():
        if line.startswith("v "):
            verts.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
            for k in range(1, len(idx) - 1):            # fan-triangulate polygons
                faces.append([idx[0] - 1, idx[k] - 1, idx[k + 1] - 1])
    return np.array(verts, float), np.array(faces, np.int64)


def to_view_frame(v_mm: np.ndarray) -> np.ndarray:
    return np.stack([-v_mm[:, 0], v_mm[:, 2], v_mm[:, 1]], axis=1) / 1000.0


def decimate(verts, faces, target):
    import open3d as o3d

    if len(faces) <= target:
        return verts, faces
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces))
    mesh.remove_duplicated_vertices()
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


def body_centre(v: np.ndarray) -> np.ndarray:
    """Centre of the vertebral body: the anterior 40 % of the bone along Z (+Z is posterior)."""
    z0, z1 = v[:, 2].min(), v[:, 2].max()
    body = v[v[:, 2] <= z0 + 0.4 * (z1 - z0)]
    return body.mean(axis=0)


def rot_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", default=str(pathlib.Path.home() / ".cache" / "bodyparts3d"))
    parser.add_argument("--faces", type=int, default=2400, help="max triangles per vertebra")
    parser.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent.parent / "public" / "models" / "spine"))
    args = parser.parse_args()
    cache = pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    elements = {}
    for line in fetch(cache, ELEMENTS_NAME).read_text(encoding="utf-8").splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) >= 3:
            elements.setdefault(cols[0], []).append(cols[2].strip())
    archive = zipfile.ZipFile(fetch(cache, ZIP_NAME))
    folder = ZIP_NAME[:-4]

    meshes = []
    for name, fma in VERTEBRAE:
        parts = elements.get(fma)
        if not parts:
            raise SystemExit(f"{name}: no element files for {fma}")
        vs, fs, offset = [], [], 0
        for fj in parts:
            v, f = read_obj(archive.read(f"{folder}/{fj}.obj").decode("utf-8", "replace"))
            vs.append(v)
            fs.append(f + offset)
            offset += len(v)
        v = to_view_frame(np.concatenate(vs))
        f = np.concatenate(fs)
        v, f = decimate(v, f, args.faces)
        meshes.append((name, fma, parts, v, f))

    centres = [body_centre(v) for _, _, _, v, _ in meshes]
    pivots = [None] * (len(meshes) + 1)
    for i in range(1, len(meshes)):
        pivots[i] = (centres[i - 1] + centres[i]) / 2
    pivots[0] = centres[0] - (pivots[1] - centres[0])        # sacrum: mirror its top pivot through its body
    pivots[-1] = centres[-1] + (centres[-1] - pivots[-2])     # above T1: same spacing as below it

    import trimesh

    scene = trimesh.Scene()
    column = []
    for i, (name, fma, parts, v, f) in enumerate(meshes):
        seg = pivots[i + 1] - pivots[i]
        length = float(np.linalg.norm(seg))
        tilt = math.atan2(seg[2], seg[1])                       # rotation about +X taking +Y onto seg
        local = (rot_x(-tilt) @ (v - pivots[i]).T).T
        # Sensor pod: on the skin behind the most posterior bone, at the height of that bone.
        back = local[np.argmax(local[:, 2])]
        pod = [0.0, float(back[1]), float(local[:, 2].max() + POD_GAP_M)]
        mesh = trimesh.Trimesh(vertices=local.astype(np.float32), faces=f, process=True)
        if mesh.volume < 0:                                     # make normals point outward
            mesh.invert()
        scene.add_geometry(mesh, node_name=name, geom_name=name)
        width = float(np.ptp(local[:, 0]))
        depth = float(np.ptp(local[:, 2]))
        column.append({"name": name, "fma": fma, "elements": parts, "height": round(length, 5),
                       "restTiltDeg": round(math.degrees(tilt), 2), "width": round(width, 4),
                       "depth": round(depth, 4), "pod": [round(c, 4) for c in pod],
                       "triangles": int(len(mesh.faces))})

    glb = out / "vertebrae.glb"
    glb.write_bytes(scene.export(file_type="glb", include_normals=True))   # smooth shading
    meta = {"source": "BodyParts3D 4.0, PART-OF tree, 99 % reduced OBJ, further decimated",
            "attribution": ATTRIBUTION, "license": "CC BY 4.0",
            "frame": "metres; +X right, +Y up, +Z posterior; each mesh local to its pivot (disc below)",
            "vertebrae": column}
    (out / "spine.json").write_text(json.dumps(meta, indent=2))
    # The spine maths (src/lib/posture3d/spine.ts) imports the same column at build time.
    column_ts = pathlib.Path(__file__).resolve().parent.parent / "src" / "lib" / "posture3d" / "vertebrae.json"
    column_ts.write_text(json.dumps({"attribution": ATTRIBUTION, "vertebrae": [
        {k: c[k] for k in ("name", "height", "width", "depth", "restTiltDeg", "pod")} for c in column]}, indent=2))
    total = sum(c["triangles"] for c in column)
    print(f"wrote {glb} ({glb.stat().st_size / 1e6:.2f} MB, {total} triangles) and spine.json")
    print(f"column length {sum(c['height'] for c in column):.3f} m")
    for c in column:
        print(f"  {c['name']:6s} h {c['height']*1000:5.1f} mm  tilt {c['restTiltDeg']:6.1f} deg  pod {c['pod']}")


if __name__ == "__main__":
    main()
