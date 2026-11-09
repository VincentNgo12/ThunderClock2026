# anny_worker: ThunderSim worker body from an Anny body

`tools/anny_body.py` builds an Anny body for a given phenotype. It measures what ThunderSim's `Worker`
(`detection/simulation/thundersim/spine.py`) needs and writes one JSON file.
`anny.py` (`thundersim.bodies.worker_from_anny`) turns that JSON into `Worker(**kwargs)`. It uses only the standard library and
does not import anny.

## Run

```
PY=/path/to/anny-venv/bin/python   # offline only (anny 0.6.2, torch CPU, scipy); never the simulator .venv
$PY tools/anny_body.py --preset avg_male   --out anny_avg_male.json
$PY tools/anny_body.py --gender 1 --age-years 35 --muscle 0.5 --weight 0.5 --proportions 0.5 \
    --height-m 1.60 --out anny_female_160.json
```

- Phenotype flags: `--gender` (0 = male, 1 = female), `--age` (Anny parameter) or `--age-years`,
  `--muscle`, `--weight`, `--proportions`, and `--height` (Anny parameter) or `--height-m`.
- `--height-m` finds the height parameter by 40-step bisection on the rest crown-to-sole height,
  using the same method as `anny/extract.py`.
- Flags override a `--preset`. Phenotype values are rounded to 4 decimals.
- `avg_male` is the `extract.py` body: gender 0, 35 y, muscle/weight/proportions 0.5, 1.765 m.
- Optional: `--voxel` (default 0.005 m) and `--knn` (default 8) set the mass partition.
- A run takes about 5 s. Output is deterministic: two reruns gave byte-identical files.

In the sim: `Worker(**worker_from_anny(path, load_kg=...))`. This was checked against the real
`Worker` and `strip_kinematics`. The chip z-axes at neutral equal the Anny pod normals exactly.

## Frame

All positions use the sim frame of spine.py:

- metres, +X wearer-right, +Y up, +Z posterior, so the body faces -Z;
- soles (lowest rest vertex) at y = 0;
- x = 0 on the mid-sagittal plane;
- z = 0 at the S1 spinal-midline point, so the sim's `root_pos` is (0, pelvis_height, 0).

The conversion from Anny's native frame (Z up, faces -Y, left +X) is `p_sim = M p_anny + t`, with
`M = [[-1,0,0],[0,0,1],[0,1,0]] = Ry(180) Rx(-90)`. `Rx(-90)` is extract.py's Y-up map, and
`det M = +1`. `t` is stored in `frame.from_anny_native`. The pose is Anny's rest (MakeHuman A-pose).

## Fields

| key | definition |
|---|---|
| `height_m` | Rest crown-to-sole height (`Anthropometry.height`). |
| `total_mass_kg` | Volume of the whole closed mesh × 980 kg/m³. 980 is Anny's own constant (`anny/anthropometry.py`, "for simplicity"). Real whole-body density is about 1010–1070, so the true mass could be 3–9 % higher. |
| `bone_heads` | All 104 Anny bones, rest pose, sim frame. |
| `spinal_midline` | Polyline through the bone heads root → spine04 → spine03 → spine02 → spine01 → neck01..03 → head, at x = 0. These bones lie 3–5 cm anterior to the back skin. |
| `spinous_anchors.levels.{S1,L3,T10,T2}` | See "Spinous anchors" below. |
| `sensor_positions` | `midline(y_k) + 0.04 m · (0,0,+1)`. "Behind" means horizontally along the sim's posterior +Z axis in the standing rest pose. The *offset* equals spine.py's `seg_i.apply([0,0,sensor_offset_m])` at neutral, but the *base points* do not: spine.py's neutral spine line is straight (see `sim_neutral_check`). The file also gives the gap to the skin along +Z and the exact signed distance (+ = outside the skin). |
| `sim_neutral_check` | Where spine.py actually puts the anchors at neutral (pitch = flex = 0) minus `sensor_positions`, and a `curved_neutral_alternative`: a stand pose (`pitch_rad`, `flex_rad`) plus a matching `neutral_curvature_deg` that puts the sim's spine line exactly on Anny's midline. That alternative lives in the PoseTrack, not in `Worker`. |
| `thunder_links` | spine.py rule: `spine_line[i+1] = spine_line[i] + spacing[i]·up_i`. `pelvis` = S1→L3, `lumbar` = L3→T10, `thoracic` = T10→T2. `length_m` is the straight midline chord. Also given: skin chord, skin arc (tape measure, 2 mm profile) and vertical separation. `upper_thoracic` (T2→C7, the neck01 head) is for information only; the sim gives that segment no length. |
| `region_masses` | See "Region masses" below. |
| `worker_fields` | Exactly the `Worker` constructor fields: `height_m, mass_kg, load_kg, spacing_m, neutral_curvature_deg, mount_error_deg, sensor_offset_m`. |
| `worker_fields_not_derived` | `load_kg` and `mount_error_deg` keep the sim defaults (8.0, 6.0); they are not body properties. `sensor_offset_m` is deliberately kept at 0.04. |
| `worker_fields_alternatives` | Anny-measured midline-to-skin depths, skin-chord, arc and vertical spacings. |
| `worker_properties_not_settable` | `pelvis_height` (sim: 0.53 H) and falls.py `r_p` (COM to pelvis anchor, sim: (0, -0.03 H, 0.10)), each with the Anny value. The kwargs cannot change these. |

**Spinous anchors.** These follow `anny_lift_spec.md` §5 / D7:

- Level height: `y = hip_y + (y_tip_BP3D − 0.8915)·s`, where `s = (C7_y − hip_y)/(1.457 − 0.8915)`.
  - `hip_y` is the mean height of the upperleg01 heads.
  - `C7_y` is the height of the neck01 head.
  - The BodyParts3D tip heights are S1 0.9708, L3 1.0560, T10 1.1836, T2 1.4093 m.
- `skin`: the farthest hit of a horizontal +Z ray at x = 0.
- `normal`: the mean area-weighted vertex normal over the 34 × 44 mm pod footprint.
- `neutral_pitch_deg`: `atan2(n_y, n_z)`.
- `pod_centre`: skin + 8 mm along the normal (12 mm pod, 2 mm stand-off).

**Region masses.** The body interior is voxelised at 5 mm.

- Each voxel takes the inverse-distance blend of the full skin weights of its 8 nearest skin
  vertices. Bone weights are summed into regions, and the volume fractions are multiplied by the
  total mass.
- Bone-to-region mapping:
  - pelvis: root, pelvis.L/R, spine05
  - lumbar: spine04–03
  - thoracic: spine02
  - upper-thoracic: spine01, clavicles, shoulder01
  - the remaining regions follow the bone names.
- The trunk is also split by horizontal planes at L3, T10 and T2, which match the sim's link
  boundaries.
- `de_leva_1996_check` holds the de Leva fractions (interpolated by gender) next to the Anny values.
- The whole-body centre of mass is included.

## Results (rest pose)

| | avg_male | female_160 |
|---|---|---|
| phenotype (g, age, mus, wt, h, prop) | 0, .7922, .5, .5, .3789, .5 | 1, .7922, .5, .5, .3497, .5 |
| height / mass / BMI | 1.7649 m / 69.62 kg / 22.35 | 1.6000 m / 48.68 kg / 19.01 |
| S1 / L3 / T10 / T2 level y (m) | 1.0144 / 1.1022 / 1.2337 / 1.4663 | 0.9058 / 0.9842 / 1.1017 / 1.3095 |
| `spacing_m` (midline chords) | 0.0878 / 0.1341 / 0.2366 | 0.0793 / 0.1181 / 0.2079 |
| sim default 0.07/0.09/0.11·H | 0.1236 / 0.1589 / 0.1941 | 0.1120 / 0.1440 / 0.1760 |
| skin arc (tape) | 0.0899 / 0.1337 / 0.2387 (= spec D7) | see JSON |
| `neutral_curvature_deg` (sim −12/−6/10/18) | 18.5 / −6.0 / −4.8 / 21.4 | 21.9 / 15.0 / −4.1 / 16.7 |
| midline→skin depth S1/L3/T10/T2 (mm) | 50 / 36 / 32 / 41 | 43 / 26 / 36 / 47 |
| 0.04 m sensor vs skin, signed (mm) | −9.7 / +3.5 / +7.8 / −1.3 | −2.6 / +13.4 / +3.9 / −6.9 |
| pelvis_height: sim 0.53 H vs Anny S1 | 0.935 vs 1.014 | 0.848 vs 0.906 |

**Mass fractions, avg_male, % body mass (Anny vs de Leva male):**

| segment | Anny | de Leva |
|---|---|---|
| head + neck | 6.94 | 6.94 |
| trunk | 46.3 | 43.5 |
| upper arm (each) | 3.72 | 2.71 |
| forearm | 1.36 | 1.62 |
| hand | 0.51 | 0.61 |
| thigh | 11.4 | 14.2 |
| shank | 4.92 | 4.33 |
| foot | 1.51 | 1.37 |

Uniform density and skin-weight boundaries move mass from the thighs into the trunk (through the
pelvis and buttock weights) and from the deltoids into the upper arms. For per-segment dynamics,
prefer de Leva. Both are stored in the JSON.

## Caveats

- **S1 sensor inside the skin.** With the 0.04 m offset, the male S1 sensor is about 1 cm inside the
  skin, because S1 skin is 5 cm behind the rig's spine line. The mean midline-to-skin depth over the
  four levels is 0.040 m, so 0.04 is right on average.
- **Straight neutral spine in the sim.** spine.py's neutral spine is one vertical line; Anny's is
  curved. With the default `worker_fields`, the sim's neutral anchors sit off `sensor_positions`
  front-to-back by 0 / −1.5 / −27.7 / +15.4 mm (avg_male, S1/L3/T10/T2) and 0 / +11.4 / +23.6 /
  +30.0 mm (female_160), and up to 6.6 mm in height (chords are longer than vertical spacing).
  Sensor *orientations* are unaffected: `neutral_curvature_deg` is measured against vertical
  segments, so chip z = pod normal. To reproduce Anny's positions as well, use
  `sim_neutral_check.curved_neutral_alternative`. Its stand pitch/flex (male segment leans −1.0 /
  −11.3 / +10.5 / +18.5°) together with its own curvature (19.4 / 5.3 / −15.4 / 2.9°) put the spine
  line exactly on Anny's midline. Sensors then sit 0.04 m along each tilted segment's posterior axis,
  which is 0.7–12.7 mm from `sensor_positions`. That needs a neutral offset in the scenario's
  PoseTrack, not a Worker field.
- **Spacing definitions.** `spacing_m` is the straight midline chord. Worker's comment calls
  spacings "surface distances" (tape-measured); `spacing_m_skin_arc_tape` is that reading and
  differs by ≤ 4.1 mm. `spacing_m_vertical` reproduces Anny's level heights exactly on the
  straight neutral spine.
- **Female levels.** The levels come from a male BodyParts3D skeleton scaled hip-to-C7. On the
  female body, L3 lands on the upper sacral/gluteal slope, which gives a +15° pitch. Female levels
  are approximate.
- **pelvis_height.** `Worker.pelvis_height` is a property (0.53 H), so a sim that wants Anny's S1
  height must change spine.py or subclass `Worker`.

## Sources and licence

- **Anny.** Code © 2025 NAVER Corp., Apache-2.0 (https://github.com/naver/anny). Citation: Brégier et
  al., "Human Mesh Modeling for Anny Body", arXiv:2511.03589. Ship a NOTICE with the Apache
  attribution.
- **Anny assets.** Mesh, blend shapes, rig and skin weights derived from MakeHuman / MPFB2, CC0 1.0.
- **BodyParts3D.** Spinous-tip heights from BodyParts3D, © The Database Center for Life Science,
  CC BY 4.0 (attribution required). Values come from `insite/anatomy-bp3d.json`
  `landmarks.spine[].tip`.
- **de Leva.** de Leva P., "Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters",
  J Biomech 29(9):1223–1230, 1996, Table 4.
- **Pod geometry and level mapping.** `anny_lift_spec.md` §5 / D7.
