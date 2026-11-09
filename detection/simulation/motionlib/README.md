# motionlib: real human motion for the ThunderSim viewer

Real motion-capture clips for two uses:

1. **Trunk check (simulator step 1).** Compare the simulator's trunk (pelvis, lumbar,
   thoracic and upper-thoracic segments) against real data, in the simulator's own segment convention.
2. **Viewer limbs (stage S6).** Drive the viewer's legs and arms from real clips for voluntary
   activities (`thundersim/viz/mocap.py`). The simulated spine stays the IMU ground truth and is never
   changed by a clip.

Mocap in the viewer is **off by default** (`thundersim viz --mocap` or `THUNDERSIM_MOCAP=1`). It stays off
until every gate passes and a person has reviewed the clips (`reviewed_by` on the card).

## Layout

| Path | What | In git |
|------|------|--------|
| `sources.json` | every downloaded file: explicit URL, sha256, bytes, tags, licence per source | yes |
| `raw/<source>/` | downloads, one folder per source; parsed, never executed | no (`.gitignore`) |
| `canonical/<clip>.npz` | stage S2 clips: 20 joints, metres, +Y up, 100 Hz, plus trunk segment orientations | yes (class A only) |
| `canonical/index.json`, `references.json` | clip index; each performer's neutral stand used for calibration | yes |
| `annotations/<clip>.npz` | stage S3: floor plane, heel/toe contacts, rung contacts, kneeling, gait phase, activity per frame | yes |
| `clips/<clip>.npz` | stage S4: limb targets for the rig (float32), see `tools/features.py` | yes (class A only) |
| `cards/<clip>.json` | data card per clip: source, sha256, licence class, tags, joint map, checks (`s2_checks`), annotation summary (`s3`), features (`s4`), gate metrics (`gates`), status | yes |
| `coverage.json` | each Builder primitive (with lift techniques) and rig mode mapped to clips, gated clips or a gap; the viewer's mocap roles | yes |
| `manifest.json` | stage S6: the clips the viewer may use, their sha256 and anchors per role (the viewer reads only this file and those clips) | yes |
| `s6_gates.json` | stage S6: gate results on the real payload per clip and role, with the procedural baseline | yes |
| `LICENCES.md` | credits for every source used | yes |
| `tools/` | our own readers and tools (numpy/scipy, standard library for downloads) | yes |
| `restricted/`, `approvals/` | class B outputs and approvals (none so far) | `restricted/` no |

## Commands

Run from `detection/simulation`. Tools that read downloads run with `-I`
(isolated mode: nothing from the current folder or `raw/` can be imported).

```
.venv/bin/python -I motionlib/tools/acquire.py verify       # re-hash every download against sources.json
.venv/bin/python -I motionlib/tools/acquire.py fetch        # S1: download what is missing (class A only)
.venv/bin/python -I motionlib/tools/canonical.py            # S2: every clip -> canonical/ and cards/
.venv/bin/python motionlib/tools/check_canonical.py         # S2 checks, written into the cards
.venv/bin/python motionlib/tools/annotate.py                # S3: contacts, events, activity tags -> annotations/, cards
.venv/bin/python motionlib/tools/features.py                # S4: limb targets -> clips/
.venv/bin/python motionlib/tools/gates.py                   # S5: gate metrics -> cards; any FAIL -> status rejected
.venv/bin/python motionlib/tools/manifest.py                # S6: manifest.json from the cards (clips the viewer may use)
.venv/bin/python motionlib/tools/payload_gates.py           # S6: gates on the real payload -> cards (s6), s6_gates.json, manifest
.venv/bin/python motionlib/tools/coverage.py                # coverage.json from the cards and the manifest
.venv/bin/python motionlib/tools/check_invariants.py        # payload ground truth identical with mocap on/off
```

## Canonical clip format

`canonical/<clip>.npz`, float32 unless noted. The full field list is in the docstring of
`tools/canonical.py`. The main fields:

- `joints (N,20,3)`: the viewer's 18 `rig.JOINTS` plus `pelvis` (hip-centre midpoint) and `head_top`,
  in metres, +Y up, at 100 Hz. `valid (20,)` is False where a source has no such joint (values are NaN).
- `seg_q (N,4,4)`: pelvis, lumbar, thoracic and upper-thoracic segment -> world quaternions (x,y,z,w)
  in the simulator convention (+X wearer-right, +Y up the spine, +Z posterior). They are calibrated like the
  simulator's calibration stand: in the neutral stand every segment reads `Ry(heading)`.
- `trunk_angles (N,12)`: `seg_q` decomposed exactly like `spine.root_rotation` and `spine.joint_rotation`:
  heading, pitch, roll, then flex, lat and twist for the three joints (radians). `check_canonical.py`
  rebuilds `seg_q` from these with the simulator's own functions (error below 1e-3 degrees).
- `seg_q_chord`, `trunk_angles_chord`: the same from the geometric spine line instead of the source bones.
- `trunk_anchors (N,4,3)`: S1, L3, T10 and T2 points along the source spine line.

### How the trunk segments are made

- **Skeleton sources (CMU, ACCAD, 100STYLE).** Each anchor takes the rotation of the rigid source bone
  under it. Clips are resampled by slerping joint rotations and rebuilding positions, so bone lengths
  stay exact.
- **Position sources (CML, NIOSH).** Segment frames come from the spine line, with the lateral axis
  blended from the hip line to the shoulder line. This is coarse: three spine points only. [design]
- **Neutral stand.** The neutral stand is the clip's stillest upright 0.5 s. If a clip has none (walking,
  lying), it borrows the neutral stand of another clip by the same performer with the same skeleton
  (`references.json`). The card records this under `trunk.neutral_reference`.
- **NIOSH roofing trials.** The workers kneel throughout, so these clips have no neutral stand. Their
  `trunk_angles` are relative to the most upright kneeling window. The positions are pelvis-relative
  (no floor), so use them for trunk shape only.

## Licence classes

These are summarised from `.claude/agents/mocap-curator.md`.

- **A (open).** CMU, ACCAD (CC BY 3.0), 100STYLE (CC BY 4.0), CML public release (CC BY 4.0) and NIOSH
  (public domain). These are the only sources used here.
- **B (restricted sharing).** Needs an entry in `approvals/approved.json`. Outputs go to `restricted/`.
- **C (personal registration).** AMASS, SMPL and similar. Never fetched by tools. Needs a request in
  `approvals/pending/`, then the user places the files.
- **D (third-party video).** Refused.
- **E (team recordings).** Needs a consent id. Only derived joint data is kept.

Of the CML files, only the in-lab samples are converted. CML samples that come from CMU duplicate
trials we read from the CMU originals. HDM05-derived samples would be class B and are not used.

## Annotations (S3)

`tools/annotate.py`. All thresholds are [design] values unless the code names a source.

- **Floor.** A plane fitted to the lowest still foot point of each frame. Ladder clips get a level floor
  under the rungs, and so does any clip whose feet stay on one spot. Some walking trials have a floor that
  rises 0.3-1.2 deg in the data (`s3.floor_slope_deg`).
- **Flat-foot heights.** Skeletons put the heel and toe joints at different heights: a CMU toe joint sits
  2-4 cm up, a 100STYLE heel 7 cm up. Each point is measured from its own height while the foot stands
  flat (`s3.flat_offsets_m`).
- **Contacts.** A heel or toe is in contact when it is less than 3 cm above its flat height and moving
  under 0.25 m/s for at least 3 frames. It stays in contact up to 4.5 cm (hysteresis), and gaps of up to
  2 frames are filled. On ladders, a foot standing still more than 8 cm up is on a rung. A knee is down
  when its centre is within 0.07 H of the floor.
- **Heel strikes** start a heel contact after the whole foot was off the floor for 0.1 s. The cycle phase
  runs 0..1 between two strikes of one foot (0.4-2 s apart).
- **Task events.** Wrist deceleration peaks: at least 4 m/s^2, after the hand moved at least 0.4 m/s,
  and the hand slows to half that speed within 0.1 s. The CMU subject-62 hammer taps are slow
  (0.5-0.7 m/s). Rung contacts are listed with their height.
- **Activity tags** use the Builder primitive names: walk, run, stand, hold, carry, lift, put_down, stoop,
  squat, kneel, kneel_work, crawl, lie, get_up, ladder, hammer. A bend that reaches the hands below
  0.25 H is classed by knee flexion at its deepest frame [src: Straker 2003 via Washmuth 2022]: under
  45 deg is `stoop`, 45 to 112.5 deg is `semi_squat` and above that is `squat`. These are the simulator's
  `motions.LIFT_TECHNIQUES`. Lifts and put-downs carry the technique as `lift:<technique>` and
  `put_down:<technique>`. Lift or put-down comes from the trial name (alternating when a trial does
  both). Bends where the knees differ by more than 30 deg are flagged `asymmetric` (split or one-legged
  bends, which the simulator does not model). Source tags that S3 does not find in the motion are listed
  in `s3.notes`.

## Limb targets (S4)

`tools/features.py` writes `clips/<clip>.npz` at 100 Hz. Its docstring lists every field.

- **Legs.** Ankle relative to the hip in the smoothed heading frame, divided by that leg's length. Also
  the knee pole, the ankle lift above its flat height, the sole direction and normal (rig `_foot_axes`),
  contacts and phase.
- **Arms.** Hand relative to the shoulder in the upper-thoracic segment frame (the rig's `M[:, 3]`),
  divided by arm length, and the elbow pole.
- **Task frame.** The hands are also given in a task frame: the hammer post (median striking-hand
  position), the ladder (mean rung contact on the floor), or the floor under the pelvis.
- **Poles.** Where the knee or elbow is nearly straight (under 15 deg), the clip's bend direction is noise.
  There the pole blends to a default: knees over the toes, elbows back and slightly out.
- **No trunk targets.** The simulated spine is never replaced.
- **Clips without a floor** (NIOSH, pelvis-relative) have arm targets only (`has_legs = 0`).

## Gates (S5)

`tools/gates.py`. A preview, written before stage S6 existed: each clip is drawn the way the rig would draw
it, at 50 Hz. S6 measures the clips used by the viewer again on the real payload (see Integration):

- the clip's own pelvis path, scaled to the default 1.75 m worker (`rig.Dims`);
- ankle targets lifted to keep the heel, ball and toe above the floor (`rig._sole_lift`);
- the foot speed limited (`rig.speed_limited`), then `rig.two_bone` IK with the rig's bone lengths;
- the rig's foot geometry with toe bend.

Limits are from `.claude/agents/mocap-curator.md`. A clamp counts only when `two_bone` leaves the end
joint more than 5 mm from its target. Skate is measured on the clip's own heel and toe joints during
their labelled contacts. The sync gates read `PENDING`: they need the S6 time warp to the simulator's
anchors.

Any FAIL sets `status: rejected`. Status `gated` means every measured gate passed. It does not switch a
clip on: a person must set `reviewed_by`. `gates.fails_only_skate` marks clips that fail nothing but skate;
for the clips the viewer uses, S6's payload measurement replaces that skate result.

### Results (S5)

Of 84 clips, 27 are gated and 57 rejected.

**Failures by gate:** skate 57, reach 13, penetration 8, hip 7, pops 7.

- **Skate.** The source heel and toe joints drift 1.4-6 cm in the worst stance. The median stance drifts
  about 1 cm, because the joint centres roll with the foot. No walking or lifting clip meets the 1 cm
  limit as recorded. 33 clips fail only skate. They become usable once S6 locks the labelled contacts,
  as the rig already plants its feet.
- **Reach and penetration.** The crouch, deep-squat and kneel-to-lie clips flex the knee to 147-165 deg.
  The rig limits knees and elbows to 145 deg (`rig.MIN_JOINT_ANGLE` = 35 deg inner angle), so `two_bone`
  pushes the ankle 0.8-13 cm below the floor. Deep squats need that limit relaxed to 160 deg in S6.
- **Hip.** Walking and running clips reach -30 to -36 deg hip extension against the -30 deg limit. The
  neutral of most of these clips is calibrated on upright walking frames, so their pelvis zero is
  uncertain by a few degrees.
- **Pops.** Some CMU walking and running clips move a foot point 0.12-0.16 m per frame against the pelvis
  at toe-off. In `cmu_35_01` the toe joint turns 38 deg in one frame. In `cmu_16_15` the hand jumps 0.9 m
  (a marker swap).
- **Gated clips.** Six stand or lie clips (ACCAD stand, sway, look-around, waist-height box lift, lie down)
  and the 21 NIOSH kneeling trials, which have arm targets only.

## Integration (S6)

`thundersim/viz/mocap.py` (numpy/scipy only) reads `manifest.json` and the clips it names, checks each
clip's sha256, and drives the limbs where a listed clip applies. **Mocap is off by default**: the viewer's
Mocap limbs button or M, `?mocap=1`, `THUNDERSIM_MOCAP=1`, or `run_payload(mocap=True)`. The simulated spine,
the signals, segment and chip orientations, phases and the worn flag are byte-identical with mocap on and
off (`tools/check_invariants.py`, regression set, seeds 0-2). With mocap off the payload is byte-identical to
the viewer before S6.

**Time warp.** Every clip is warped piecewise-linearly between paired anchors: simulator events and the
same events in the clip. Where the clip loops back, each sampled signal carries the jump as an offset that
fades out over 0.15 s.

| Role | Limbs | Simulator anchors | Clip anchors |
|------|-------|-------------------|--------------|
| `walk` | legs and arms | heel strikes: the extremes of `motions.walk`'s pelvis yaw, which are its bob minima (`rig._stride_strikes`) | heel strikes [src: Zeni 2008]: the heel's most forward point relative to the pelvis |
| `carry_walk` | legs (the arms hold the load) | as walk | as walk, from carry clips |
| `hammer` | arms, waist-height hammering only | `Builder.impulses` times | right-wrist deceleration peaks in steady runs |
| `kneel_work` | arms | `kneel_work` impulses, at 0.85-1.15x real speed | wrist deceleration peaks |

- **Walking.** The rig still plans the footsteps and plants the feet (`rig.stand_feet`). A clip-shaped step
  takes from the clip:
  - the toe-off time (Zeni 2008: the toe's most backward point relative to the pelvis);
  - the swing path relative to the hip, bent so that it leaves from the rig's lift-off pose and lands on its
    planted pose;
  - the foot roll in swing and stance: a toes-up heel strike, then a heel rise;
  - the knee direction and the arm swing, in the simulated upper-thoracic frame and scaled with speed.

  Some rules keep the result physical. A swinging foot clears the floor by at least 1.5 cm in mid-swing
  [src: Winter 1992, minimum toe clearance]. A step lifts early when the simulated hip height would
  overstretch the leg, or when the thigh would extend past -26 deg [design: 4 deg inside the -30 deg gate].
  The walk clip is the one whose speed is closest to the simulated walk.
- **Hammering.** The hand reaches the rig's impact point exactly at each impulse. Between impulses it follows
  the clip's stroke in the clip's task frame. Overhead and on-ladder hammering stay procedural.
- **Kneeling work.** The clip's hands relative to the shoulders in the heading frame, so the hands point down
  to the work whatever the trunk angle. The hand height comes from the clip's height above the kneeling
  surface (S4 `hand_deck_h`: knees' 5th percentile minus 5 cm [design], roof slope unknown). The height is
  raised where a 0.9-arm reach cannot get that low.
- **Stays procedural:** lifts and put-downs (hands on the simulated grip point, feet planted), ladders,
  running, standing, falls, and anything without a listed clip (`coverage.json` `procedural_in_viewer`).

**Gates on the payload** (`tools/payload_gates.py`). Each listed clip is forced into its role on scenarios
of the regression set, seeds 0-2: walk on A-HAMMER, A-CARRY, A-KNEEL, A-LADDER, A-RUN and L-TRIP; carry walk
on A-CARRY; hammer on A-HAMMER; kneeling work on A-KNEEL. The gates are measured where the clip drives the
limbs, outside fall phases. The same windows of the procedural payload are measured as a baseline.

- **Skate** uses the clip-shaped stances: a frame counts while the rig foot is planted and the heel or toe
  tip is within 1 cm of the support.
- **Sync** compares the rig touchdown (heel within 3 mm of the support) with the bob minimum of
  `track.root_pos[:,1]`. For hammering it compares the striking hand's closest approach to the impact point
  with the impulse time.
- **Status.** An S6 PASS replaces the S5 skate result, which measured the clip's own heel and toe joints:
  the card becomes `gated` if S5 failed nothing else. An S6 FAIL makes it `rejected`.

**S6 results (seeds 0-2).**
- **Walk:** 4 of 7 clips pass (`accad_male1_b3_walk`, `accad_male1_b19_walktopickupbox_working`,
  `accad_male1_b21_putdownboxtowalk`, `cmu_86_13`).
  - Skate is at most 2.1 mm per stance, against 0.25 m procedurally.
  - Heel-strike sync is 10 ms median and 25 ms p90.
  - Hip extension is at least -28.2 deg, against -35.1 procedurally.
  - The three 100STYLE clips fail pops: the feet move 0.12-0.13 m per frame. These are slow walks (about
    0.8 m/s at the rig's size), so on the faster simulated walks the swing must cover a longer step in their
    shorter swing time.
- **Carry walk:** `accad_male1_b20_walkwithbox` passes.
- **Hammer:** all 3 CMU subject-62 clips pass, with strike sync of 7-9 ms median and 13-17 ms p90.
  - They are light taps of about 5 cm, far smaller than the procedural swing (whose hand pops fail at 0.25 m
    per frame).
- **Kneeling work:** all 21 NIOSH trials pass, where the procedural arms clamp on 99 % of frames.
  - The roofers' hand height is out of reach on 64-94 % of frames, because the simulated kneeling shoulder
    is about 0.95 m above the floor, about 0.2 m above the roofers'.
  - The hands therefore stay 0.35-0.65 m above the floor.
- The procedural baseline fails skate and hip extension while walking, pops while hammering and reach in
  kneeling work. Those failures are the rig's own, not a clip's.

## Status

- S0 to S6 are done. Mocap is off by default and no clip is on by default: `reviewed_by` is empty on every
  card (stage S7).
- Gaps with no open data (the team would need to record them): rebar tying, overhead drilling,
  jackhammer, work while lying, cab work.
- Open gaps not yet acquired: sitting, sitting on the ground, jumping down, on hands and knees (the ACCAD
  crawl is a low belly crawl). See `coverage.json`.
