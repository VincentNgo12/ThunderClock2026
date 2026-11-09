# ThunderSim realism scorecard

How close the simulated worker is to **real recordings**, measured like for like, which gaps were closed,
and which remain. Run date 2026-11-08. Everything is reproducible with the commands in section 3.

> **Simulator changes in this work make earlier results stale.** See section 10: falldet suite numbers,
> `shifts` false-alarm rates, the frozen `collapse_v1` held-out result in `detection/hub`,
> `calibration.json`, `tuned_baseline.json`. Viewer caches refresh automatically.
>
> **Round 2 (section 13):**
> - re-baselined on the merged Anny/motionval code;
> - UNIVRFall added (CC BY 4.0, lower-back sensor, video-labelled falls, worksite data);
> - fall descent fixed (iteration 4);
> - syncope re-fit (no change); `calibrate` re-run;
> - second seed set run.
> - Section 13.8 holds the current numbers.

## 1. Summary

- **Baseline:** in all 20 categories, a classifier separates real from simulated signals almost
  perfectly (subject/seed-grouped AUC 0.98-1.00). The largest downstream-relevant gaps were all in the
  **fall impact sequence**:
  - waist impact peaks about 2x real (on mats), jerk 2.5-4x;
  - the pelvis bounced off the floor after impact (waist reads 0.15-0.3 g; real about 0.6 g);
  - a second trunk impact and trunk "ringing" (chest 3-10 rad/s, 0.2-0.8 s after impact; real about 2);
  - the upper trunk lay about 20 deg past horizontal after every fall;
  - **syncope** landed at about 1.4 g, so falldet's Stage 1 never fired (real acted syncope: 3.5 g,
    Stage 1 in 95 %).
- **Kept changes** (three iterations, each fitted to real data on separate fit seeds):
  1. the upper-trunk curl relaxes onto the floor after landing, to a residual drawn from the real spread;
  2. body-first contact: a restitution-based hysteretic contact (no bounce, e = 0.25), a damped trunk
     landing, a slower trunk topple about the landed pelvis, and a softer crash mat (mat falls only);
  3. syncope: a short knee buckle, then a partially supported drop that lands through the ordinary contact.
- **Result** (40 runs per event category, dev seeds):
  - summed event-level gap 770.9 -> 635.7 nW (-18 %);
  - 48 event features shrank beyond their CI and 14 grew;
  - features inside real-data noise rose from 51 to 64 of 176;
  - activity-level features were not changed (420 features, 88 inside noise).
- **Downstream:**
  - falldet's `impact_peak_g` input gap fell 2-3x in every fall category (side 10.3 -> 5.1 nW,
    backward 6.2 -> 2.6, forward 3.4 -> 1.4);
  - simulated syncope now fires Stage 1 in 100 % of runs (was 0 %; real 95 %).
- **Hazard-relevant side effect** (simulated only; there is no real hard-floor data):
  - site-fall pelvis peaks on their own surfaces, at target-firmware bandwidth, rose about 10 %
    (L-TRIP median 14.4 -> 16.5 g);
  - SYNC on site floors rose from 1.6 g to 8.3 g.
  - A first contact model (Hunt-Crossley) fitted the mats about as well, but raised hard-surface peaks
    20-50 % by extrapolation and was rejected for that reason.
- **Still not realistic:**
  - fall descent is about twice as fast as real (rotation 10-90 % in 0.44 s vs 0.92 s);
  - backward and side impact peaks on mats are still 1.5-1.7x real;
  - voluntary activities (walking, bending, squatting, jogging, lying down, construction tasks) were not
    tuned here and are still easy to tell apart from real data;
  - post-fall stillness and sitting differ.
  - The classifier AUC is still about 1.0 everywhere: the simulator is better, not indistinguishable.

## 2. Data

| Dataset | Use | Licence | Access |
|---|---|---|---|
| Erciyes "Simulated Falls and Daily Living Activities" (UCI 455): 17 volunteers, Xsens MTw at 25 Hz, waist + chest (sensor 340539, see the ID-swap note in `detection/models/scripts/load_raw.py`) | 20 fall and 16 ADL types; falls onto mats | CC BY 4.0 | already on disk (`research/fall_model/data/raw/erciyes_uci455`) |
| VTT-ConIot (Zenodo 4683703): 13 participants, 16 construction activities, hip ("trousers") + back sensors, 100 Hz | normal construction work | CC BY 4.0 | already on disk (`research/fall_model/data/raw/vtt_coniot`) |

- Both datasets are read from research-95's `data/interim/raw_trials.pkl` (sensor frame, SI units) and
  rotated into the simulator's segment frame with research-95's own data-driven calibration
  (`harmonize.calibrate`, imported read-only). Nothing is resampled or device-emulated on the real side.
- **No new datasets were downloaded.** The machine was heavily loaded (load average 30-60 on 10 cores),
  and the two datasets on disk already exposed more gaps than three iterations could close.
  - Next open candidates (verify the licence page first): UMAFall (figshare), UNIVRFall (Zenodo; falls
    from height and worksite data), the 2026 ladder-fall dataset on GitHub.
  - KFall, FARSEEING, MobiAct and AMASS are gated and were not used.
- Measured in the VTT files: every accelerometer axis clips at exactly +/-2 g and the gyro at about
  4.4 rad/s (250 deg/s). The VTT sensors therefore ran at +/-2 g / +/-250 deg/s, and the emulation clips
  the simulated signals the same way.

## 3. Method (`detection/simulation/realism/`)

```
cd detection/simulation
.venv/bin/python -m realism score --out <scratch>/run --events 40 --act 20 --jobs 4
.venv/bin/python -m realism compare <scratch>/before <scratch>/after
.venv/bin/python -m realism rescore <scratch>/run           # recompute from a saved sim_side.pkl
.venv/bin/python -m realism.fit <grid> --out <scratch>/fit  # parameter grids on fit seeds
.venv/bin/python -m realism.site_impacts --n 10             # site-fall peaks, original vs current constants
```

**Runtime:** 6.6 min end to end with the real-side cache built (simulation 216 s, scoring 180 s, at load
average 16-28). The first build of the real-side cache takes about 20 min. Timings are saved in
`scorecard.json` -> `meta`.

**Scratch output:** `research/artifacts/realism_runs/`
- scorecard runs `baseline/`, `iter1/`, `iter2/`, `iter3_hc/`, `final/`, `final2/` (current state); each
  has `scorecard.json`, `scorecard.md`, `event_traces.png` and `top_event_gaps.png`;
- parameter fits `fit_contact/`, `fit_sync/`, `fit_anny/`;
- `site_impacts_final.md`, and `cache/` (real side).

**Package modules:** `paths`, `real`, `emulate`, `mapping`, `features`, `stats`, `score`, `fit`,
`site_impacts`, `__main__`.

**Like-for-like emulation** (`emulate.py`). The real data is never altered; the simulated body is made to
look like each real sensor:
- frame: the segment frame on both sides (mount error removed on both);
- placement:
  - Erciyes waist = the pelvis anchor;
  - Erciyes chest = a sternum point 0.20 m in front of and 0.08 m below the upper-thoracic anchor (the
    inverse of `harmonize.CHEST_TO_UT_M`), so it sees the same lever-arm terms as the real chest sensor;
  - VTT hip = the pelvis anchor; VTT back = the upper-thoracic anchor;
- signal chain: the simulator's own soft-tissue filter, a 4th-order low-pass (Erciyes 11 Hz, VTT 40 Hz),
  decimation to 25 / 100 Hz, small white noise, per-axis clipping (VTT +/-2 g, +/-250 deg/s);
- Erciyes events only: falls onto `crash_mat`, dry, light clothing, and an Erciyes-like population
  (1.57-1.84 m, 47-92 kg, no tool belt [verify against the dataset description]). The scenario scripts
  run unchanged; only the site and worker draws are patched, in the same random-number order;
- seeds: scorecard 500000 + 1000*block + i; fits on blocks 50 and up (disjoint); tests on block 90;
  nothing at or above 1000000.

**Features** (`features.py`).
- **Event level:** events are aligned on the waist |a| peak, found the same way on both sides (the
  simulator's ground-truth impact time is not used). 22 features:
  - impact peak (waist, chest); free-fall depth and duration;
  - waist and chest angular-rate peaks and their timing; descent time (10-90 % of chest rotation above the
    pre-event baseline); total rotation;
  - inclination change and final inclination (waist, chest);
  - number of impact peaks; post-impact minimum (rebound flight) and |a| spread; trunk rate 0.2-0.8 s after
    impact; settling time;
  - post-fall stillness (acc, gyro); pre-fall activity; jerk.
- **Activity level:** 5 s windows, hop 2.5 s:
  - per-axis acc and gyro spread; |a| spread and 95th percentile;
  - inclination mean and spread; dominant frequency; band RMS at 0.3-3 / 3-8 / 8-12 Hz (VTT 8-40 Hz);
  - jerk; gait periodicity; chest-minus-waist inclination.

**Gap statistics** (`stats.py`).
- Per feature:
  - nW = Wasserstein-1 distance / robust real scale;
  - 95 % CI by bootstrapping whole subjects and seeds;
  - **null95** = 95th percentile of nW between two random halves of the real subjects (how much the real
    data disagrees with itself);
  - "inside real-data noise" = CI low <= null95; excess = nW / null95.
- Per category: a classifier two-sample test (gradient-boosted trees, subject/seed GroupKFold,
  out-of-fold AUC with a group-bootstrap CI). Univariate AUCs name the features that separate real from
  sim.

**Ranking.**
- Size = excess.
- Impact = excess x downstream relevance x mapping confidence:
  - relevance: 1 for falldet Stage-1/Stage-2 inputs and impact severity, 0.5-0.7 otherwise, and 0 for
    features fixed by the real protocol (Erciyes falls start from quiet standing; A-SIT is a deliberately
    hard sit);
  - confidence: 1 / 0.7 / 0.4 for high / medium / low mapping confidence;
  - activity rows are additionally weighted 0.6.

**Downstream** (`score.py`):
- falldet's target model (`detect(force_stage2=True)`, read-only) runs on the real
  `canonical_target.npz` windows and on simulated 4-anchor streams band-limited like Erciyes;
- compared: Stage-1 rate, P(fall_confirmed), and gaps on falldet's own Stage-2 input features.

**Plug-in feature providers** (for `realism/kinematics.py`, the motionval work):
- `score.register_provider(fn)`; modules listed in `PROVIDER_MODULES` (default `realism.kinematics`) are
  imported if present.
- `fn(context) -> rows`; each row is `{level, category, feature, real, real_groups, sim, sim_groups,
  relevance, confidence, description}`.
- Provider rows get the same nW, CI, null95 and ranking; `rescore` keeps them (`provider_rows.pkl`).

## 4. Scenario <-> real mapping

| Category | Real (codes) | Simulator | Confidence | Main uncertainty |
|---|---|---|---|---|
| fall_forward | Erciyes 901, 902, 904 | L-TRIP | medium | Real falls start from standing, on purpose, onto mats; the sim trips while walking |
| fall_backward | Erciyes 909-912 | L-SLIP | medium | Real falls are deliberate, not slips; same direction and landing |
| fall_side | Erciyes 913, 915 | STRUCK-KD (left/right blows only) | low | Erciyes has no push; the knock-down runs the ordinary side fall |
| syncope | Erciyes 919 | SYNC | medium | Acted faint onto a mat; acted faints keep some muscle control |
| fall_recover | Erciyes 905, 906, 914, 916 | FALL-GETUP | medium | Real volunteers are up in about 2 s; the sim lies 3-15 s by design (post-fall features are protocol) |
| stumble | Erciyes 808 | NF-STUMBLE | low | Stumble with recovery vs stumble-and-catch |
| sit_hard | Erciyes 811, 812 | A-SIT (first sit) | medium | A-SIT is always hard by design; Erciyes sits are normal (impact features are protocol) |
| lie_down | Erciyes 815 | A-LIE lie-down | medium | Lying onto a bed vs a controlled lie-down onto the ground |
| walk | Erciyes 801 | walk | high | |
| jog | Erciyes 803 | run (2.5-3.5 m/s) | medium | Erciyes jogging pace unknown |
| bend | Erciyes 805, 806 | stoop 60-90 deg | high | |
| squat | Erciyes 804 | squat | high | |
| walk_site | VTT 1013, 1014 | walk | high | VTT sensors on clothing (see section 7) |
| cart | VTT 1015 | walk with a 20 kg load | medium | Pushing vs carrying |
| pick | VTT 1005 | stoop-carry-stoop | medium | |
| kneel | VTT 1011, 1012 | kneel_work | medium | |
| overhead | VTT 1009 | drill_overhead | low | Real has no tool; the sim adds 20-50 Hz vibration |
| lie_work | VTT 1008 | lie supine + lying_activity | medium | |
| jump_down | VTT 1007 | jump_down 0.3-0.6 m | high | |
| stairs | VTT 1006, 1016 | ladder | low | ThunderSim has no stair walking; the ladder is the nearest primitive |

Unmapped real codes:
- Erciyes 903 (ends kneeling), 907/908 (oblique), 917 (rolling out of bed), 918 (podium), 920 (syncope
  against a wall), 802, 807, 809, 810, 813, 814, 816;
- VTT 1001-1004 and 1010.

Unmapped simulator scenarios (no open real recording on disk covers them):
- falls from height: H-LAD, H-SCAF, H-ROOF, H-HOLE, H-EXC;
- ARR, NET, STAIR, SYNC-KNEEL, NF-RAIL, STRUCK-NF, SLIP-FRICTION;
- the tool and vehicle hard negatives: A-HAMMER, A-JACK, A-EQUIP, A-REBAR, HAT-OFF, STRAP-SLIP,
  DEVICE-OFF, A-NAP.

## 5. Baseline scorecard (simulator as found)

40 simulated events per event category and 20 clips per activity category (real side: all Erciyes and
VTT trials). The table includes the two rebound features (`postmin_w`, `gyro_post_c`) that were added
after the first run and rescored.

### Classifier two-sample tests (AUC; 0.5 = indistinguishable)

| Category | AUC | Category | AUC |
|---|---|---|---|
| fall_forward | 1.00 | walk | 1.00 |
| fall_backward | 1.00 | jog | 0.99 |
| fall_side | 1.00 | bend | 1.00 |
| syncope | 1.00 | squat | 0.99 |
| fall_recover | 0.99 | walk_site | 1.00 |
| stumble | 1.00 | cart | 1.00 |
| sit_hard | 1.00 | pick | 1.00 |
| lie_down | 1.00 | kneel | 1.00 |
| | | overhead | 1.00 |
| | | lie_work | 1.00 |
| | | jump_down | 0.98 |
| | | stairs | 1.00 |

95 % CIs are all within 0.96-1.00.

### Ranked gaps (top 20 by impact)

| # | Category (AUC) | Feature | Real median | Sim median | nW (95 % CI) | null95 | Downstream effect |
|---|---|---|---|---|---|---|---|
| 1 | syncope (1.00) | waist |a| spread before impact | 0.00 g | 0.05 g | 49.0 (42.9-54.6) | 0.79 | Symptom of the 2.4-3.4 s guided collapse; falldet Stage 1 fired on 0 % of sim vs 95 % of real syncope |
| 2 | fall_side (1.00) | waist jerk | 43 g/s | 175 g/s | 11.2 (10.4-12.0) | 0.55 | falldet `jerk_peak_g_s` gap 12.2 nW |
| 3 | fall_backward (1.00) | chest rate 0.2-0.8 s after impact | 2.19 rad/s | 10.26 rad/s | 5.0 (4.2-5.7) | 0.46 | Trunk slams down a second time; falldet `post_gyro_mean` |
| 4 | fall_side (1.00) | waist impact peak | 3.59 g | 8.73 g | 11.9 (11.1-12.8) | 0.72 | falldet `impact_peak_g` gap 10.3 nW; hazard severity |
| 5 | fall_backward (1.00) | waist impact peak | 4.24 g | 8.72 g | 5.4 (4.8-6.0) | 0.59 | falldet `impact_peak_g` gap 6.2 nW |
| 6 | fall_backward (1.00) | total chest rotation | 2.00 rad | 4.82 rad | 4.4 (3.7-5.1) | 0.42 | falldet `rot_peak_rad_s` |
| 7 | lie_down (1.00) | chest rate peak | 2.72 rad/s | 0.16 rad/s | 6.1 (5.8-6.4) | 0.58 | Voluntary motion (finding for the motion owners) |
| 8 | squat (0.99) | chest |a| 95th percentile | 1.03 g | 1.33 g | 2.9 (2.7-3.1) | 0.35 | Voluntary motion |
| 9 | fall_backward (1.00) | waist jerk | 72 g/s | 178 g/s | 4.5 (4.1-5.0) | 0.64 | falldet `jerk_peak_g_s` gap 5.3 nW |
| 10 | fall_backward (1.00) | chest inclination after the fall | 80.7 deg | 102.6 deg | 1.19 (1.04-1.35) | 0.18 | falldet `incl_post_upper_thoracic`; the same gap in all 5 lying categories |
| 11 | walk_site (1.00) | hip 3-8 Hz band RMS (log10 g) | -0.90 | -2.06 | 9.1 (8.1-10.1) | 0.65 | Mostly a VTT clothing-mount effect (section 7) |
| 12 | fall_backward (1.00) | chest inclination change | 75.6 deg | 97.3 deg | 0.93 (0.80-1.06) | 0.17 | as #10 |
| 13 | jump_down (0.98) | waist rate mean | 1.08 rad/s | 0.16 rad/s | 4.7 (4.4-5.1) | 0.72 | Voluntary motion |
| 14 | fall_backward (1.00) | waist minimum after impact (bounce) | 0.67 g | 0.25 g | 2.9 (2.6-3.2) | 0.53 | The pelvis leaves the floor: reads near free fall after impact |
| 15 | jog (0.99) | waist |a| spread | 0.27 g | 0.95 g | 1.7 (1.6-1.8) | 0.19 | Voluntary motion |
| 16 | lie_down (1.00) | chest rate when lying (log10) | -1.80 | -0.59 | 4.9 (4.2-5.4) | 0.97 | falldet stillness features |
| 17 | squat (0.99) | chest |a| spread | 0.04 g | 0.15 g | 1.9 (1.8-2.1) | 0.34 | Voluntary motion |
| 18 | bend (1.00) | chest rate mean | 0.13 rad/s | 0.49 rad/s | 1.0 (1.0-1.1) | 0.18 | Voluntary motion |
| 19 | jump_down (0.98) | chest rate mean | 0.93 rad/s | 0.21 rad/s | 5.1 (4.7-5.5) | 0.92 | Voluntary motion |
| 20 | fall_backward (1.00) | chest impact peak | 4.92 g | 9.43 g | 3.5 (3.0-3.9) | 0.52 | Second trunk impact |

Also outside noise at baseline, ranked lower:
- forward and side chest inclination after the fall (nW 1.61 and 1.76; sim 109 and 101 deg vs real 88
  and 84 deg);
- double impacts: backward 2 peaks vs 1 (nW 8.5), side 2 vs 1 (nW 16.7);
- waist minimum after impact: forward 0.31 vs 0.59 g, side 0.15 vs 0.57 g.

### Downstream at baseline (falldet target model)

| Category | Stage-1 rate real (CI) | Stage-1 rate sim (CI) | P(fall) real | P(fall) sim | Largest falldet input gaps (nW) |
|---|---|---|---|---|---|
| fall_forward | 1.00 (1.00-1.00) | 1.00 (1.00-1.00) | 1.00 | 1.00 | pre_std_g 12.3 (protocol), post_gyro_mean 7.3, jerk_peak_g_s 4.2 |
| fall_backward | 1.00 (0.99-1.00) | 1.00 | 1.00 | 1.00 | pre_std_g 18.2 (protocol), impact_peak_g 6.2, jerk_peak_g_s 5.3 |
| fall_side | 1.00 | 1.00 | 1.00 | 1.00 | jerk_peak_g_s 12.2, impact_peak_g 10.3 |
| syncope | 0.95 (0.88-1.00) | **0.00** | 1.00 | 0.68 | post_std_g 5.6, impact_peak_g 2.1 |
| fall_recover | 1.00 | 1.00 | 0.00 | 1.00 | n_anchors_lying 83 (protocol: the sim lies 3-15 s) |
| stumble | 0.53 (0.35-0.70) | 0.28 (0.15-0.40) | 0.00 | 0.00 | still_frac 6.4, longest_still_s 4.3 |
| sit_hard | 0.02 (0.00-0.04) | 0.70 (0.55-0.80) | 0.00 | 0.72 | Hard sit by design vs normal real sits |
| lie_down | 0.05 (0.00-0.21) | 0.00 | 0.00 | 0.00 | post_gyro_mean 20.9 |

## 6. Changes, one at a time (changelog)

**Rule:** keep a change if its target gaps shrink and nothing else grows beyond its CI. Judgement calls,
where a few smaller gaps grew, are listed explicitly. Every change is controlled by module constants whose
comments give the original values, so each can be reverted on its own (section 10). None of these
changes touch `thundersim/motions.py`, `tests/test_physics.py` or `thundersim/viz/`.

### Iteration 1: the trunk lies flat after a fall (KEPT)

- **Gap.** Chest inclination 2-5 s after a lying fall was 101-111 deg in the sim vs 80-88 deg real
  (nW 1.2-1.8, outside noise in all 5 lying categories). Real chest-minus-waist has a median of about
  -1 deg and a robust SD of about 10 deg. The sim was always +14 to +26 deg, because the descent "curl"
  (0-40 deg of upper-trunk flexion) was kept while lying.
- **Change** (`thundersim/falls.py`). From touchdown, the curl relaxes over 0.4 s to a residual drawn from
  N(-1, 10) deg. The draw uses a side generator that does not advance the run's random stream.
  - New constants `CURL_REST_DEG = (-1.0, 10.0)` and `FLATTEN_S = 0.4`.
  - New function `side_rng()`.
  - `Faller.__init__` (`self.flatten`); `Faller.record` (curl blend); new `Faller.start_flatten`.
  - `Faller.body_contact` calls `start_flatten` at pelvis touchdown and at trunk landing.
- **Before -> after (nW; null95):**
  - chest inclination after the fall: forward 1.61 -> 0.74 (0.78, inside noise); backward 1.19 -> 0.56;
    side 1.76 -> 0.49 (inside); syncope 1.28 -> 0.25 (inside); recover 2.73 -> 2.14;
  - falldet `incl_post_upper_thoracic` gap: side 1.74 -> 0.42, syncope 1.31 -> 0.25;
  - nothing grew beyond its CI.
- **Test change.** `tests/test_regression.py`: L-SLIP seed 3 -> 4.
  - Seed 3 ends curled (pelvis about 60 deg). It passed only because the upper trunk kept +20-40 deg of
    curl. With the trunk flat, the baseline model's ">= 3 anchors past 60 deg" rule now calls it a near
    fall.
  - This is a real finding about the baseline model (it misses curled lying), not a regression of the
    gate: seeds 3 and 7 of 3-8 now end this way.

### Iteration 2: no bounce after body-first impacts (KEPT in form 2b; 2a REJECTED)

- **Gap** (the largest group by impact):
  - after impact the sim pelvis left the floor (waist minimum 0.15-0.31 g vs real 0.57-0.67 g);
  - the trunk landed a second time and rang (chest rate 0.2-0.8 s after impact 3.2-10.3 rad/s vs real
    1.4-2.2);
  - 2 impact peaks instead of 1; total rotation 1.5-2.4x real; peaks 1.8-2.4x real.
  - The impulse of the waist pulse (sim about 5.2 m/s vs real about 3.0) showed the excess was elastic
    rebound (impulse = v(1+e)), not only contact speed. This is the README's known limit "the rebound after
    hard body-first impacts".
- **Fits** (`realism/fit.py`, 15 runs per category per grid point, fit seeds, forward/backward/side falls):
  1. `contact`: Hunt-Crossley pelvis damping x trunk-landing damping x topple gain; best at the grid edges.
  2. `contact2`: adds a crash-mat stiffness scale; the softest mat searched (x0.2) was best (grid edge).
  3. `contact3`: Hunt-Crossley alpha = 0.6 s/m is an interior optimum (with topple gain 0.5).
     - **REVERTED: topple gain 0.25.** It fitted the impact features better, but the first full
       scorecard showed backward falls no longer lying within 2-5 s (`post_incl_w` grew; chest
       inclination 84 -> 112 deg), so gain 0.5 is used.
  4. **REJECTED: 2a, Hunt-Crossley contact.** On the mats it scored well: e.g. backward peak nW
     5.37 -> 3.23, side 11.94 -> 8.56; trunk rebound backward 4.98 -> 2.71; 41 event features shrank,
     6 grew. But the hard-surface check (`realism/site_impacts.py`) rejected it: its velocity-proportional
     force, fitted on about 3 m/s mat impacts, raised site-fall pelvis peaks on concrete at 44 Hz by
     20-50 % (L-TRIP 14.4 -> 19.1 g, H-SCAF 34.6 -> 51.1 g), with no real hard-floor data to support that.
     `CONTACT_HC_ALPHA` stays in the code, set to None.
  5. `contact4`: **2b, KEPT.** A restitution-based hysteretic contact.
     - Loading is a pure spring, so the hard-surface loading peak stays as before. A stiff, damped
       (zeta 0.7) unloading line k/e^2 returns e^2 of the energy.
     - Objective 55 vs 61 for Hunt-Crossley on the same seeds; flat for e = 0.1-0.4, so e = 0.25 is
       weakly identified.
     - Its permanent set left lying bodies up to about 9 cm inside the support (the viewer's stairs test
       found it). The set now recovers like soft tissue with a 1.5 s time constant.
       **REVERTED: a 0.3 s recovery**, which lifted the body sharply enough to re-create the
       post-impact dip.
- **Kept constants and code** (`thundersim/falls.py` unless noted):
  - `CONTACT_RESTITUTION = 0.25` and `SET_RECOVERY_S = 1.5` (both new);
  - `ROT_CONTACT_ZETA = 1.2`, ramped in over `ROT_CONTACT_RAMP_S = 0.05` (was 0.4 with no ramp);
  - `TRUNK_TOPPLE_GAIN = 0.5` (was 1.0);
  - in `Faller.body_contact`: new contact-force branches (`max_comp`, `t_max`); a trunk-landing latch
    (`landed_at`: the trunk no longer returns to a free topple after an overshoot); ramped rotational
    damping; topple gain;
  - `thundersim/site.py`: `SURFACES["crash_mat"]` stiffness 3e3-6e3 N/m (was 1.5e4-3e4).
- **Before (iteration 1) -> after (final, which includes iteration 3):**
  - over 176 event features: 41 shrank beyond CI, 11 grew; inside noise 54 -> 64;
  - headline numbers are in section 8.
- **Hysteretic (2b) vs Hunt-Crossley (2a), both with iteration 3:**
  - summed event nW 631.9 (2a) vs 635.7 (2b); 2b shrank 11 features and grew 15;
  - inside noise 73 (2a) vs 64 (2b);
  - impact peaks are closer to real with 2b (forward 2.68 -> 1.94 nW, backward 3.23 -> 2.12,
    side 8.56 -> 5.97);
  - 2b post-impact |a| spread is too quiet (0.02-0.07 g vs real 0.09-0.10 g).
  - **2b was chosen for its safer extrapolation to hard surfaces, not for a better fit on mats.**
- **Judgement call.** The gaps that grew are smaller than those that closed. Mainly:
  - forward/recover descent time: the sim was already too fast (0.64 -> 0.44 s vs real 0.92-1.24 s).
    Damping stops the post-impact rotation sooner, which exposes the pre-impact speed;
  - post-impact |a| spread;
  - chest rate-peak timing.
  - A strict reading of the rule would revert iteration 2; the numbers are here so the team can decide.
- **Hazard-relevant side effect.** Site falls on their own surfaces, pelvis |a| peak at 44 Hz / 100 Hz,
  10 runs each, original vs current constants; median (IQR) in g:

| Scenario | Original | Current |
|---|---|---|
| L-TRIP | 14.4 (10.2-16.1) | 16.5 (11.8-18.2) |
| L-SLIP | 15.0 (12.1-15.5) | 16.4 (13.6-16.9) |
| STRUCK-KD | 14.4 (13.4-16.6) | 16.1 (14.9-18.2) |
| SYNC | 1.6 (1.6-1.6) | 8.3 (8.2-8.6) |
| H-LAD | 22.2 (19.9-27.4) | 22.2 (19.9-27.4) |
| H-SCAF | 34.6 (30.8-39.3) | 38.5 (34.5-41.1) |
| STAIR | 14.9 (11.9-19.6) | 16.7 (13.6-21.4) |

  - H-LAD is unchanged (feet-first landing). The crash-mat stiffness in `site.py` was not switched for
    this table.
  - 10 % of random sites are crash mats, and falls there are softer now.

### Iteration 3: syncope lands (KEPT)

- **Gap.**
  - Acted syncope (Erciyes 919): waist peak 3.46 g, a dip to 0.46 g, chest rotation peaking at impact,
    Stage 1 fires in 95 %.
  - Sim SYNC: 1.4-1.7 g; rotation peak 0.3-0.5 s before contact; Stage 1 in 0 %; P(fall) 0.66-0.88.
  - The guided sag was a [design] assumption, and the biomechanics note found no kinematic data on
    syncope, so Erciyes 919 is the only real evidence available.
- **Change** (`thundersim/falls.py`):
  - new constants `SYNC_SUPPORT = 0.7` and `SYNC_BUCKLE_S = (0.4, 0.8)` (originals: None, the guided
    sag, and (1.0, 2.0));
  - `Faller.body_contact` gains a `support` parameter: the fraction of gravity the legs still carry until
    touchdown;
  - `slow_collapse`: a knee buckle to a half-buckled stance, then a supported drop that lands through the
    ordinary contact. With `SYNC_SUPPORT = None`, the old path runs in the same RNG order.
- **Fit** (`sync`, `sync2`, 20 runs per point):
  - support 0.6 / 0.7 / 0.8 score 15.8 / 15.0 / 14.9, which is flat (weakly identified), so 0.7 was
    chosen;
  - a 0.4-0.8 s buckle beat 0.2-0.4 s and the original 1-2 s; the original guided sag scored 20.6.
- **Before -> after** (iteration-2a state -> iteration 3):
  - 11 syncope features shrank; none grew anywhere;
  - waist peak 1.66 -> 3.56 g (real 3.46); minimum before impact 0.09 -> 0.55 g (real 0.46); chest
    rate-peak timing -0.30 -> +0.32 s (real +0.08); settling 0.32 -> 0.74 s (real 0.92);
  - downstream: Stage 1 0.00 -> 1.00 (real 0.95); P(fall) 0.88 -> 1.00 (real 1.00); falldet
    `impact_peak_g` gap 2.14 -> 0.58; `ff_min_g` gap 2.05 -> 0.63.
- **Caveats.**
  - After the switch to the hysteretic contact (2b), syncope's post-impact minimum is too low
    (0.40 vs 0.71 g; nW 1.09 -> 1.85). The syncope fit was done with the Hunt-Crossley contact and was
    not repeated.
  - SYNC-KNEEL uses the same drop from kneeling; no real data covers it.

### Measured candidate, REJECTED: Anny-derived body geometry

- **Candidate:** Anny average-male geometry, scaled by height: pelvis height 0.575 x height, and centre of
  mass to pelvis anchor (0, 0.0042 x height, 0.0275 x height + 0.04). Tested with and without Anny's
  spine spacing (0.0498 / 0.0760 / 0.1340 x height). Grid `anny`, 20 runs per category, forward/backward/
  side falls.
- **Result:** objective 58.4 (current formulas) vs 61.8 (Anny) vs 61.7 (Anny + spacing).
  - It did not touch the descent gap (forward descent nW 2.04 -> 2.03, side 2.57 -> 2.55).
  - It worsened impact peaks (side 5.98 -> 7.00, backward 2.15 -> 2.52).
  - Anny's centre of mass sits about 1.0 m above the floor, almost the same as ours, so it does not change
    the topple dynamics. Not adopted.

## 7. Worker interface changes

`thundersim/spine.py`, backward-compatible. Existing fields and their order are unchanged; the
Anny-to-Worker adapter JSON still works.
- New optional field `pelvis_height_m: float | None = None`. When set, it replaces 0.53 x height in
  `Worker.pelvis_height`.
- New optional field `com_to_pelvis_m: tuple[float, float, float] | None = None`: whole-body centre of mass
  to the pelvis anchor, segment frame (+X right, +Y up, +Z posterior). When None, the old formula
  (0, -0.03 x height, 0.10) is used.
- New method `Worker.com_to_pelvis()`. `falls.Faller.__init__` now reads `r_p` from it (it was hard-coded).
- Defaults reproduce the previous output exactly: summed event gap 635.7 before and after the refactor.
- Still hard-coded elsewhere:
  - the impact effective mass, 0.35 x body mass (`site.effective_mass`);
  - the lying rest heights (`falls.lying_rest_height`);
  - the Erciyes chest lever arm in `realism/emulate.py` (a property of the real dataset, not of the worker).

## 8. Tests

- **90 tests, all pass:** 86 existing plus 4 new in `tests/test_realism.py`. The new tests:
  - the trunk lies flat after falls;
  - no rebound flight and no trunk ringing;
  - syncope lands with an impact (above 2.2 g at Erciyes bandwidth);
  - a scorecard smoke test that runs only when the real-data cache exists.
- **Changed expectations:**
  - `tests/test_regression.py`: L-SLIP seed 3 -> 4 (iteration 1, above).
  - `tests/test_viz_props.py`: STAIR seed 7 -> 9.
    - The contact change moved seed 7's fall so that its lead-in walk crosses the stairwell; seed 9 again
      fits one consistent scene.
    - Seeds 6, 13 and 19 revealed the buried-body issue, now fixed by the set recovery.
    - Only the seed in the test changed; `thundersim/viz/` was not touched.

## 9. Final state

### Headline fall features (`final/`, 40 runs per category)

Each sim cell is median (nW).

| Category | Feature | Real median | Baseline sim | Final sim | null95 | Final inside noise? |
|---|---|---|---|---|---|---|
| fall_forward | waist impact peak (g) | 3.77 | 6.85 (4.02) | 4.61 (1.94) | 2.34 | yes |
| fall_forward | waist jerk (g/s) | 51 | 126 (3.82) | 69 (1.40) | 2.19 | yes |
| fall_forward | waist minimum after impact (g) | 0.59 | 0.31 (1.02) | 0.60 (0.23) | 0.36 | yes |
| fall_forward | total chest rotation (rad) | 1.92 | 3.01 (3.48) | 2.00 (0.27) | 0.66 | yes |
| fall_forward | chest inclination after (deg) | 88.0 | 108.9 (1.61) | 86.2 (0.67) | 0.78 | yes |
| fall_forward | descent time (s) | 0.92 | 0.64 (1.33) | 0.44 (2.02) | 0.65 | no (grew) |
| fall_backward | waist impact peak (g) | 4.24 | 8.72 (5.37) | 6.17 (2.12) | 0.59 | no |
| fall_backward | waist jerk (g/s) | 72 | 178 (4.54) | 106 (1.31) | 0.64 | no |
| fall_backward | waist minimum after impact (g) | 0.67 | 0.25 (2.92) | 0.43 (1.50) | 0.53 | no |
| fall_backward | chest rate 0.2-0.8 s after (rad/s) | 2.19 | 10.26 (4.99) | 7.05 (2.91) | 0.46 | no |
| fall_backward | impact peaks (count) | 1 | 2 (8.54) | 1 (1.57) | 2.99 | yes |
| fall_backward | chest inclination after (deg) | 80.7 | 102.6 (1.19) | 87.2 (0.65) | 0.18 | no |
| fall_side | waist impact peak (g) | 3.59 | 8.73 (11.94) | 6.00 (5.97) | 0.72 | no |
| fall_side | waist jerk (g/s) | 43 | 175 (11.23) | 99 (4.72) | 0.55 | no |
| fall_side | waist minimum after impact (g) | 0.57 | 0.15 (2.62) | 0.45 (1.16) | 0.66 | no |
| fall_side | impact peaks (count) | 1 | 2 (16.74) | 1 (0.74) | 2.50 | yes |
| fall_side | chest inclination after (deg) | 83.9 | 101.0 (1.76) | 80.9 (0.47) | 0.84 | yes |
| syncope | waist impact peak (g) | 3.46 | 1.36 (2.76) | 3.59 (1.47) | 2.70 | yes |
| syncope | chest rate 0.2-0.8 s after (rad/s) | 2.94 | 0.50 (1.84) | 5.13 (0.98) | 0.91 | yes |
| syncope | chest inclination after (deg) | 84.9 | 110.7 (1.28) | 85.6 (0.18) | 1.05 | yes |
| syncope | waist minimum after impact (g) | 0.71 | 0.94 (1.09) | 0.40 (1.85) | 0.51 | no (grew) |

Event features that grew beyond their baseline CI:
- descent time (forward, recover);
- chest rate-peak timing (forward, side, recover);
- post-impact |a| spread (forward, backward, side);
- settling time (side, recover);
- chest rate after impact (side, recover; the sim is now too quiet, 0.8 vs 2.2 rad/s);
- impact-peak count (forward);
- syncope post-impact minimum.

### Downstream (falldet target model), baseline -> final

| Category | Stage-1 real | Stage-1 sim baseline | Stage-1 sim final | P(fall) real | P(fall) sim baseline | P(fall) sim final | `impact_peak_g` gap (nW) |
|---|---|---|---|---|---|---|---|
| fall_forward | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 3.4 -> 1.4 |
| fall_backward | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 6.2 -> 2.6 |
| fall_side | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 10.3 -> 5.1 |
| syncope | 0.95 | 0.00 | 1.00 | 1.00 | 0.68 | 1.00 | 2.1 -> 0.6 |
| fall_recover | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 4.2 -> 1.8 |
| stumble | 0.53 | 0.28 | 0.28 | 0.00 | 0.00 | 0.00 | 1.5 -> 1.5 |
| sit_hard | 0.02 | 0.70 | 0.70 | 0.00 | 0.72 | 0.72 | 4.7 -> 4.7 |
| lie_down | 0.05 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 2.3 -> 2.3 |

- Overall: summed event-level nW 770.9 -> 635.7; 48 features shrank and 14 grew; inside noise 51 -> 64 of
  176.
- The classifier AUC is still 0.98-1.00 in every category.

### Remaining top gaps (final, by impact)

| # | Category | Feature | Real | Sim | nW (CI) | null95 | Owner / type |
|---|---|---|---|---|---|---|---|
| 1 | lie_down | chest rate peak (rad/s) | 2.72 | 0.16 | 6.1 (5.8-6.4) | 0.58 | voluntary motion (motions.py owners) |
| 2 | squat | chest |a| 95th percentile (g) | 1.03 | 1.33 | 2.9 (2.7-3.1) | 0.35 | voluntary motion |
| 3 | fall_backward | chest rate 0.2-0.8 s after impact (rad/s) | 2.19 | 7.05 | 2.9 (2.3-3.5) | 0.46 | falls (trunk topple) |
| 4 | walk_site | hip 3-8 Hz band (log10 g) | -0.90 | -2.06 | 9.1 (8.1-10.1) | 0.65 | mostly VTT sensor mounting |
| 5 | jump_down | waist rate mean (rad/s) | 1.08 | 0.16 | 4.7 (4.4-5.1) | 0.72 | voluntary motion |
| 6 | jog | waist |a| spread (g) | 0.27 | 0.95 | 1.7 (1.6-1.8) | 0.19 | voluntary motion |
| 7 | fall_backward | total chest rotation (rad) | 2.00 | 4.27 | 3.1 (2.5-3.6) | 0.42 | falls |
| 8 | lie_down | chest rate when lying (log10) | -1.80 | -0.59 | 4.9 (4.2-5.4) | 0.97 | lying activity / fidget |
| 9 | fall_backward | waist inclination after (deg) | 85.8 | 92.1 | 1.1 (0.8-1.3) | 0.21 | falls |
| 10 | squat | chest |a| spread (g) | 0.04 | 0.15 | 1.9 (1.8-2.1) | 0.34 | voluntary motion |
| 11 | bend | chest rate mean (rad/s) | 0.13 | 0.49 | 1.0 (1.0-1.1) | 0.18 | voluntary motion |
| 12 | fall_side | waist jerk (g/s) | 43 | 99 | 4.7 (4.1-5.3) | 0.55 | falls |
| 13 | fall_backward | waist inclination change (deg) | 81.8 | 88.4 | 0.9 (0.7-1.1) | 0.19 | falls |
| 14 | jump_down | chest rate mean (rad/s) | 0.93 | 0.21 | 5.1 (4.7-5.5) | 0.92 | voluntary motion |
| 15 | fall_side | waist impact peak (g) | 3.59 | 6.00 | 6.0 (5.4-6.6) | 0.72 | falls |

Ranked lower but systematic:
- fall descent about twice too fast (forward/recover 0.44 s vs 0.92-1.24 s; side 0.60 vs 1.16 s);
- walking pelvis pitch rate 2x real (0.28 vs 0.13 rad/s) and too regular (periodicity 0.80 vs 0.50);
- stumbles (real 3 oscillation peaks and longer settling vs sim 1);
- post-fall stillness (the sim moves more).

### What would close them

1. **Fall descent too fast** (the pivot/kick model topples faster than free fall; README known limit).
   Fit the trip kick and knee-collapse distributions to the Erciyes descent features next, checked
   against the literature's 0.7-1.2 s. Anny geometry was tested and does not fix it.
2. **Backward/side impact peaks still 1.5-1.7x on mats; backward trunk topple too strong.** Before
   touching body stiffness, real data on harder surfaces is needed:
   - **team recordings with the strip** of controlled falls onto a crash mat and onto a firmer gym mat,
     same protocol, ideally with a waist/chest reference sensor to tie them to Erciyes;
   - open real hard-floor fall data is gated (FARSEEING).
3. **Voluntary activities** (lying down 15x too slow, squat and bend rotation, jogging 3x too intense,
   jump-down without rotation, walking pitch and periodicity). These are findings for the motion owners
   (the motionval / Anny work), not tuned here. Their kinematics module can plug in as a feature provider.
4. **Strip-specific realism** (4 anchors, MPU-6050s, the strap, clothing, mount wobble): only **team
   recordings with the strip** can validate it:
   - standing, walking and the site activities, plus mat falls;
   - worn both tight and over clothing, to measure the mid-band energy that VTT shows.
5. **No real data at all** for falls from height, harness arrests, nets, stair falls, struck-by, tools
   and vehicles. Candidates: UNIVRFall (falls from height, worksite) and the 2026 ladder-fall dataset,
   after licence checks.

## 10. What the real data can and cannot tell us

- **Acted falls onto mats.**
  - Erciyes volunteers fall on purpose, from standing, onto mats, and get up. Peaks, jerk, rebound and
    settling on concrete or steel deck are not measured anywhere here.
  - The crash-mat stiffness was fitted (to the softest value searched, x0.2), so it describes Erciyes'
    mats, not site surfaces. Hard-surface behaviour follows only from the contact model's physics.
  - Acted falls are more controlled than real ones: per the biomechanics note, real falls arrive 20-40 %
    slower than an inverted pendulum predicts, and young lab falls at 2.75-3.0 m/s. Acted faints keep some
    muscle control.
- **25 Hz, 12.5 Hz bandwidth.** Anything above about 11 Hz is invisible in Erciyes: heel-strike transients,
  the 22 Hz tool impulses, the 8-25 Hz soft-tissue mount filter. VTT is 100 Hz but clips at 2 g.
- **Protocol differences** are kept out of the impact ranking (weight 0) but reported:
  - Erciyes falls start from quiet standing; the sim trips while walking;
  - Erciyes recoveries are up within about 2 s; FALL-GETUP lies 3-15 s by design;
  - A-SIT is a hard sit by design; Erciyes sits are normal.
- **VTT sensor mounting.**
  - VTT's real walking has 3x more 3-8 Hz energy than Erciyes' real walking, for the same activity, and
    the gap appears in every VTT activity at both sensors.
  - The VTT sensors are on clothing ("trousers", "back"), so most of the large VTT mid-band gap (e.g.
    walk_site hip 3-8 Hz RMS 0.126 g real vs 0.009 g sim) is likely mount wobble, not missing body motion.
  - The simulator was not changed for it. It is a warning that a loosely worn strip will see far more
    mid-band energy than the simulator produces.
- **Mapping caveats:** the Erciyes waist placement (back vs front) and the population ranges are
  [verify]; the chest-to-spine lever arm is research-95's estimate.
- **The classifier test is harsh by design.** With dozens of features, the AUC stays about 1.0 even when
  each single feature is close. Judge progress by per-feature nW against null95.

## 11. Stale results (re-run on the current simulator)

- `thundersim suite` and `shifts` results for every model: fall timing, impact, lying posture and SYNC
  changed. This includes falldet's sim-suite reports in `detection/models/reports`.
- **The frozen `collapse_v1` held-out result in `detection/hub`.** SYNC and SYNC-KNEEL changed the most
  (no Stage-1 trigger before; an impact now). The collapse-hub agent has marked it stale and waits for
  your go-ahead to re-run.
- `calibration.json` (`thundersim calibrate`): its mat peaks change with the new contact and the new
  crash-mat stiffness; not re-run.
- `tuned_baseline.json`, if the baseline was tuned on simulated frames.
- After the motionval / Anny merge: re-run this scorecard plus the detector suite on the combined code as
  the single acceptance check.
- **To restore the old behaviour:**
  - in `thundersim/falls.py`: `CONTACT_RESTITUTION = None`, `ROT_CONTACT_ZETA = 0.4`,
    `ROT_CONTACT_RAMP_S = 0.0`, `TRUNK_TOPPLE_GAIN = 1.0`, `SYNC_SUPPORT = None`,
    `SYNC_BUCKLE_S = (1.0, 2.0)`;
  - in `thundersim/site.py`: `crash_mat` stiffness (1.5e4, 3e4);
  - the curl relaxation and the trunk-landing latch have no switch.

## 12. Not done (round 1; see section 13.8 for what round 2 closed)

- No new datasets were downloaded (machine load, time). **Round 2: UNIVRFall added; UMAFall downloaded.**
- No activity-level or voluntary-motion fixes, by agreement: motions.py belongs to the motion owners.
- The fits are coarse grids of 15-20 runs per point. Several optima are at a grid edge or on a flat ridge
  (crash mat x0.2, trunk-landing zeta 1.2, restitution 0.1-0.4, syncope support 0.6-0.8): treat them as
  fitted but weakly identified.
- The syncope fit was not repeated after the final contact model; `calibrate` was not re-run.
  **Round 2: both done.**
- Runs were kept modest because the machine was busy: 40 runs per event category, 20 clips per activity
  category, one scorecard seed set; the real side uses all trials. **Round 2: second seed set, 60 runs.**
- No README section was added for `realism/`. **Round 2: added.**
- Nothing is committed.

## 13. Round 2 (2026-11-08, after the Anny/motionval merge)

Code state: the live repo after the Anny session's merge (motions.py lifts and tuning, motion_validation.py,
bodies/, viz mocap/contact, motionlib). Round 2 did not edit motions.py, motion_validation.py, viz/,
bodies/, motionlib/ or realism/kinematics.py.

### 13.1 New baseline on the merged code, and what the merge moved

Run `merged/` (40 events per category, same seeds as `final2/`), compared with `final2/`:
- **Fall events: no real change.**
  - summed nW 635.7 -> 681.1;
  - the only features that grew beyond CI are the pre-fall activity spreads (`pre_std_w`: forward,
    backward, recover; sim 0.12-0.13 -> 0.14-0.16 g vs real 0.003). They have weight 0, because the
    Erciyes falls start from quiet standing.
- **Activities:** 700.5 -> 682.4; 50 features shrank, 42 grew; inside noise 88 -> 94.
  - **Better:**
    - pick low band (nW 5.2 -> 1.9);
    - cart posture (pelvis inclination 2.4 -> 0.7, chest 2.4 -> 0.7);
    - stairs chest/pelvis x-axis spread (1.8 -> 0.3, 2.1 -> 1.0);
    - kneel chest posture (1.4 -> 0.4) and rotation rates (2.1 -> 1.1, 1.3 -> 0.3);
    - jog pelvis inclination spread.
  - **Worse (findings for the motion owners):**
    - kneeling is now far too variable: pelvis inclination spread 6.6 deg vs real 1.6 deg (was 0.1); chest
      12.1 vs 3.3 deg; x-gyro spread 0.19 vs 0.05 rad/s;
    - jogging chest z/x acceleration spread 0.53 / 0.38 g vs real 0.09 / 0.09 g;
    - jogging pelvis rate mean 0.88 vs 0.24 rad/s.
- Downstream unchanged except stumble Stage 1: 0.28 -> 0.25 (real 0.53).

### 13.2 New data: UNIVRFall (added), UMAFall (downloaded only)

| Dataset | Licence (checked 2026-11-08) | Status |
|---|---|---|
| UNIVRFall, Zenodo 18346755 | CC BY 4.0, open, no login | downloaded, MD5 matches the record; loader `realism/real_univr.py`; 5 event categories + worksite windows |
| UMAFall, figshare 4214283 v8 | CC BY 4.0 per figshare; the RIUMA mirror says CC BY-NC 4.0 | downloaded (78 MB); not integrated (time); treat as non-commercial until clarified |
| Ladder-fall dataset, BDCC 10(7):227 | article CC BY 4.0; data subset licence unknown | not downloaded: MDPI blocks automated access; GitHub subset not found |

The manifest is `research/fall_model/data/raw/REALISM_DATA_MANIFEST.json`.

UNIVRFall facts **measured from the files**, which differ from the dataset page:
- one lower-back sensor (L1-L2), 100 Hz;
- the accelerometer is in mg and **clips at +/-3997 mg per axis** (the page says +/-16 g), so impact peaks
  above 4 g per axis are invisible (emulated the same way on the sim);
- the gyroscope is in mdps with an uncalibrated bias (removed per recording);
- **tasks 41/42 are backward falls while climbing up/down STAIRS** (label sheets), not a ladder;
- the "falls from height" (39/40) show only 0.17-0.25 s of free fall, so they are low platforms.
- Every fall has video-labelled onset and impact frames. The construction-site recordings have their own
  format (type == 1 rows are IMU).

The archive is parsed with own code only (CSV, plus the label .xlsx read as zipped XML with a size limit).
Zip-slip was checked before extraction.

New mapping rows (lower-back sensor emulated at the lumbar anchor in both slots; duplicated chest
magnitude/posture features dropped; `label_dur_s` = labelled onset -> impact, sim = ground-truth
`fall_onset` -> first `impact`):

| Category | UNIVRFall task | Simulator | Confidence |
|---|---|---|---|
| u_trip | 30 trip while walking | L-TRIP | high |
| u_slip_back | 34 backward slip while walking | L-SLIP | high |
| u_faint_walk | 28 fainting while walking | SYNC (stands 5-20 s first) | medium |
| u_height | 39/40 fall from a low platform | H-LOW: `height_fall(start="edge")` from 0.4-0.8 m [verify height] | low |
| u_stairs | 41/42 backward fall on stairs | STAIR (3-12 steps; real: a few steps onto mats [verify]) | low |
| u_site (activity) | OnField, 10 workers, 42 h, 6 random 5-min chunks per file | fall-free shift blocks (SIM_SPEC 5.2 mix), 0.15 h per seed | medium |

UNIVRFall maps two previously unmapped areas: low falls from height and stair falls. H-SCAF/H-LAD heights
(> 1 m) are still not covered by any open data; harness arrest and nets have none.

### 13.3 Iteration 4: fall descent too fast (KEPT)

- **Gap.** Forward/side/recover falls descended about twice as fast as real (rotation 10-90 % in 0.44 /
  0.60 s vs Erciyes 0.92 / 1.16 s). UNIVRFall's video labels make this sharper:
  - walking trips go from onset to impact in 0.62 s (IQR 0.54-0.71) vs sim 0.41 s;
  - sim trip hip speeds were 3.6-4.0 m/s, against 2.1-3.0 m/s for real falls (Choi 2015: real pelvis
    impact velocity 38 % below inverted-pendulum predictions).
- **Cause.** In `Faller.pivot` a walking trip starts with the walking rotation about the stance foot
  (about 1.1-1.45 rad/s) plus a 0.6-1.4 rad/s kick, and nothing resists the fall. This is the research
  report's "trips topple at 1.3-1.7 g".
- **Change** (`thundersim/falls.py`; no motions.py edit):
  - new constants `TRIP_KICK = (0.6, 1.4)` (unchanged range, now named), `PIVOT_MOMENTUM_KEEP = 0.5` and
    `PIVOT_RESIST = 2.0` (originals: implicit 1.0 and 0.0);
  - `Faller.pivot` gains `keep` (the rest of the walking rotation is absorbed smoothly over the kick time,
    so there is no velocity step) and `resist` (protective rotational resistance, 1/s, ramped in over
    0.1 s);
  - `same_level_fall` passes them for trips and knock-downs only (slips keep 1.0 / 0.0) and allows 3 s
    instead of 1.5 s for the pivot. RNG order is unchanged.
- **Fits** (`realism/fit.py`; fit seeds; categories: Erciyes forward/side/recover plus u_trip):
  - `descent_keep`: 0.5 beat 1.0 and tied 0.2. Momentum alone barely moved the descent (0.44 -> 0.52 s).
  - `descent_resist`, `descent2` (resist 0-6): resist 4-6 best matches the rotation-spread descent, but
    overshoots the labelled onset->impact (0.86-1.0 s vs 0.62 s).
  - **resist 2** matches the labelled time (sim 0.63 s vs real 0.62 s) and gives the best overall
    objective. The rotation-spread window (-3..+1 s) also counts post-impact rolling, which the damped
    sim landing does not have, so it is not a clean pre-impact measure.
- **A/B on the full scorecard** (`pre_descent/` with `--set` the old constants vs `descent/`; same seeds;
  251 event features):
  - 17 shrank beyond CI and 6 grew; inside noise 106 -> 107; activities unchanged.
  - u_trip labelled onset->impact: sim 0.41 -> 0.62 s vs real 0.62 (nW 1.78 -> 0.35, inside noise).
  - waist impact peak, nW: forward 1.93 -> 1.00 (inside), side 5.97 -> 2.35, recover 1.91 -> 0.62
    (inside).
  - jerk, nW: side 4.72 -> 2.85, recover 1.32 -> 0.67.
  - rotation-spread descent, nW: forward 2.00 -> 1.34, side 2.52 -> 2.00, recover 3.12 -> 2.50.
  - falldet `impact_peak_g` gap: forward 1.36 -> 0.40, side 5.08 -> 2.33, recover 1.84 -> 0.61 (all
    inside noise except side).
- **Grew:**
  - chest rotation peak in forward/side/recover (sim 4.1-4.5 vs real 5.0-6.3 rad/s; the resistance slows
    the upper trunk too);
  - side post-impact minimum (0.45 -> 0.21 vs real 0.57 g);
  - forward settling;
  - UNIVRFall trip free-fall duration.
  - Section 13.6 shows that about 6 % of features cross their CI between two seed sets of identical code,
    so 6 "grew" out of 251 is at chance level; the 17 "shrank" are not.
- **Hazard side effect** (site falls on their own surfaces, 44 Hz; `site_impacts --keys
  PIVOT_MOMENTUM_KEEP,PIVOT_RESIST`; median g):
  - L-TRIP 16.5 -> 11.6; STRUCK-KD 16.1 -> 12.5; H-LAD 22.2 -> 22.9 (its secondary topple);
  - L-SLIP, SYNC, H-SCAF and STAIR unchanged;
  - trip hip speeds 3.6-4.0 -> 2.7-2.8 m/s.
- **Tests:**
  - new `test_realism.test_trip_descent_is_not_faster_than_real` (median onset->impact > 0.5 s, median
    hip speed < 3.2 m/s);
  - `test_regression.test_current_firmware_gap_is_visible`: L-TRIP seed 3 -> 1. The current firmware
    (0.5 Hz, +/-2 g) still misses 6 of 10 trips (seeds 0-9; it missed 9 of 10 before; the target firmware
    confirms 10 of 10), but seed 3 is now caught by its posture rules. **Finding: with realistic, slower
    trips, the current firmware looks less hopeless than before; the gap is still large.**
  - `test_realism.test_no_rebound_flight_after_body_impact`: now 12 seeds per category and threshold
    0.33 g (was 6 seeds, 0.4 g). Per-run minima spread from 0.03 to 0.82 g; the 24-run median is 0.40 and
    the 40-run scorecard medians are 0.52-0.58, while the old bounce medians were 0.15-0.31.

### 13.4 Syncope re-fit under the final contact model (DONE, no change)

- Grid `sync3`: support 0.5-0.8 x buckle (0.4-0.8 s, 0.8-1.2 s); Erciyes syncope plus UNIVRFall fainting
  while walking.
- The current values (support 0.7, buckle 0.4-0.8 s) are still best (objective 48.6; next 50.3).
- The Erciyes post-impact minimum stays a gap (nW 1.36 on fit seeds).
- **New conflict between datasets:**
  - UNIVRFall acted faints go from labelled onset to impact in 0.67 s and land at 5.6 g at the lower back;
  - Erciyes faints land at 3.5 g at the waist;
  - the sim takes 1.1-1.4 s (u_faint_walk `label_dur_s` nW 4.9) and lands at 3.6 g.
  - Lower support moves toward UNIVRFall and away from Erciyes. The datasets disagree, so this was left.

### 13.5 calibrate re-run (DONE; calibration.json NOT rewritten)

- `calibrate(runs=40, write=False)` with the real processed data; output in
  `research/artifacts/realism_runs/calibrate/`.
  - The CLI has no `--out`, always writes, and its default data path does not exist on this machine, so
    the function was called directly.
- **Simulated mat peaks now match Erciyes at every soft-tissue scale:** median 3.86-4.03 g vs real 3.95 g;
  quartiles 3.4 / 3.9 / 4.6 vs 3.33 / 3.95 / 4.64; log-quantile error 0.034-0.079. They were 6.8 g with
  error 0.49-0.58. The old 1.7x "calibration finding" is closed by the fall-physics fixes, not by soft
  tissue.
- **Not written:**
  - the rule would pick `fn_scale` 0.4, a grid edge chosen only by the 90th percentile, and Erciyes cannot
    constrain the mount filter (transparent below 12.5 Hz);
  - the scorecard runs with scale 1.0 while the runner reads 1.5 from the file.
  - Recommendation: decide `fn_scale` with the team's own strip recordings. The README "Calibration
    finding" section is updated.

### 13.6 Bigger run on a second seed set (DONE)

- `final_seed2/`: 60 events per category, seeds offset by 400 (`--seed-offset 400`), current code.
- Against `descent/` (40 events, original seeds, identical code):
  - summed event nW 860.5 vs 882.9 (+2.6 %);
  - 16 of 251 event features (6 %) differ beyond their CI, 11 of 433 activity features.
  - This calibrates the keep/revert rule: a few crossings per comparison are chance-level.
- **Runtime:**
  - 11.2 min for this 60-event run at load average about 25;
  - 14-15 min for the 40-event runs at load average 25-35;
  - 51.5 min for the first merged baseline at load average 70-100;
  - the UNIVRFall real cache builds once (about 1 min for the events, longer for the site chunks).
- Command: `.venv/bin/python -m realism score --out <dir> [--events 60] [--seed-offset 400]`.
- `--set module.NAME=VALUE` scores old constants on current code (A/B).

### 13.7 Measured candidate, NOT kept: backward-fall trunk topple

- **Gap:** backward falls slam the trunk after the pelvis lands. Chest rate 0.2-0.8 s after impact:
  - Erciyes backward: 7.0 vs 2.2 rad/s;
  - UNIVRFall slips: 7.3 vs 0.9 rad/s, with post-impact minimum 0.30 vs 0.64 g.
- **Grid `topple`** (`TRUNK_TOPPLE_GAIN` 0.5 / 0.35 / 0.25):
  - gain 0.25 fixes the slam (Erciyes backward nW 3.31 -> 0.49; UNIVRFall slip 5.32 -> 1.59);
  - but backward falls no longer finish lying (pelvis inclination nW 1.25 -> 2.03), and forward/side
    rotation and settling get worse.
- The constant is global and the landing sequence of a sitting-first backward fall needs its own
  structure, so it was not kept. Next step: a backward-specific trunk landing, fitted to both datasets.

### 13.8 Final state after round 2 (`final_seed2/`, 60 events per category)

**Classifier AUC:** still 0.97-1.00 in every category, including the UNIVRFall ones (u_trip 0.98,
u_faint_walk 0.98, u_height 0.99, u_slip_back 1.00, u_stairs 1.00, u_site 1.00).

**UNIVRFall categories** (real median / sim median, nW; * = outside real-data noise):
- **u_trip:** labelled onset->impact 0.62 / 0.62 (0.31); rotation descent 0.90 / 0.64 (1.00); peak
  4.96 / 4.24 g (1.13); pelvis rate peak 6.3 / 4.7 rad/s (0.98); post-impact minimum 0.54 / 0.80 g (1.21*).
- **u_slip_back:** onset->impact 0.58 / 1.02 s (4.12*); peak 5.21 / 4.09 (2.25*); rate peak 15.5 / 4.9
  rad/s (1.30*); post-impact minimum 0.64 / 0.30 (3.00*).
  - Sim slips take too long by the labels: the foot slides out, then the pendulum falls unassisted.
- **u_faint_walk:** onset->impact 0.67 / 1.25 s (4.90*); peak 5.57 / 3.55 g (2.06*).
- **u_height** (low platform): onset->impact 0.95 / 0.84 s (0.76); peak 5.60 / 5.18 (0.77); minimum before
  impact 0.09 / 0.09 (0.13); rate peak 14.1 / 4.4 rad/s (1.41*).
- **u_stairs:** onset->impact 0.64 / 0.33 s (1.78*); post-impact |a| spread 0.07 / 0.96 g (27*: STAIR
  tumbles 3-12 steps, real falls are a few steps onto mats; mapping-limited).
- Real lower-back rate peaks in slips, height and stair falls (14-22 rad/s) are 3-5x the sim's (about
  5 rad/s). The sim has no fast pelvis spin at contact. This could also partly be belt-sensor wobble
  [verify].
- **u_site** (worksite, 2592 real vs 876 sim windows):
  - sim posture is more inclined (pelvis 9.6 vs 6.0 deg, nW 2.0);
  - less 3-8 Hz energy (log10 -2.90 vs -1.70, nW 1.2);
  - up-axis rate spread 0.02 vs 0.28 rad/s;
  - dominant frequency 1.75 vs 2.0 Hz.
  - Unlike VTT, this sensor is on the lower back, not on clothing, so the mid-band deficit is more likely
    real (missing small trunk motion during work) [verify the mounting].

**Remaining top gaps (by impact, `final_seed2/`):**
1. Stairs post-impact |a| spread (27 nW; mostly a mapping limit).
2. UNIVRFall backward slip: rebound, trunk rate after impact, rate-peak timing, onset->impact time
   (7.2 / 7.5 / 14.6 / 4.1 nW). The backward trunk landing (13.7) and slip timing are the next fall
   fixes.
3. Lying down too slow (chest rate peak 0.16 vs 2.72 rad/s) and too restless lying (voluntary motion;
   motion owners).
4. Squat chest peaks, jog intensity, jump-down rotation (voluntary motion).
5. Erciyes backward trunk slam (3.1 nW) and side impact peak (2.35 nW).
6. Fainting: onset->impact and peak, which conflict between the two datasets.
7. Worksite posture and mid-band texture (u_site).

**Data that would close them:**
- team recordings with the strip: mat falls (trip, slip, faint, sideways) with video onset/impact labels
  like UNIVRFall, and site work;
- a few-step stair-fall protocol matched to UNIVRFall;
- confirmation of UNIVRFall's platform height and sensor mounting;
- the BDCC ladder subset, if its licence allows.

**Not done in round 2:**
- UMAFall loader (downloaded only);
- the ladder dataset (not obtainable automatically);
- the backward trunk landing (measured, not kept);
- the faint-timing conflict (left);
- `calibration.json` `fn_scale` decision (left to the team);
- activity-level fixes (motion owners).
- Nothing is committed.

**Stale after round 2:** everything in section 11, plus anything that relies on trip/knock-down impact
or timing:
- the `suite`/`shifts` numbers;
- the current-firmware miss rate (now 6 of 10 trips instead of 9 of 10 on seeds 0-9);
- falldet sim-suite reports;
- `collapse_v1`, if it used trips or knock-downs (SYNC itself is unchanged in round 2).

## 14. Combined acceptance run (frozen simulator, 2026-11-08)

### 14.1 Simulator pin

- Pinned files: `thundersim/**/*.py` (37 files) plus `calibration.json`.
- Manifest: one `sha256  path` line per file, sorted by path (`LC_ALL=C`); saved as
  `research/artifacts/realism_runs/accept_pin_manifest.txt`.
- **sha256 of the manifest: `f05639c99ccacc4635258cbc48aa5fecaf817fc790beb995fb3b071a3b8fd68a`.**
- Recomputed after both runs: unchanged.
- Key files: falls.py `f1220918…f9089e`, motions.py `1047354c…a56bbc`, site.py `7a99d506…c20396`,
  spine.py `d45b37d7…0eb0ec2`, calibration.json `aaf037be…466b0953`.

### 14.2 Measurement fixes made before the run (scorecard only; no simulator change)

Both claims from the Anny session were checked on my own caches before acting on them.

**1. Activity windowing artefact: confirmed, smaller than reported.**
- Erciyes activity trials are 32-56 % quiet windows (walk 32 %, jog 35 %, squat 45 %, bend 56 %); sim clips
  are 0-29 %. VTT has no such problem (0 % on both sides).
- **Fix:** `ActivityCategory.active_only` for the four Erciyes activities. Windows that are quiet (pelvis
  rate mean < 0.15 rad/s AND |a| spread < 0.03 g, [design]) are dropped on BOTH sides
  (`score.active_mask`).
- **Old -> new** (rescore of the same simulations, `final_seed2/` -> `final_seed2_active/`):

| Activity | Feature | Sim/real ratio before | After |
|---|---|---|---|
| jog | waist \|a\| spread | 3.4x | 1.9x (Anny session: 1.4x) |
| walk | pelvis rate mean | 2.3x | 1.6x |
| walk | pelvis pitch-rate spread | 2.2x | 1.8x (they reported 0.196 vs 0.276) |
| bend | chest rate mean | 3.6x | 1.4x (they: 0.36 vs 0.49; agrees) |
| squat | chest rate mean | 2.9x | 1.8x |

  - Features inside real-data noise: walk 7 -> 11, jog 4 -> 6, bend 1 -> 5, squat 2 -> 9 (of 35).
  - Summed nW rises (walk 31 -> 45), because the real distributions narrow and nW is scaled by their
    spread. Judge by the ratios and the noise-floor count.
  - AUC stays 0.98-1.00.

**2. Lie-down alignment artefact: confirmed.**
- The sim search window spanned the lying activity after the lie-down, so the event could align after
  the rotation.
- **Fix:** `emulate.event_run` searches only the lie-down move.
- Chest rate peak: sim 0.16 -> 1.85 rad/s (real 2.72); nW 6.1 -> 2.0.

**3. lie_down `still_acc_w` does not respond to motion tuning: explained (protocol, plus a small sensor
note).**
- Acceleration noise floor matches: Erciyes' quietest 1 s per trial reads 0.0015 g at the waist (median;
  p10 0.0010) vs the emulated 0.0010 g, and sim lying reads 0.0014 g. Real "lying" (0.0027 g) is people
  settling onto a bed.
- The sim category follows the lie-down with A-LIE's lying *work* (chest rate 0.11 vs 0.016 rad/s). So
  the still_* features compare resting on a bed with working on the ground: a protocol mismatch, and
  motion tuning cannot fix it.
- **Proposal (measurement only, not applied):**
  - compare lie_down against a resting lie, not A-LIE's lying work;
  - raise the emulated Erciyes gyro noise from 0.003 to about 0.01 rad/s (real quiet floor 0.008-0.013
    rad/s).

**4. VTT waist about 2x Erciyes in walking: agrees** with section 10. Real VTT walking has about 3x
Erciyes' 3-8 Hz energy for the same activity; a clothing mount is the likely cause.

### 14.3 Acceptance results

- **Runs:** `accept_s0/` and `accept_s400/`: 60 events per category, 20 clips per activity category, two
  independent dev seed sets.
- **Runtime:** 29 min each, run concurrently at load average 20-35. The kinematics provider is included
  (motionval scenarios, seeds 0-9, 16 s).

| | Seed set 0 | Seed set 400 |
|---|---|---|
| Event features (251): summed nW / inside noise | 929.1 / 110 | 954.8 / 108 |
| Activity features (433): summed nW / inside noise | 641.5 / 149 | 636.7 / 153 |
| Kinematics rows (88; 37 with >= 5 mocap clips per side): inside noise | 35 of 37 | 35 of 37 |
| Classifier AUC range | 0.975-1.000 | 0.965-1.000 |

- **Replication:** 11 of 684 features (1.6 %) differ beyond their CI between the two seed sets.
- **Event sums vs earlier sections:** the higher totals come from the 5 UNIVRFall categories, which were
  not in sections 9 and 13.6.
- **Activity sums vs round 2 (`final_seed2`):** 680.5 -> 636.7, and inside noise 109 -> 153, from the
  Anny tuning and the active-window fix together.

**Downstream (falldet target model), seed set 0 / 400:**

| Category | Stage 1 real | Stage 1 sim | P(fall) real | P(fall) sim | `impact_peak_g` gap (nW) |
|---|---|---|---|---|---|
| fall_forward | 1.00 | 1.00 / 1.00 | 1.00 | 1.00 / 1.00 | 0.44 / 0.61 (inside noise) |
| fall_backward | 1.00 | 1.00 / 1.00 | 1.00 | 1.00 / 1.00 | 2.60 / 2.55 |
| fall_side | 1.00 | 1.00 / 1.00 | 1.00 | 1.00 / 1.00 | 2.39 / 2.58 |
| syncope | 0.95 | 1.00 / 1.00 | 1.00 | 1.00 / 1.00 | 0.60 / 0.57 (inside) |
| fall_recover | 1.00 | 1.00 / 1.00 | 0.00 | 1.00 / 1.00 (protocol) | 0.64 / 0.69 (inside) |
| stumble | 0.53 | 0.17 / 0.20 | 0.00 | 0.00 / 0.00 | 1.60 / 1.46 |
| sit_hard | 0.02 | 0.73 / 0.60 | 0.00 | 0.76 / 0.77 (hard sit by design) | 4.56 / 4.61 |
| lie_down | 0.05 | 0.00 / 0.00 | 0.00 | 0.00 / 0.00 | 1.26 / 1.27 |

**Top gaps by impact** (seed set 0; seed set 400 in brackets; nW):
1. u_stairs post-impact |a| spread 26.2 [29.2]: sim 0.89 vs real 0.07 g. STAIR tumbles 3-12 steps;
   UNIVRFall falls a few steps onto mats. Mostly a mapping limit.
2. u_slip_back: rebound 5.7 [7.1], chest rate after impact 6.9 [7.5] (sim 6.6 vs real 0.9 rad/s),
   onset->impact 4.5 [4.3] (1.05 vs 0.58 s), rate-peak timing 12.1 [14.1], settling 4.4 [5.3].
3. fall_backward chest rate after impact 3.1 [3.0] (7.1 vs 2.2 rad/s) and total rotation 3.0 [2.6].
   This is the backward trunk slam (13.7).
4. bend jerk (log) 5.3 (sim -1.42 vs real -0.27): voluntary motion.
5. u_faint_walk: chest rate after impact 5.2 [4.4], onset->impact 4.6 [4.9] (1.18 vs 0.67 s).
6. jump_down pelvis/chest rate mean 4.5 / 5.0 (0.19 vs 1.08 rad/s): voluntary motion, no rotation.
7. sit_hard post-sit chest stillness 3.9.
8. fall_forward post-fall stillness 2.0 (sim quieter than real).
9. stairs (VTT) jerk 4.3: ladder vs stairs mapping.
10. jog waist |a| spread 2.1 (0.93 vs 0.47 g, active windows).
- **Kinematics provider:** the largest gap is the kneeling upper-trunk inclination 10th percentile, sim 27
  vs mocap 2.3 deg (nW 16.8). Then walk trunk-rate p95 (0.73 vs 1.18 rad/s) and kneel pelvis share
  (0.60 vs 0.93).

### 14.4 Proposals (NOT applied; the simulator is frozen)

1. **Backward-fall trunk landing.** Give a sitting-first backward fall its own landing:
   - pelvis, then the trunk lowered by an extensor-damped topple;
   - fitted on Erciyes 909-912 plus UNIVRFall 34, together with the slip timing (onset->impact 1.05 vs
     0.58 s).
   - A global `TRUNK_TOPPLE_GAIN = 0.25` fixes the slam but breaks the final posture (13.7).
2. **Stair-fall mapping.** Add a few-step backward stair fall to match UNIVRFall 41/42, or restrict u_stairs
   to impact-free features. Without it, the stairs gap stays dominated by the mapping.
3. **Faint timing.** UNIVRFall's acted faints (0.67 s) and Erciyes' faints (support-fitted) disagree; it
   needs a third source (team recordings).
4. **Measurement:**
   - compare lie_down against a resting lie;
   - raise the emulated Erciyes gyro noise to about 0.01 rad/s;
   - scale the stairs category to fewer steps.
5. **Voluntary motion (motion owners):** bend jerk, jump-down rotation, jog intensity (1.9x on active
   windows), kneeling upper-trunk inclination.

### 14.5 Review of `tests/test_viz_props.py` (resting-frame rule)

- **The rule:** `RestingProps.test_sacks_stand_on_the_ground` now counts a sack frame as "resting" only if
  BOTH neighbouring 20 ms steps are zero after the payload's 0.1 mm rounding (`still = zero[i-1] &
  zero[i]`), instead of a single zero step.
- **Reason given:** a held sack moving slowly (quiet stand, start of a put-down) can show one rounded-zero
  step.
- **Verdict: it does not hide a real problem.**
  - A sack that truly rests on the ground produces long runs of zero steps, so it is still checked.
  - A sack frozen in mid-air for 2 or more frames (40 ms) would still fail.
  - Only an isolated single-frame stop is skipped, and that is not a resting state.
- **Residual weakness:** "moving" is inferred from the rounding. A prop drifting slower than about 5 mm/s
  while held in the air would never be checked; a held sack paused for 40 ms or more in the air would
  fail (a false failure, not a hidden one).
- **Better rule:** use the payload's held/attached state, if the export has one.
- I could not diff against the previous version (the file is untracked); the review is of the current
  rule and its comment.

