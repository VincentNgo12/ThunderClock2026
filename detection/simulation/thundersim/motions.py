# File: motions.py
# Purpose: Builder that strings voluntary motion primitives (stand, walk, run, stoop, kneel,
#          crawl, ladder, sit, lie down, tools, equipment...) into one continuous PoseTrack.
# Contributors: Unassigned
"""Voluntary motion primitives. Physics-driven falls live in falls.py and use the same Builder.

Postures are kinematic keyframes joined by minimum-jerk transitions whose duration is
stretched so the trunk never exceeds MAX_VOLUNTARY_RATE. Oscillations (gait, vibration,
breathing, sway, fidget) are added on top. All amplitudes are [design] values unless a
source is named; they were chosen to land in the ranges SIM_SPEC.md section 4 describes.
Values tagged [src: motionval ...] were checked against real motion with `thundersim
motionval` (thundersim/motion_validation.py): "mocap" means the class A clips in
motionlib/canonical, as a per-clip median (10th-90th percentile).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .site import SURFACES, Site
from .spine import (MAX_VOLUNTARY_RATE, PHASE_CODE, PoseTrack, Worker, clamp_joints, soft_limit)

G = 9.80665

# --- lifting (Builder.lift / Builder.put_down) ------------------------------------------------
# Sources (numbers tagged below):
#   [Straker 2003] via Washmuth et al. 2022, Int J Sports Phys Ther 17(1):104: stoop < 45 deg knee
#       flexion and ~90 deg trunk flexion; semi-squat ~90 / ~45; squat ~135 / < 30 (idealised).
#   [Plamondon 2014] Appl Ergon 45(3):471: 15 kg boxes from ground level, knee flexion ~71 deg
#       (experts) and ~48 deg (novices).
#   [von Arx 2021] Front Bioeng Biotechnol 9:769117: 30 adults, 15 kg box from the floor. Lumbar
#       flexion range of the lift-up: stoop 35.9 +- 10.1, freestyle 24.2 +- 7.3, squat 25.1 +- 8.2 deg.
#       Lift-up time in Hof (1996) units sqrt(l0/g): stoop 5.9 +- 1.1, freestyle 4.6 +- 0.7, squat
#       4.9 +- 0.7, i.e. about x 0.30 s for a 0.9 m leg.
#   [Burgess-Limerick 1995] Human Factors 37(2):395: early in a self-selected lift, knee extension
#       leads hip extension, which leads lumbar extension; the lead grows with load mass.
#   [Burgess-Limerick & Abernethy 1997] Int J Ind Ergon 19(5):413: 39 adults; semi-squat starts are
#       usual; stoops appear on some trials, more with light loads, and only semi-squats at the higher loads.
#   [Saraceni 2021] PLoS One 16(7):e0254241: 41 adults, self-selected floor lifts of a box at 10 % of
#       body mass (no-LBP group, lift 95): peak knee flexion 85.9 (95 % CI 73.2-98.6) deg, peak thorax
#       inclination 102.8 (97.3-108.3) deg, pelvis inclination at lift-off 51.3 (45.0-57.6) deg, peak
#       intra-lumbar flexion 21.8 (19.7-23.8) deg, peak thorax velocity 138 (126-150) deg/s.
#   [Geissinger 2020] IISE Trans Occup Ergon Hum Factors 8(2):88: 666 bends of retail stockers at work:
#       27.3 % stoops, 20.3 % split-legged bends with both heels down, 12.3 % with a heel raised,
#       22.1 % one-legged lifts, 4.6 % squats (the rest mixed).
LIFT_TECHNIQUES = ("stoop", "semi_squat", "squat")
# Share of workers who habitually use each technique [src: Geissinger 2020, the three symmetric
# classes renormalised: stoop 27.3, split-legged bends (knees bent, both heels or one heel down)
# 32.6 -> semi-squat, squat 4.6; one-legged and mixed lifts are not modelled] [design: mapping].
LIFT_TECHNIQUE_SHARE = {"stoop": 0.42, "semi_squat": 0.51, "squat": 0.07}
# Share of the hands' drop to the grip taken by the legs (pelvis drop); the trunk takes the rest
# [design, tuned for a 1.75 m worker at the mean lumbar flexion and share; knee angles are absolute
# flexion, about 14 deg of it already in the standing pose (knee_at_stand)]:
#   - semi-squat to a 0.15 m grip: knee 87, thorax 97, pelvis 56 deg, at or inside the Saraceni 2021
#     self-selected CIs (higher grips bend less);
#   - stoop: the knees stay under Straker's 45 deg only for grips of about 0.25 m and above (40 deg at
#     0.25 m, 39 at 0.30 m), and the lumbar and share draws move that by several degrees (47-48 deg at
#     0.27-0.28 m in A-CARRY). Below that the trunk reaches TRUNK_MAX and lift_posture bends the knees
#     further (57 deg at 0.20 m, 68 at 0.15 m), which is a semi-squat by knee angle;
#   - squat to a 0.15-0.25 m grip: knees 105-114 deg, pelvis drop 0.33-0.39 m, close to the offline
#     Anny solve of the A-CARRY sack (0.40-0.44 m, knee 110-117 deg); grips above about 0.22 m stay
#     under 112.5 deg, a semi-squat by knee angle.
# The events therefore carry both the planned technique and the class of the knee angle reached
# (knee_class, KNEE_CLASS_DEG).
LIFT_LEG_SHARE = {"stoop": 0.08, "semi_squat": 0.38, "squat": 0.65}
# Technique class by the knee flexion reached: stoop below 45 deg [src: Straker 2003 via Washmuth 2022],
# squat from 112.5 deg, the midpoint of Straker's semi-squat ~90 and squat ~135 [design]; the same
# boundaries as motionlib/tools/annotate.py uses for the mocap lifts.
KNEE_CLASS_DEG = (45.0, 112.5)
LIFT_LUMBAR_DEG = {"stoop": (35.9, 10.1), "semi_squat": (24.2, 7.3), "squat": (25.1, 8.2)}  # [src: von Arx 2021]; freestyle for semi-squat [design]
LIFT_UP_S = {"stoop": (1.8, 0.3), "semi_squat": (1.4, 0.2), "squat": (1.5, 0.2)}  # [src: von Arx 2021, x 0.30 s]
LIFT_DOWN_S = (1.1, 1.6)            # lowering the body to the grip (s) [design]
LIFT_MIN_HOLD_S = 0.05              # shortest set-and-grip hold (s) [design]
# Set-and-grip hold at the lowest posture (s). Real bends turn round quickly: 0.44 s (0.23-0.81)
# above 90 % of their deepest trunk angle and 0.31 s above 95 % [src: motionval, 27 mocap bends].
# A minimum-jerk stop with the old 0.1-0.4 s hold and later trunk start gave 1.15 s and 0.98 s
# (eval/bend). The rounded turn (LIFT_TURN) and the rise already account for most of the 0.44 s,
# so the hold itself is very short [design]. Time left in a carry's fixed budget is spent standing
# upright after the rise.
LIFT_HOLD_S = (0.05, 0.15)
LIFT_THORAX_RATE = np.deg2rad(150.0)  # peak thorax inclination rate [src: Saraceni 2021, 138 (126-150) deg/s; upper CI]
# Lift-up coordination windows as fractions of the lift-up: legs (knee extension, pelvis rise),
# pelvis pitch (hip extension) and spinal flexion (lumbar extension) [design ordering from
# Burgess-Limerick 1995]; the spine starts later with heavier loads.
LIFT_LEGS_WINDOW = (0.0, 0.85)     # the hips rise through most of the lift-up (von Arx 2021 times the whole phase)
LIFT_SPINE_LEAD = (0.10, 0.20)      # spine start at 0 kg .. at 25 kg and above; the hips start at half of it
#                                     [design; was 0.20-0.30, shortened with LIFT_HOLD_S to the mocap hold]
# The lowering arrives at the grip still decelerating, and the rise departs with the same
# acceleration (window_ramp arrive / depart, in units of the change per move^2): a rounded turn
# instead of a flat stop. [design, tuned with motionval: time above 50 / 90 / 95 % of the deepest
# trunk angle 1.33 / 0.65 / 0.52 s against mocap 1.29 / 0.44 / 0.31 (was 1.84 / 1.15 / 0.98);
# thorax rate at the 90 % crossings 75 / 76 deg/s against mocap 66 / 68 (was 65 / 68)]
LIFT_TURN = 10.0
# Body proportions for the lift geometry, fractions of height [design: the same Drillis-style
# proportions the 3D viewer draws, viz/rig.py Dims].
THIGH_H, SHANK_H, ANKLE_H = 0.24, 0.235, 0.039
HIP_BELOW_ROOT_H, HIP_ANTERIOR_H = 0.02, 0.075 / 1.75     # hip joint from the spine root, pelvis frame
SHOULDER_UP_H, SHOULDER_ANTERIOR_H = 0.022, 0.055 / 1.75  # shoulder from the upper-thoracic anchor
ARM_H = 0.18 + 0.19                                         # shoulder to grip
LIFT_REACH = 0.97                   # arm length used to reach the grip: the load hangs on near-straight arms [design]
SHANK_TILT_MAX = np.deg2rad(30.0)   # forward shank tilt (ankle dorsiflexion) [design; Anny probe 23-31 deg]
KNEE_MAX = np.deg2rad(125.0)        # deepest knee flexion used [design; Anny squat probe 110-117 deg]
TRUNK_MAX = np.deg2rad(120.0)       # deepest upper-trunk inclination [design]
PELVIS_PITCH_MAX = np.deg2rad(60.0)  # deepest pelvis inclination in a forward bend [design]
OBJECT_CLEARANCE_M = 0.15           # grip this far ahead of the knee: half a 0.16 m sack plus the knee [design]
LUMBAR_SHARE = 0.45 + 0.4 * 0.35    # L1-S1 share of the model's spinal flexion (joint 0 + 2/5 of joint 1) [design]
SPINE_FLEX_SPLIT = np.array([0.45, 0.35, 0.20])  # as stoop() [design]

# --- squatting (Builder.squat) ----------------------------------------------------------------
# [src: Erciyes 804, 95 squat episodes, eval/diag/bend_squat_shape: per episode the waist sensor
# inclines 20 (7-34) deg and the chest 34 (20-62); 10-90 % of the chest inclination takes 0.60 s
# (0.38-2.9) going down and 0.96 s (0.44-3.3) coming up]. The sim's 30 deg pelvis and 18 deg spine
# gave 30 and 49 deg, and its 1.0 s rise 0.52 s. The pelvis drop (0.30 x height) and the 1.0 s
# descent already give the real waist |a| range (min 0.78 vs 0.77 g, 95th percentile 1.21 vs 1.19).
SQUAT_PELVIS_DEG = 20.0
SQUAT_FLEX_DEG = (6.0, 5.0, 3.0)    # spinal joints, low to high [design split of the 14 deg]
SQUAT_DOWN_S = 1.0                  # [design; see above]
SQUAT_UP_S = 1.75                   # minimum-jerk 10-90 % is 0.55 of the move: 0.96 s


# --- gait, ladder, kneeling and standing amplitudes (checked with motionval) ----------------------
# Walking. Amplitudes are per side (peak-to-peak is twice the value); bob is peak-to-peak.
WALK_BOB_M = 0.030          # pelvis vertical bob p-p at 1.3 m/s [src: motionval: mocap 2.9 cm (2.2-4.7);
                            # Orendurff 2004 COM 2.7-4.8 cm at 0.7-1.6 m/s] (was 0.025 [design])
                            # With WALK_BOB_EXTRA the stride waveform is 3.06 cm p-p. WALK_STEP_VARIATION then
                            # spreads it from step to step: the mean per-step p-p is about 3.5 cm (3.16 without the
                            # variation; motionval bob_cm 3.13, mocap 2.85) [eval/fix_0/bob_pp_breakdown.txt]
WALK_YAW_DEG = 6.0          # pelvis rotation [src: motionval: mocap p-p 14.7 (8.3-17.2); Latorre 2019 10-13] (was 4)
WALK_ROLL_DEG = 4.0         # pelvis obliquity [src: motionval: mocap p-p 10.9 (3.5-13.3); Latorre 2019 6.5-7.6] (was 3)
                            # Obliquity and side sway stay in phase with the rotation: mocap puts them ~90 and ~75 deg
                            # later, but then the waist's lateral force and tilt cancel (Erciyes 801 lateral 0.12 g and
                            # tilt SD 1.7 deg, sim 0.02 g and 0.4 deg), so the two references disagree [eval/walk]
WALK_PITCH_DEG = 1.3        # pelvis tilt, twice per stride [src: Erciyes 801 walking-window waist pitch rate
                            # 0.196 rad/s (2.0 gave 0.274); mocap twice-per-stride tilt 0.94 (0.55-1.55) per
                            # performer (eval/walk/harmonics_before.txt)] (was 2.0)
WALK_PITCH_STRIDE_DEG = 0.75  # pelvis tilt once per stride (left/right difference) [src: mocap 0.77 (0.53-0.89)
                            # per performer]; with it the tilt range stays near mocap p-p 3.9 (3.1-4.9)
WALK_LATERAL_M = 0.02       # pelvis side sway [src: motionval: mocap p-p 3.7 cm (1.9-5.1); Orendurff 2004 3.9-7 cm COM]
WALK_COUNTER_TWIST_DEG = 7.5  # thorax counter-rotation against the pelvis [src: motionval: mocap thorax-pelvis
                            # p-p 15.6 (11.9-19.5)] (was 2)
WALK_COUNTER_TWIST_LEAD_DEG = 31.0  # the counter-rotation runs this far ahead of exact anti-phase, so the thorax
                            # still turns about 0.65x the pelvis, ~95 deg behind it [src: mocap twist phase 31 deg
                            # ahead of anti-phase, R 0.8 (eval/walk/harmonics_before.txt); Erciyes 801 chest/waist
                            # yaw-rate ratio 0.69] (was 0: the thorax yaw nearly cancelled)
WALK_THORAX_ROLL_SHARE = 0.5  # the thorax rolls half as much as the pelvis ...
WALK_THORAX_ROLL_LAG_DEG = 90.0  # ... and a quarter stride later: the spine bends sideways against the pelvic
                            # obliquity [src: mocap thorax/pelvis roll 0.40-0.50, lag 80-134 deg, spinal lateral
                            # bend amplitude 4.5-5.0 deg, i.e. 9-10 deg p-p (eval/walk/harmonics_before.txt "lat"
                            # h1; the sim's is 4.85 deg amplitude); Erciyes 801 chest/waist roll-rate
                            # ratio 0.60] (was 1.0 and 0: the thorax rolled rigidly with the pelvis)
# Timing within the stride: theta is the stride phase of walk(), 90 deg at the right heel strike (pelvis rotated
# furthest left); a wave is amp * sin(h * theta + phase). Phases from harmonic fits to the class A mocap walks
# (eval/walk/harmonics.py: per performer / per clip, R = how alike the clips are, 1 = identical).
WALK_PITCH_STRIDE_PHASE_DEG = -100.0  # [src: mocap -110 / -91 (R 0.5)]
# Fore-aft (surge) and extra vertical pelvis oscillations at 1.3 m/s, scaled with speed like the bob: (harmonic of the
# stride frequency, amplitude m, phase deg). The surge: the pelvis slows over the stance foot and speeds up in
# double support (twice per stride, about 100 deg ahead of the bob: forward speed lowest at the top of the bob);
# the twice-per-stride pelvic tilt runs opposite to it, so the trunk tilts back while the pelvis surges forward and
# the chest moves less fore-aft than the pelvis. The 4th and 6th harmonics give the sharper loading after each heel
# strike. [src: mocap pelvis per performer (eval/walk/harmonics.py): fore-aft 0.157 / 0.101 / 0.061 g, vertical
# 0.118 / 0.066 g at harmonics 2 / 4 / 6, phases from the displacement (R 0.4-0.8) or force fits; Erciyes 801
# walking windows: waist fore-aft 0.150 g std (0.025 without the surge), chest/waist fore-aft 0.096 / 0.150 g, which
# sets the surge against the tilt (mocap puts the tilt ~90 deg from the surge, R 0.5) and the 6th vertical harmonic
# below the mocap value (3-8 Hz waist energy)] (was none: constant forward speed and a sinusoidal bob)
WALK_SURGE_MOCAP = ((2, 0.013, -151.4), (4, 0.002, 30.0), (6, 0.0005, -150.0))
# The surge is OFF for now: the viewer's leg rig (viz/rig.py, not this file) cannot follow a pelvis whose speed
# changes within the stride. Its heel-strike finder divides the yaw by the pelvis speed, so every footstep lands
# ~55 ms late, and three viz tests on fixed seeds fail (eval/JOINT_TUNING.md). Turn it on with
# WALK_SURGE = WALK_SURGE_MOCAP once the rig is fixed; the realism gain is measured in eval/combined_after_surge.
WALK_SURGE: tuple = ()
WALK_BOB_EXTRA = ((4, 0.0022, -134.0), (6, 0.0004, -155.0))
WALK_STEP_VARIATION = 0.2   # step-to-step spread of the bob, surge and tilt amplitudes (SD, fraction) [design: sets
                            # the Erciyes 801 walking-window periodicity, real 0.74 waist / 0.80 chest, sim 0.82 / 0.82
                            # with identical steps]
CARRY_LEAN_DEG = 1.0        # extra trunk lean with a load > 10 kg [design for these loads; the only reference,
                            # motionval ACCAD walk-with-box minus walk of the same performer, +0 to +1 deg, is
                            # light boxes, not > 10 kg] (was 6)
# Running (2 short CMU clips only, 1.2-1.3 s each, so these are rough).
RUN_YAW_DEG = 5.0           # [src: motionval: CMU 09_01/16_35 pelvis rotation p-p 9.3-11] (was 0)
RUN_ROLL_DEG = 5.5          # [src: motionval: CMU p-p 10.3-12.5] (was 4)
RUN_PITCH_DEG = 2.5         # twice per stride [src: motionval: CMU p-p 5.0-6.5] (was 0)
RUN_LATERAL_M = 0.01        # [src: motionval: CMU p-p 2.0-2.2 cm] (was 0)
RUN_COUNTER_TWIST_DEG = 12.0  # [src: motionval: CMU thorax-pelvis p-p 24.5-28.3] (was 0)
# The thorax turns about twice as far as the pelvis and out of phase with it: CMU thorax yaw p-p 21.6-33.8 vs
# pelvis 9.1-11.3 deg, 45-84 deg apart; Erciyes 803 chest/waist yaw-rate ratio 2.9 (2.1-4.7), chest |w| 95th
# percentile 1.89 rad/s, chest yaw-rate std 0.94 rad/s (eval/diag/run_kin.py, jog_coupling.py). A 42 deg lead
# on the counter-twist gives thorax yaw p-p 21, 113 deg from the pelvis, ratio 1.8, chest yaw-rate std 0.96,
# chest |w| p95 1.56 [src: those, fitted]. 55 deg fits the p95 (1.78) better but trips baseline-rules' near-fall
# rotation threshold (1.5 rad/s) in every A-RUN run, so it is held at 42 [design].
RUN_COUNTER_TWIST_LEAD_DEG = 42.0
# Lean: the pelvis tilts a little and the spine leans the trunk [src: Erciyes 803 windows that jog throughout:
# waist inclination 2.9, chest 10.7 deg (eval/diag/steady_windows.md)]; the chest accelerometer reads about
# 2 deg less than the geometric lean while the thorax counter-pitches. (Was RUN_LEAN_DEG = 8, all in the pelvis.)
RUN_PELVIS_LEAN_DEG = 2.5
RUN_SPINE_LEAN_DEG = 8.5
RUN_SPINE_LEAN_SPLIT = np.array([0.4, 0.35, 0.25])  # over the three spinal joints, as walk() [design]
# The thorax does not ride rigidly on the pelvis (that swung the sternum fore-aft 0.53 g vs Erciyes 0.16 g):
# it pitches against the pelvis [src: CMU thorax pitch p-p 4.7-5.3 vs pelvis 3.9-4.7, 131-175 deg apart;
# Erciyes chest/waist pitch-rate ratio 0.90 (0.18-1.49); gain 0.8 keeps the Erciyes chest pitch-rate std in
# its noise] and rolls less and later [src: CMU thorax/pelvis roll p-p ratio 0.59-0.61, 83-149 deg apart;
# a 60 deg lag reads 82 deg with the counter-twist on].
RUN_THORAX_PITCH_GAIN = 0.8       # thorax world pitch oscillation / the pelvis's
RUN_THORAX_PITCH_LAG_DEG = 180.0  # its phase lag behind the pelvis's
RUN_THORAX_ROLL_GAIN = 0.6
RUN_THORAX_ROLL_LAG_DEG = 60.0
RUN_COUNTER_SPLIT = np.array([0.7, 0.3, 0.0])  # the lumbar joints make that counter-motion [design]
# Fore-aft braking then propulsion in each stance (peak, g) [src: Erciyes 803 waist fore-aft std 0.174 g in the
# windows that jog throughout (sim 0.165 with it); CMU pelvis 0.33-0.38 g at 2.8-3.6 m/s is the upper end]
# (was 0: constant speed).
RUN_BRAKE_G = 0.2
# Ladder: the pelvis shifts and rolls to alternate sides on successive rungs.
LADDER_ROLL_DEG = 5.5       # [src: motionval: CMU ladder p-p 11.5 (7.5-16.4) over two rungs] (was 3, same side)
LADDER_LATERAL_M = 0.06     # [src: motionval: CMU ladder lateral p-p 14.5 cm (11-15.8)] (was 0.015, same side)
LADDER_FLEX_DEG = (2.0, 3.0, 4.0)  # [src: motionval: CMU upper-trunk inclination 16.9 (16-17.5)] (was 2, 2, 3)
# Kneeling floor work: the trunk keeps moving between near upright and the working depth.
KNEEL_DEPTH_DEG = (30.0, 55.0)  # working trunk inclination [src: motionval: NIOSH roofing kneeling, mean
                                # 39 (33-47)] (was 45-90 [design])
KNEEL_MOVE_SD_DEG = 18.0        # trunk excursions around it, 0.08-0.5 Hz [src: motionval: NIOSH p10-p90 2-63 deg,
                                # 95th-percentile rate 61 (46-82) deg/s] (was static)
# Quiet standing.
SWAY_POS_M = 0.006          # pelvis sway per horizontal axis, < 0.4 Hz [src: motionval: mocap idle 7.7 mm RMS
                            # (6.2-8.8)] (was 0.004)

# --- lying down (Builder.lie) -----------------------------------------------------------------
# "Erciyes 815" below is the realism scorecard's lie_down category: 93 Erciyes trials of lying
# down on a bed, read with the same sensor emulation as the sim (realism/score.py). Medians (IQR):
# chest rotation peak 2.7 (2.5-3.1) rad/s, waist rotation peak 2.4 (2.0-2.8) rad/s, a back-on-bed
# contact peak of 1.5 (1.4-1.8) g at the waist, the chest still turning at 1.4 (1.0-1.9) rad/s
# 0.2-0.8 s after it, and lying still within 0.7 (0.4-1.0) s. The bed lie-down also swings the legs
# up and turns, which a lie-down on the ground does not, so the sim stays below those peaks.
LIE_BACK_S = (1.0, 1.5)     # sitting -> pelvis on the ground [src: Erciyes 815 peaks above for the sim's
                            # 80 deg pelvis turn; scorecard dev seeds] (was the caller's 2-4 s min-jerk)
LIE_TOUCH_RATE = 0.4        # trunk rate when the back touches, x the mean lowering rate [design; the
                            # Erciyes 815 contact peak needs a back that is still moving]
LIE_CONTACT_S = 0.05        # time constant of the back stopping on the surface [design]
LIE_DROP_S = 0.35           # the pelvis rolls back onto the ground in the last part of the lowering [design]
LIE_TOUCH_SPEED = 0.3       # lower-back speed at touch (m/s) [fitted to the Erciyes 815 waist contact peak;
                            # weak: a bed, while the sim lies down on the ground, where people probably land
                            # softer; there is no ground lie-down reference, and the sim waist contact peak is
                            # already below the bed's (1.32 vs 1.52 g)]
LIE_UNROLL_S = 0.9          # the curled upper trunk unrolls onto the ground after the pelvis [src: Erciyes
                            # 815 chest turning after the waist contact, above]
LIE_UNROLL_LEAD_S = 0.3     # it starts this long before the pelvis touches [design]
LIE_SIT_PAUSE_S = (0.5, 1.5)  # sitting a moment before lying back [design; Erciyes 815 waist is quiet
                              # 1.5-3 s before the contact: pre_std_w 0.034 g]


# --- sitting down hard (Builder.sit(hard=True)) ------------------------------------------------
# "Erciyes 811" below is the realism scorecard's sit_hard category, chair trials only (93 trials of
# sitting down on a chair, same emulation). Medians (IQR): chest rotation peak 1.43 (1.20-1.64) rad/s
# about 0.7 s before the seat contact, 1.48 (1.16-1.81) rad of chest rotation in all, and the chest
# still turning back for about 1 s after the contact.
SIT_LEAN_DEG = 35.0         # forward trunk lean while lowering onto the seat [src: Erciyes 811 chest rotation
                            # peak and total above, for the sim's 0.8 s lowering] (was 5)
SIT_PELVIS_SHARE = 0.6      # share of the lean at the pelvis [design; as kneel()]
SIT_SEATED_PITCH_DEG = -8.0  # pelvis tilted back once seated [src: Erciyes 811 waist inclination change
                             # 6.9 (3.0-13.4) deg] (was -5)
SIT_SEATED_FLEX_DEG = (8.0, 5.0, 3.0)  # spine flexion once seated, with the pelvis tilted back [src: Erciyes 811
                                      # chest inclination change 8.0 (4.4-14.9) deg] (was 4, 3, 2)
SIT_SETTLE_S = 1.0          # trunk back up after the seat contact [src: Erciyes 811 chest turning back for
                            # about 1 s; peak 0.2-0.8 s after the contact 1.2 (0.9-1.6) rad/s] (was 0.5)


def shank_tilt(kappa):
    """Forward shank tilt (rad) for knee flexion kappa: kappa / 2 at first, saturating smoothly."""
    return SHANK_TILT_MAX * np.tanh(kappa / (2.0 * SHANK_TILT_MAX))


def window_ramp(u: np.ndarray, window: tuple, arrive: float = 0.0, depart: float = 0.0) -> np.ndarray:
    """Minimum-jerk 0 -> 1 inside window = (start, end), as fractions u of a move; flat outside.

    arrive > 0 ends the ramp still decelerating (second derivative -arrive, in units of the change
    per window^2) instead of at zero acceleration; depart > 0 starts it with that acceleration. A
    move that arrives this way and a move back that departs this way reverse quickly, with a
    rounded rather than a flat turning point. Both stay monotonic up to 20."""
    a, b = window
    x = np.clip((u - a) / max(b - a, 1e-6), 0.0, 1.0)
    ramp = 10 * x**3 - 15 * x**4 + 6 * x**5
    ramp = ramp - 0.5 * arrive * x**3 * (1 - x) ** 2
    return ramp + 0.5 * depart * x**2 * (1 - x) ** 3


def lift_hip(kappa, height: float):
    """Hip joint relative to the ground point under the ankle (forward, up), for knee flexion kappa
    (rad): the shank tilts forward by about half of it, smoothly saturating at SHANK_TILT_MAX (a
    hard cap would put a kink, i.e. an acceleration spike, into the hip path); the thigh takes the
    rest."""
    a = shank_tilt(kappa)
    b = kappa - a
    forward = -(THIGH_H * np.sin(b) - SHANK_H * np.sin(a)) * height
    up = (ANKLE_H + SHANK_H * np.cos(a) + THIGH_H * np.cos(b)) * height
    return forward, up


def lift_root(kappa, pitch, height: float):
    """Spine root (Builder pos) relative to the ground point under the ankle (forward, up)."""
    hip_f, hip_y = lift_hip(kappa, height)
    off_f = -HIP_BELOW_ROOT_H * height * np.sin(pitch) + HIP_ANTERIOR_H * height * np.cos(pitch)
    off_y = -HIP_BELOW_ROOT_H * height * np.cos(pitch) - HIP_ANTERIOR_H * height * np.sin(pitch)
    return hip_f - off_f, hip_y - off_y


def knee_at_stand(height_fraction: float = 0.53) -> float:
    """Knee flexion (rad) that puts the spine root at height_fraction x H standing upright (the
    standing legs are not quite straight with these proportions)."""
    lo, hi = 0.0, np.pi / 2
    for _ in range(40):
        mid = (lo + hi) / 2
        if lift_root(mid, 0.0, 1.0)[1] > height_fraction:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def knee_class(knee_deg: float) -> str:
    """Lifting technique named by the knee flexion reached (deg): stoop, semi_squat or squat."""
    if knee_deg < KNEE_CLASS_DEG[0]:
        return "stoop"
    if knee_deg < KNEE_CLASS_DEG[1]:
        return "semi_squat"
    return "squat"


def minjerk(k: int) -> np.ndarray:
    tau = np.linspace(0.0, 1.0, max(k, 2))[:k]
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def forward_vec(heading: float) -> np.ndarray:
    return np.array([-np.sin(heading), 0.0, -np.cos(heading)])


def right_vec(heading: float) -> np.ndarray:
    return np.array([np.cos(heading), 0.0, -np.sin(heading)])


def smooth_pulse(fs: float, f_hz: float, sigma_s: float) -> np.ndarray:
    """Displacement shape (Gaussian-windowed sine) whose 2nd derivative peaks at 1 m/s^2 per m.

    It starts and ends with zero displacement AND zero velocity, so adding it never creates a
    velocity step (which finite differences would turn into a fake acceleration spike).
    """
    centre = 3 * sigma_s
    tau = np.arange(int(2 * centre * fs) + 1) / fs - centre
    shape = np.exp(-(tau / sigma_s) ** 2) * np.sin(2 * np.pi * f_hz * tau)
    accel = np.gradient(np.gradient(shape, 1 / fs), 1 / fs)
    return shape / np.max(np.abs(accel))


def heel_pulse(fs: float, sigma_s: float) -> np.ndarray:
    """Displacement dip -exp(-tau^2/sigma^2): its 2nd derivative is an upward spike (peak 1)
    with small negative side lobes (~0.45), like a trunk heel-strike transient. Zero velocity
    at both ends."""
    tau = np.arange(int(6 * sigma_s * fs) + 1) / fs - 3 * sigma_s
    shape = -np.exp(-(tau / sigma_s) ** 2)
    accel = np.gradient(np.gradient(shape, 1 / fs), 1 / fs)
    return shape / np.max(accel)


def stride_waves(theta: np.ndarray, waves: tuple) -> np.ndarray:
    """Sum of amp * sin(h * theta + phase) over (h, amp, phase_deg) for a stride phase theta (rad)."""
    out = np.zeros_like(theta)
    for harmonic, amp, phase_deg in waves:
        out += amp * np.sin(harmonic * theta + np.deg2rad(phase_deg))
    return out


def taper(k: int, fs: float, seconds: float = 0.2) -> np.ndarray:
    """0 -> 1 -> 0 minimum-jerk window, used to fade overlays in and out without steps."""
    w = np.ones(k)
    r = min(int(seconds * fs), k // 2)
    if r >= 2:
        w[:r] = minjerk(r)
        w[k - r:] = minjerk(r)[::-1]
    return w


def lowering(k: int, end_rate: float) -> np.ndarray:
    """0 -> 1 over k samples that starts at rest and ends still moving, at end_rate (in units of
    the mean rate) with no acceleration: a lowering that reaches the ground (quintic; end_rate 0
    gives minjerk). Monotonic for end_rate in [0, 1.6]."""
    tau = np.linspace(0.0, 1.0, max(k, 2))[:k]
    a3, a4, a5 = 10 - 4 * end_rate, -15 + 7 * end_rate, 6 - 3 * end_rate
    return a3 * tau**3 + a4 * tau**4 + a5 * tau**5


def band_noise(rng: np.random.Generator, k: int, fs: float, lo: float | None, hi: float) -> np.ndarray:
    """Unit-RMS noise limited to [lo, hi] Hz (low-pass when lo is None)."""
    # 4th-order, zero-phase (effective 8th order): steep enough that the noise stays smooth
    # after double differentiation. A 2nd-order roll-off leaves white-noise *acceleration*.
    x = rng.normal(size=k + 4000)
    if lo is None:
        sos = signal.butter(4, hi, fs=fs, output="sos")
    else:
        sos = signal.butter(4, [lo, min(hi, fs / 2 - 1)], btype="band", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)[2000:2000 + k]
    return y / max(np.std(y), 1e-12)


@dataclass
class Pose:
    pos: np.ndarray
    heading: float
    pitch: float
    roll: float
    flex: np.ndarray
    lat: np.ndarray
    twist: np.ndarray
    worn: bool = True

    def copy(self) -> "Pose":
        return Pose(self.pos.copy(), self.heading, self.pitch, self.roll, self.flex.copy(), self.lat.copy(), self.twist.copy(), self.worn)


class Seg:
    """Arrays for one segment, initialised to a constant pose."""

    def __init__(self, pose: Pose, k: int, phase: int, breath_amp: float, breath_hz: float, fidget: float, sway: float):
        self.k = k
        self.pos = np.tile(pose.pos, (k, 1))
        self.heading = np.full(k, pose.heading)
        self.pitch = np.full(k, pose.pitch)
        self.roll = np.full(k, pose.roll)
        self.flex = np.tile(pose.flex, (k, 1))
        self.lat = np.tile(pose.lat, (k, 1))
        self.twist = np.tile(pose.twist, (k, 1))
        self.phase = np.full(k, phase, np.int16)
        self.worn = np.full(k, pose.worn)
        on = 1.0 if pose.worn else 0.0
        self.breath_amp = np.full(k, breath_amp * on)
        self.breath_hz = np.full(k, breath_hz)
        self.fidget = np.full(k, fidget * on)
        self.sway = np.full(k, sway * on)


class Builder:
    """Accumulates segments at fs = 1 kHz and records ground-truth events."""

    def __init__(self, worker: Worker, site: Site, rng: np.random.Generator, fs: float = 1000.0):
        self.worker, self.site, self.rng, self.fs = worker, site, rng, fs
        self.dt = 1.0 / fs
        self.segs: list[Seg] = []
        self.events: list[tuple[float, str, dict]] = []
        self.mount_steps: list[tuple[float, int, np.ndarray]] = []
        self.work_modes: list[tuple[float, str]] = [(0.0, "normal")]
        self.n = 0
        h = worker.height_m
        self.pose = Pose(np.array([0.0, worker.pelvis_height, 0.0]), float(rng.uniform(-np.pi, np.pi)), 0.0, 0.0,
                         np.zeros(3), np.zeros(3), np.zeros(3))
        self.vel = np.zeros(3)
        self.ground_y = 0.0  # current support-surface height under the worker
        self.breath_hz = float(rng.uniform(0.2, 0.33))   # SIM_SPEC.md 1.3 breathing range
        self.breath_amp = np.deg2rad(rng.uniform(0.4, 0.8))  # ~ +/-0.01 g gravity projection
        self.conscious = True
        self.height = h
        self._lift_rng: np.random.Generator | None = None  # lift details, see lift_rng()

    # --- bookkeeping ---------------------------------------------------------------------
    @property
    def t(self) -> float:
        return self.n / self.fs

    def event(self, name: str, **info) -> None:
        self.events.append((self.t, name, info))

    def set_work_mode(self, mode: str) -> None:
        self.work_modes.append((self.t, mode))

    def seg(self, duration: float, phase: str = "activity", fidget: float | None = None, sway: float = 0.0) -> Seg:
        k = max(int(round(duration * self.fs)), 2)
        if fidget is None:
            fidget = 1.0 if self.conscious else 0.0
        return Seg(self.pose, k, PHASE_CODE[phase], self.breath_amp, self.breath_hz, fidget, sway)

    CHANNELS = ("pos", "heading", "pitch", "roll", "flex", "lat", "twist")

    def _match_join(self, s: Seg) -> None:
        """Make every channel C1-continuous with the previous segment.

        The new segment is shifted so its first sample continues the previous one, and any
        rate mismatch dv is removed with c(t) = dv * tau * (1 - exp(-t / tau)), tau = 50 ms:
        a brief, finite acceleration instead of a velocity step (which differentiation would
        turn into a fake impact). Intentional rapid changes inside a segment are untouched.
        """
        if not self.segs or s.k < 3:
            return
        prev = self.segs[-1]
        tau = 0.05
        t = (np.arange(s.k) + 1) * self.dt
        corr = tau * (1 - np.exp(-t / tau))
        for name in self.CHANNELS:
            cur = getattr(s, name)
            old = getattr(prev, name)
            prev_rate = (old[-1] - old[-2]) * self.fs if prev.k >= 2 else 0.0 * old[-1]
            first = old[-1] + prev_rate * self.dt
            new_rate = (cur[1] - cur[0]) * self.fs
            shifted = cur + (first - cur[0])
            dv = prev_rate - new_rate
            if cur.ndim == 1:
                shifted = shifted + dv * corr
            else:
                shifted = shifted + corr[:, None] * dv[None, :]
            setattr(s, name, shifted)

    def commit(self, s: Seg) -> None:
        self._match_join(s)
        self.segs.append(s)
        self.n += s.k
        last = s.k - 1
        if s.k >= 2:
            self.vel = (s.pos[last] - s.pos[last - 1]) * self.fs
        self.pose = Pose(s.pos[last].copy(), float(s.heading[last]), float(s.pitch[last]), float(s.roll[last]),
                         s.flex[last].copy(), s.lat[last].copy(), s.twist[last].copy(), bool(s.worn[last]))

    # --- generic voluntary transitions ---------------------------------------------------
    def move(self, duration: float, phase: str = "activity", voluntary: bool = True, **target) -> Seg:
        """Minimum-jerk move of any of: pos, y, heading, pitch, roll, flex, lat, twist."""
        p = self.pose
        goal = {"pos": p.pos.copy(), "heading": p.heading, "pitch": p.pitch, "roll": p.roll,
                "flex": p.flex.copy(), "lat": p.lat.copy(), "twist": p.twist.copy()}
        if "y" in target:
            goal["pos"][1] = target.pop("y")
        if "forward" in target:
            goal["pos"] = goal["pos"] + forward_vec(p.heading) * target.pop("forward")
        for key, value in target.items():
            goal[key] = np.array(value, float) if key in ("pos", "flex", "lat", "twist") else float(value)
        if voluntary:
            change = max(abs(goal["pitch"] - p.pitch), abs(goal["roll"] - p.roll),
                         float(np.max(np.abs(goal["flex"] - p.flex))), float(np.max(np.abs(goal["lat"] - p.lat))))
            duration = max(duration, change / MAX_VOLUNTARY_RATE * 1.9)  # min-jerk peak rate = 1.875x mean
        s = self.seg(duration, phase)
        w = minjerk(s.k)
        s.pos = p.pos + (goal["pos"] - p.pos) * w[:, None]
        s.heading = p.heading + (goal["heading"] - p.heading) * w
        s.pitch = p.pitch + (goal["pitch"] - p.pitch) * w
        s.roll = p.roll + (goal["roll"] - p.roll) * w
        s.flex = p.flex + (goal["flex"] - p.flex) * w[:, None]
        s.lat = p.lat + (goal["lat"] - p.lat) * w[:, None]
        s.twist = p.twist + (goal["twist"] - p.twist) * w[:, None]
        self.commit(s)
        return s

    def hold(self, duration: float, phase: str = "activity", sway: float = 1.0, fidget: float | None = None) -> Seg:
        s = self.seg(duration, phase, fidget=fidget, sway=sway)
        self.commit(s)
        return s

    # --- postures (pelvis-sensor heights are [design] fractions of body height) ----------
    def stand_up(self, duration: float = 1.2, phase: str = "activity") -> None:
        self.move(duration, phase, y=self.ground_y + self.worker.pelvis_height, pitch=0.0, roll=0.0,
                  flex=np.zeros(3), lat=np.zeros(3), twist=np.zeros(3))

    def stand(self, duration: float, phase: str = "activity") -> None:
        self.hold(duration, phase, sway=1.0)

    def stoop(self, depth_deg: float, down_s: float = 1.2, hold_s: float = 1.0, up_s: float = 1.2, phase: str = "activity") -> None:
        """Trunk flexion with lumbo-pelvic rhythm (~42 % pelvis, Paloschi 2021 [src via spine note])."""
        hip = np.deg2rad(depth_deg) * 0.45
        spine = np.deg2rad(depth_deg) - hip
        self.move(down_s, phase, y=self.ground_y + self.worker.pelvis_height - 0.03, pitch=hip,
                  flex=spine * np.array([0.45, 0.35, 0.20]))
        self.hold(hold_s, phase, sway=0.5)
        self.stand_up(up_s, phase)

    def squat(self, hold_s: float, phase: str = "activity") -> None:
        """Squat down, hold, and stand up (amplitudes and timing: SQUAT_* constants)."""
        self.move(SQUAT_DOWN_S, phase, y=self.ground_y + 0.30 * self.height, pitch=np.deg2rad(SQUAT_PELVIS_DEG),
                  flex=np.deg2rad(SQUAT_FLEX_DEG))
        self.hold(hold_s, phase, sway=0.4)
        self.stand_up(SQUAT_UP_S, phase)

    def kneel(self, trunk_deg: float = 10.0, duration: float = 1.8, phase: str = "activity") -> None:
        hip = np.deg2rad(trunk_deg) * 0.6
        self.move(duration, phase, y=self.ground_y + 0.30 * self.height, pitch=hip,
                  flex=(np.deg2rad(trunk_deg) - hip) * np.array([0.45, 0.35, 0.20]))

    def quadruped(self, duration: float = 1.5, phase: str = "activity") -> None:
        self.move(duration, phase, y=self.ground_y + 0.28 * self.height, pitch=np.deg2rad(82), flex=np.deg2rad([2, 2, 4]))

    def sit_ground(self, duration: float = 1.8, phase: str = "activity") -> None:
        self.move(duration, phase, y=self.ground_y + 0.10, pitch=np.deg2rad(-8), flex=np.deg2rad([10, 8, 6]))

    def lie(self, posture: str, duration: float = 2.0, phase: str = "activity") -> None:
        """Controlled lie-down from sitting/kneeling into supine, prone or side posture.
        Supine: duration is the time from sitting to lying settled (lowering, then lying still)."""
        if posture == "supine":
            if self.pose.pos[1] > self.ground_y + 0.25:
                self.kneel(10, 1.5, phase)
                self.sit_ground(1.5, phase)
                self.hold(float(self.keyed_rng(0x5354).uniform(*LIE_SIT_PAUSE_S)), phase, sway=0.3)
            lie_s = self.lie_back(phase)
            self.hold(max(duration - lie_s, 0.5), phase, sway=0.0)
        elif posture == "prone":
            if self.pose.pos[1] > self.ground_y + 0.45 * self.height:
                self.kneel(10, 1.5, phase)
            self.quadruped(1.2, phase)
            self.move(duration, phase, y=self.ground_y + 0.20, pitch=np.deg2rad(88), flex=np.zeros(3))
        else:  # side
            if self.pose.pos[1] > self.ground_y + 0.25:
                self.kneel(10, 1.5, phase)
                self.sit_ground(1.5, phase)
            sign = 1.0 if posture == "right" else -1.0
            self.move(duration, phase, y=self.ground_y + 0.17, pitch=0.0, roll=sign * np.deg2rad(85), flex=np.deg2rad([8, 8, 10]))

    def touchdown(self, k: int, end_rate: float) -> np.ndarray:
        """0 -> 1: a lowering over k samples that touches the surface still moving (end_rate x its
        mean rate), then stops on it with time constant LIE_CONTACT_S (a short contact tail)."""
        overshoot = end_rate * LIE_CONTACT_S / (k * self.dt)   # the share of the move taken by the contact
        t = (np.arange(int(round(5 * LIE_CONTACT_S * self.fs))) + 1) * self.dt
        contact = 1.0 + overshoot * (1.0 - np.exp(-t / LIE_CONTACT_S))
        return np.concatenate([lowering(k, end_rate), contact]) / (1.0 + overshoot)

    def lie_back(self, phase: str = "activity") -> float:
        """From sitting on the ground onto the back, rolling down: the pelvis rolls back and the lower
        back drops onto the surface first, then the curled upper trunk unrolls and lands. Each part
        touches still moving and stops on the surface. Returns the time taken (s)."""
        p = self.pose
        drop_m = p.pos[1] - (self.ground_y + 0.03)
        goal_pitch = np.deg2rad(-88)
        # The lowering's peak rate is about 1.7x its mean; keep it under the voluntary trunk-rate cap.
        change = max(abs(goal_pitch - p.pitch), abs(p.roll))
        lower_s = max(float(self.keyed_rng(0x4C49).uniform(*LIE_BACK_S)), 1.7 * change / MAX_VOLUNTARY_RATE)
        k_low = int(round(lower_s * self.fs))
        turn = self.touchdown(k_low, LIE_TOUCH_RATE)
        # The pelvis drops in the last LIE_DROP_S, landing at LIE_TOUCH_SPEED.
        k_drop = min(int(round(LIE_DROP_S * self.fs)), k_low)
        drop_rate = min(LIE_TOUCH_SPEED * LIE_DROP_S / max(drop_m, 1e-3), 1.6)
        down = np.concatenate([np.zeros(k_low - k_drop), self.touchdown(k_drop, drop_rate)])
        # The spine keeps its sitting curl and unrolls from LIE_UNROLL_LEAD_S before the pelvis touches.
        k_start = max(k_low - int(round(LIE_UNROLL_LEAD_S * self.fs)), 0)
        unroll = np.concatenate([np.zeros(k_start), self.touchdown(int(round(LIE_UNROLL_S * self.fs)), LIE_TOUCH_RATE)])
        k = max(len(turn), len(unroll))
        turn, down, unroll = (np.concatenate([x, np.ones(k - len(x))]) for x in (turn, down, unroll))
        s = self.seg(k * self.dt, phase)
        s.pos[:, 1] = p.pos[1] - drop_m * down
        s.pitch = p.pitch + (goal_pitch - p.pitch) * turn
        s.roll = p.roll * (1.0 - turn)
        s.flex = p.flex * (1.0 - unroll)[:, None]
        self.commit(s)
        return k * self.dt

    def get_up(self, phase: str = "recovery") -> None:
        """From any lying posture: sit or kneel, then stand."""
        if self.pose.pitch > np.deg2rad(45):  # prone-ish: push up to hands and knees
            self.quadruped(1.5, phase)
            self.kneel(10, 1.2, phase)
        else:
            self.sit_ground(1.8, phase)
            self.kneel(10, 1.2, phase)
        self.stand_up(1.3, phase)

    # --- locomotion ----------------------------------------------------------------------
    def walk(self, duration: float, speed: float | None = None, phase: str = "activity", load_kg: float = 0.0,
             turn_rate: float = 0.0, stop: bool = True, lean_deg: float = 3.0) -> None:
        surface = SURFACES[self.site.surface]
        if speed is None:
            speed = float(self.rng.uniform(1.1, 1.45))
        speed *= 0.75 if surface.name in ("mud", "gravel", "rebar_mat") else 1.0
        speed *= 0.85 if load_kg > 10 else 1.0
        s = self.seg(duration, phase)
        k, t = s.k, np.arange(s.k) * self.dt
        ramp = np.ones(k)
        r = min(int(0.6 * self.fs), k // 2)
        if self.vel[[0, 2]].dot(self.vel[[0, 2]]) < 0.04:  # starting from rest
            ramp[:r] = minjerk(r)
        if stop:
            ramp[k - r:] = minjerk(r)[::-1]
        f_step = float(np.clip(1.8 * np.sqrt(speed / 1.3) * self.rng.uniform(0.95, 1.05), 1.4, 2.3))
        jitter = np.cumsum(band_noise(self.rng, k, self.fs, None, 0.5)) * self.dt * 0.15  # step timing variability
        theta = np.pi * f_step * t + jitter + self.rng.uniform(0, 2 * np.pi)  # stride phase
        path_heading = self.pose.heading + turn_rate * t
        heading = path_heading + np.deg2rad(WALK_YAW_DEG) * np.sin(theta) * ramp  # pelvis rotation within the stride
        fwd = np.stack([-np.sin(path_heading), np.zeros(k), -np.cos(path_heading)], axis=1)
        right = np.stack([np.cos(path_heading), np.zeros(k), -np.sin(path_heading)], axis=1)
        s.pos = self.pose.pos + np.cumsum(fwd * (speed * ramp)[:, None], axis=0) * self.dt
        # No two steps are alike: the bob, surge and tilt amplitudes wander from step to step. The keyed generator
        # leaves the main random stream (and so every other draw of a run) unchanged.
        vary = 1.0 + WALK_STEP_VARIATION * band_noise(self.keyed_rng(0x5741), k, self.fs, None, f_step)
        gait_scale = (speed / 1.3) * ramp * vary
        bob = (WALK_BOB_M * (np.cos(2 * theta) - 1) / 2 + stride_waves(theta, WALK_BOB_EXTRA)) * gait_scale
        surge = stride_waves(theta, WALK_SURGE) * gait_scale   # fore-aft, along the path
        lateral = WALK_LATERAL_M * ramp * np.sin(theta)
        s.pos[:, 1] += bob
        s.pos += lateral[:, None] * right + surge[:, None] * fwd
        # Uneven ground: random step-height offsets, smoothed within each step.
        if surface.irregularity_m > 0:
            steps = np.floor((theta - theta[0]) / np.pi).astype(int)
            offsets = self.rng.normal(0, surface.irregularity_m, steps.max() + 2)
            raw = offsets[steps]
            s.pos[:, 1] += signal.sosfiltfilt(signal.butter(2, 3.0, fs=self.fs, output="sos"), raw) * ramp
        # Heel-strike transients: damped oscillation at each foot contact, amplitude by surface.
        if surface.heel_strike_g > 0:
            contacts = np.where(np.diff(np.floor((2 * theta - np.pi) / (2 * np.pi))) > 0)[0]
            pulse = surface.heel_strike_g * G * heel_pulse(self.fs, 0.25 / surface.heel_strike_hz)
            for c in contacts:
                end = min(c + len(pulse), k)
                s.pos[c:end, 1] += pulse[: end - c] * ramp[c] * self.rng.uniform(0.7, 1.3)
        lean = np.deg2rad(lean_deg + (CARRY_LEAN_DEG if load_kg > 10 else 0.0))
        s.heading = heading
        tilt_twice = np.deg2rad(WALK_PITCH_DEG) * np.sin(2 * theta + 0.5) * vary   # opposite to WALK_SURGE_MOCAP
        tilt_once = np.deg2rad(WALK_PITCH_STRIDE_DEG) * np.sin(theta + np.deg2rad(WALK_PITCH_STRIDE_PHASE_DEG))
        s.pitch = self.pose.pitch + (lean * 0.5 + tilt_twice + tilt_once) * ramp
        pelvis_roll = np.deg2rad(WALK_ROLL_DEG) * np.sin(theta)
        s.roll = self.pose.roll + pelvis_roll * ramp
        s.flex = self.pose.flex + (lean * 0.5 * np.array([0.4, 0.35, 0.25]))[None] * ramp[:, None]
        # The thorax counter-rotates against the pelvis, slightly ahead of anti-phase, and bends sideways
        # against the pelvic obliquity (spinal joint sums: thorax angle minus pelvis angle).
        twist = -np.deg2rad(WALK_COUNTER_TWIST_DEG) * np.sin(theta + np.deg2rad(WALK_COUNTER_TWIST_LEAD_DEG))
        s.twist = self.pose.twist + np.outer(twist * ramp, [0.2, 0.4, 0.4])
        thorax_roll = (WALK_THORAX_ROLL_SHARE * np.deg2rad(WALK_ROLL_DEG)
                       * np.sin(theta - np.deg2rad(WALK_THORAX_ROLL_LAG_DEG)))
        s.lat = self.pose.lat + np.outer((thorax_roll - pelvis_roll) * ramp, [0.4, 0.35, 0.25])
        s.breath_hz[:] = 0.4
        self.commit(s)
        if stop:
            self.vel = np.zeros(3)

    def run(self, duration: float, speed: float = 3.0, phase: str = "activity") -> None:
        """Running: stance pulses (~2-4 g) separated by flight phases (~0 g) at 2.5-3 Hz, with fore-aft
        braking and propulsion in stance, and a thorax that moves against the pelvis (RUN_* above)."""
        s = self.seg(duration, phase)
        k = s.k
        cadence = float(self.rng.uniform(2.5, 3.0))
        t_step = 1.0 / cadence
        t_stance = t_step * self.rng.uniform(0.6, 0.7)
        t = np.arange(k) * self.dt
        tin = np.mod(t, t_step)
        # Specific force: half-sine during stance, zero in flight; mean equals g over a step.
        peak = G * t_step * np.pi / (2 * t_stance)
        f = np.where(tin < t_stance, peak * np.sin(np.pi * tin / t_stance), 0.0)
        ramp = taper(k, self.fs, 0.8)          # zero slope at both ends, so no angle rate leaks into the next segment
        acc_y = (f - G) * ramp
        vy = np.cumsum(acc_y) * self.dt
        vy -= np.linspace(0, vy[-1], k)
        y = np.cumsum(vy) * self.dt
        y -= np.linspace(0, y[-1], k)
        # Fore-aft: braking in the first half of stance, propulsion in the second, so each stance
        # starts and ends at the same speed; the mean is removed so the average speed stays `speed`.
        brake = np.where(tin < t_stance, -RUN_BRAKE_G * G * np.sin(2 * np.pi * tin / t_stance), 0.0)
        dv = np.cumsum(brake) * self.dt
        dv -= dv.mean()
        fwd = forward_vec(self.pose.heading)
        stride = np.pi * cadence * t            # stride phase: one cycle per two steps
        s.pos = self.pose.pos + np.outer(np.cumsum((speed + dv) * ramp) * self.dt, fwd)
        s.pos[:, 1] += y
        s.pos += np.outer(RUN_LATERAL_M * np.sin(stride) * ramp, right_vec(self.pose.heading))
        s.heading = self.pose.heading + np.deg2rad(RUN_YAW_DEG) * np.sin(stride) * ramp
        pitch_osc = np.deg2rad(RUN_PITCH_DEG) * np.sin(2 * stride) * ramp
        roll_osc = np.deg2rad(RUN_ROLL_DEG) * np.sin(stride) * ramp
        s.pitch = self.pose.pitch + np.deg2rad(RUN_PELVIS_LEAN_DEG) * ramp + pitch_osc
        s.roll = self.pose.roll + roll_osc
        # The spine leans the trunk further, and the lumbar joints make the thorax pitch and roll
        # out of phase with the pelvis instead of riding on it rigidly.
        thorax_pitch = (RUN_THORAX_PITCH_GAIN * np.deg2rad(RUN_PITCH_DEG)
                        * np.sin(2 * stride - np.deg2rad(RUN_THORAX_PITCH_LAG_DEG)) * ramp)
        thorax_roll = (RUN_THORAX_ROLL_GAIN * np.deg2rad(RUN_ROLL_DEG)
                       * np.sin(stride - np.deg2rad(RUN_THORAX_ROLL_LAG_DEG)) * ramp)
        s.flex = (self.pose.flex + np.outer(np.deg2rad(RUN_SPINE_LEAN_DEG) * ramp, RUN_SPINE_LEAN_SPLIT)
                  + np.outer(thorax_pitch - pitch_osc, RUN_COUNTER_SPLIT))
        s.lat = self.pose.lat + np.outer(thorax_roll - roll_osc, RUN_COUNTER_SPLIT)
        twist = -np.deg2rad(RUN_COUNTER_TWIST_DEG) * np.sin(stride + np.deg2rad(RUN_COUNTER_TWIST_LEAD_DEG)) * ramp
        s.twist = self.pose.twist + np.outer(twist, [0.2, 0.4, 0.4])
        s.breath_hz[:] = 0.6
        self.commit(s)
        self.vel = np.zeros(3)

    def crawl(self, duration: float, phase: str = "activity") -> None:
        s = self.seg(duration, phase)
        t = np.arange(s.k) * self.dt
        f = float(self.rng.uniform(0.8, 1.2))
        fwd = forward_vec(self.pose.heading)
        ramp = taper(s.k, self.fs, 0.8)
        s.pos = self.pose.pos + np.outer(np.cumsum(0.3 * ramp) * self.dt, fwd)
        s.pos[:, 1] += 0.01 * (np.cos(2 * np.pi * f * t) - 1) * ramp
        s.roll = self.pose.roll + np.deg2rad(4) * np.sin(np.pi * f * t) * ramp
        s.heading = self.pose.heading + np.deg2rad(3) * np.sin(np.pi * f * t) * ramp
        self.commit(s)

    def ladder(self, height: float, up: bool = True, phase: str = "activity") -> None:
        """Climb at ~0.6-1 rung/s, rungs 0.3 m apart, ladder at 1:4 (29 CFR 1926.1053(b)(5)(i))."""
        rungs = max(int(round(height / 0.3)), 1)
        rate = float(self.rng.uniform(0.6, 1.0))
        self.move(0.8, phase, pitch=np.deg2rad(8), flex=np.deg2rad(LADDER_FLEX_DEG))
        sign = 1.0 if up else -1.0
        for rung in range(rungs):
            s = self.seg(1.0 / rate, phase)
            w = minjerk(s.k)
            rise = sign * 0.3 * w
            side = 1.0 if rung % 2 == 0 else -1.0     # weight shifts to alternate sides on successive rungs
            s.pos[:, 1] += rise
            s.pos += np.outer(sign * 0.3 / 4.0 * w, forward_vec(self.pose.heading))
            s.roll = self.pose.roll + side * np.deg2rad(LADDER_ROLL_DEG) * np.sin(np.pi * w)
            s.pos += np.outer(side * LADDER_LATERAL_M * np.sin(np.pi * w), right_vec(self.pose.heading))
            if not up and rung == rungs - 1:
                # Back on the ground: straighten up during the last step down, instead of walking off
                # still leaning into the ladder (found by motionval: walks after a ladder leaned 20 deg).
                s.pitch = self.pose.pitch * (1.0 - w)
                s.flex = self.pose.flex * (1.0 - w)[:, None]
            self.commit(s)
        self.ground_y += sign * rungs * 0.3

    # --- work activities -----------------------------------------------------------------
    def impulses(self, s: Seg, times: np.ndarray, upper_g: tuple[float, float], pelvis_g: float = 0.4) -> None:
        """Tool strikes: short damped jerks of the upper joints (and a smaller pelvis jolt)."""
        shape = smooth_pulse(self.fs, 22.0, 0.012)
        spacing = self.worker.spacings()
        for c in (np.asarray(times) * self.fs).astype(int):
            if c < 0 or c >= s.k:
                continue
            end = min(c + len(shape), s.k)
            angle = self.rng.uniform(*upper_g) * G / spacing[2]  # tangential accel = L * angular accel
            s.flex[c:end, 2] += angle * shape[: end - c]
            s.flex[c:end, 1] += 0.4 * angle * shape[: end - c]
            s.pos[c:end, 1] += pelvis_g * G * self.rng.uniform(0.3, 1.0) * shape[: end - c]

    def hammer(self, duration: float, overhead: bool = False, phase: str = "activity") -> None:
        if overhead:
            self.move(1.0, phase, flex=np.deg2rad([-4, -6, -8]))
        s = self.seg(duration, phase, sway=0.5)
        rate = self.rng.uniform(2.0, 6.0)  # SIM_SPEC A-HAMMER: 2-6 Hz strikes
        base = np.arange(0.5, duration, 1.0 / rate)
        self.impulses(s, base + self.rng.normal(0, 0.02, len(base)), (1.5, 3.0), pelvis_g=0.5)
        self.commit(s)
        if overhead:
            self.stand_up(1.0, phase)

    def vibrate(self, duration: float, band: tuple[float, float], peak_g: float, slots: str = "trunk", phase: str = "activity") -> Seg:
        """Band-limited vibration (displacement from double-integrated acceleration noise)."""
        s = self.seg(duration, phase, sway=0.3)
        for axis, weight in ((1, 1.0), (0, 0.5), (2, 0.5)):
            a = band_noise(self.rng, s.k, self.fs, band[0], band[1]) * peak_g * G / 3 * weight
            spec = np.fft.rfft(a)
            freqs = np.fft.rfftfreq(s.k, self.dt)
            freqs[0] = 1.0
            disp = np.fft.irfft(-spec / (2 * np.pi * freqs) ** 2, n=s.k)
            disp = (disp - np.linspace(disp[0], disp[-1], s.k)) * taper(s.k, self.fs, 0.3)
            if slots == "upper":
                spacing = self.worker.spacings()[2]
                s.flex[:, 2] += disp / spacing
            else:
                s.pos[:, axis] += disp
        self.commit(s)
        return s

    def jackhammer(self, duration: float, phase: str = "activity") -> None:
        self.move(1.0, phase, pitch=np.deg2rad(18), flex=np.deg2rad([5, 4, 3]))
        self.vibrate(duration, (20.0, 40.0), float(self.rng.uniform(1.0, 5.0)), "trunk", phase)  # SIM_SPEC A-JACK [design]
        self.stand_up(1.0, phase)

    def drill_overhead(self, duration: float, phase: str = "activity") -> None:
        ext = self.rng.uniform(10, 30)
        self.move(1.2, phase, flex=-np.deg2rad(ext) * np.array([0.3, 0.35, 0.35]))
        self.vibrate(duration, (20.0, 50.0), float(self.rng.uniform(0.2, 1.0)), "upper", phase)
        self.stand_up(1.0, phase)

    def carry(self, duration: float, load_kg: float, phase: str = "activity") -> None:
        """Pick a load up from the floor, walk with it, put it down.

        The two pairs of main-stream draws (they used to set a stoop's depth and hold) are kept in
        their old order, so every other random number of a run is unchanged: the first draw of a
        pair now sets the hand height, the second the total time of the lift. Everything new
        (technique, durations, spinal flexion) comes from the lift sub-generator.
        """
        pick_u = (self.rng.uniform(60, 90) - 60.0) / 30.0
        pick_s = self._old_stoop_s(self.rng.uniform(1, 3))
        # Grip from 0.15 m (a sack held near its bottom corners) to 0.50 m (knee height) [design].
        self.lift(0.15 + 0.35 * (1.0 - pick_u), load_kg, total_s=pick_s)
        self.walk(duration, load_kg=load_kg, phase=phase)
        drop_u = (self.rng.uniform(60, 90) - 60.0) / 30.0
        drop_s = self._old_stoop_s(self.rng.uniform(1, 3))
        # Set down on the floor, held at 0.25-0.45 m (about mid-height of a sack) [design].
        self.put_down(0.25 + 0.20 * (1.0 - drop_u), load_kg, total_s=drop_s)

    def _old_stoop_s(self, hold_s: float) -> float:
        """Length of the stoop the carry used before lift(): 1.2 s down, the hold, 1.2 s up, counted
        in whole samples exactly as those three segments were, so a run keeps its old length."""
        return (2 * int(round(1.2 * self.fs)) + int(round(hold_s * self.fs))) / self.fs

    # --- lifting ---------------------------------------------------------------------------
    def lift_rng(self) -> np.random.Generator:
        """Sub-generator for lift details. Spawning does not advance the main stream."""
        if self._lift_rng is None:
            self._lift_rng = self.rng.spawn(1)[0]
        return self._lift_rng

    def preferred_technique(self) -> str:
        """The worker's habitual lifting technique, the same in every Builder of a shift: drawn from
        a generator keyed on the worker, never from the main stream."""
        key = [0x11F7, int(round(self.worker.height_m * 1e6)), int(round(self.worker.mass_kg * 1e6))]
        names = list(LIFT_TECHNIQUE_SHARE)
        shares = np.array([LIFT_TECHNIQUE_SHARE[n] for n in names])
        return names[int(np.random.default_rng(key).choice(len(names), p=shares / shares.sum()))]

    def choose_technique(self, load_kg: float) -> str:
        """Habitual technique, except that a stooper switches to a semi-squat more often as the load
        grows (stoops appear mostly with light loads, Burgess-Limerick & Abernethy 1997) [design]."""
        technique = self.preferred_technique()
        p_switch = float(np.clip((load_kg - 10.0) / 20.0, 0.0, 0.7))
        if technique == "stoop" and self.lift_rng().random() < p_switch:
            technique = "semi_squat"
        return technique

    def _lift_pose(self, kappa: float, trunk: float, lumbar_deg: float):
        """Pelvis pitch and joint flexion for upper-trunk inclination `trunk` (rad): the spine takes
        the drawn lumbar flexion (scaled to all three joints), kept within 35-65 % of the trunk
        angle and the joint range, and so that the pelvis stays within PELVIS_PITCH_MAX; the pelvis
        takes the rest."""
        spine = np.deg2rad(lumbar_deg) / LUMBAR_SHARE
        low = max(0.35 * trunk, trunk - PELVIS_PITCH_MAX)
        spine = float(np.clip(spine, low, max(low, min(0.65 * trunk, np.deg2rad(28.0) / SPINE_FLEX_SPLIT[0]))))
        return trunk - spine, spine * SPINE_FLEX_SPLIT

    def _shoulder(self, kappa: float, pitch: float, flex: np.ndarray):
        """Shoulder (forward, up) relative to the ground point under the ankle."""
        h = self.height
        f, y = lift_root(kappa, pitch, h)
        angle = pitch
        for length, bend in zip(self.worker.spacings(), flex, strict=True):
            f, y = f + length * np.sin(angle), y + length * np.cos(angle)
            angle += bend
        f += SHOULDER_UP_H * h * np.sin(angle) + SHOULDER_ANTERIOR_H * h * np.cos(angle)
        y += SHOULDER_UP_H * h * np.cos(angle) - SHOULDER_ANTERIOR_H * h * np.sin(angle)
        return f, y

    def _grip_gap(self, kappa: float, trunk: float, lumbar_deg: float, grip_h: float):
        """(gap, grip): distance (m) from the shoulder to the grip beyond the reach, and the grip
        (forward, up). The grip sits under the shoulder, or ahead of the knee when that is further."""
        pitch, flex = self._lift_pose(kappa, trunk, lumbar_deg)
        sf, sy = self._shoulder(kappa, pitch, flex)
        knee_f = SHANK_H * self.height * np.sin(shank_tilt(kappa))
        gf = max(knee_f + OBJECT_CLEARANCE_M, sf)
        reach = LIFT_REACH * ARM_H * self.height
        return float(np.hypot(gf - sf, sy - grip_h) - reach), (gf, grip_h)

    def lift_posture(self, grip_h: float, technique: str, lumbar_deg: float, leg_share: float) -> dict:
        """Lowest posture of a lift: the legs drop the hips by leg_share of the hands' drop to the
        grip (`grip_h` m above the support), then the trunk inclines until the hands reach it. If
        the trunk cannot reach at TRUNK_MAX the knees bend further."""
        h = self.height
        k_stand = knee_at_stand(self.worker.pelvis_height / h)
        _, sy0 = self._shoulder(k_stand, 0.0, np.zeros(3))
        need = max(sy0 - LIFT_REACH * ARM_H * h - grip_h, 0.0)
        hip0 = lift_hip(k_stand, h)[1]
        lo, hi = k_stand, KNEE_MAX
        for _ in range(40):                              # knee flexion for the legs' share of the drop
            mid = (lo + hi) / 2
            if hip0 - lift_hip(mid, h)[1] < leg_share * need:
                lo = mid
            else:
                hi = mid
        kappa = lo
        while True:
            if self._grip_gap(kappa, TRUNK_MAX, lumbar_deg, grip_h)[0] <= 0 or kappa >= KNEE_MAX:
                break
            kappa = min(kappa + np.deg2rad(5.0), KNEE_MAX)
        lo, hi = 0.0, TRUNK_MAX
        if self._grip_gap(kappa, 0.0, lumbar_deg, grip_h)[0] <= 0:
            hi = 0.0
        for _ in range(40):                              # least trunk inclination that reaches
            mid = (lo + hi) / 2
            if self._grip_gap(kappa, mid, lumbar_deg, grip_h)[0] > 0:
                lo = mid
            else:
                hi = mid
        trunk = hi
        pitch, flex = self._lift_pose(kappa, trunk, lumbar_deg)
        grip = self._grip_gap(kappa, trunk, lumbar_deg, grip_h)[1]
        return {"technique": technique, "knee": kappa, "knee_stand": k_stand, "trunk": trunk, "pitch": pitch,
                "flex": flex, "grip": grip, "pelvis_drop": float(lift_root(k_stand, 0.0, h)[1] - lift_root(kappa, pitch, h)[1])}

    def _lift_segment(self, duration: float, start: dict, end: dict, windows: dict, phase: str) -> None:
        """One body move between two lift postures (dicts with knee, pitch, flex). Each channel
        follows its own minimum-jerk window (fractions of the duration), and the spine root is
        placed from the leg geometry every sample, so the feet stay where they are."""
        s = self.seg(duration, phase)
        u = np.arange(s.k) / max(s.k - 1, 1)
        turn = {"arrive": windows.get("arrive", 0.0), "depart": windows.get("depart", 0.0)}
        w_legs, w_pitch, w_spine = (window_ramp(u, windows[key], **turn) for key in ("legs", "pitch", "spine"))
        kappa = start["knee"] + (end["knee"] - start["knee"]) * w_legs
        pitch = start["pitch"] + (end["pitch"] - start["pitch"]) * w_pitch
        root_f, root_y = lift_root(kappa, pitch, self.height)
        f0, y0 = lift_root(start["knee"], start["pitch"], self.height)
        fwd = forward_vec(self.pose.heading)
        s.pos = self.pose.pos + np.outer(root_f - f0, fwd)
        s.pos[:, 1] = self.pose.pos[1] + (root_y - y0)
        if "y_end" in end:                               # settle onto the standing height at the end
            s.pos[:, 1] += (end["y_end"] - (self.pose.pos[1] + root_y[-1] - y0)) * w_legs
        s.pitch = pitch
        s.flex = start["flex"] + np.outer(w_spine, end["flex"] - start["flex"])
        self.commit(s)

    @staticmethod
    def _rate_limited(duration: float, start: dict, end: dict, windows: dict,
                      thorax_rate: float = LIFT_THORAX_RATE) -> float:
        """Stretch a lift move so that no trunk segment turns faster than MAX_VOLUNTARY_RATE (the
        pelvis, each spinal joint, and each segment's inclination: the pelvis pitch plus the joints
        below it, whose windows overlap and so add up) and the thorax no faster than thorax_rate."""
        u = np.linspace(0.0, 1.0, 2001)
        turn = {"arrive": windows.get("arrive", 0.0), "depart": windows.get("depart", 0.0)}
        pitch = (end["pitch"] - start["pitch"]) * window_ramp(u, windows["pitch"], **turn)
        flex = np.outer(window_ramp(u, windows["spine"], **turn), end["flex"] - start["flex"])
        joints = np.column_stack([pitch, flex])
        thorax = pitch + flex.sum(axis=1)
        rate = lambda x: float(np.max(np.abs(np.diff(x, axis=0)))) / (u[1] - u[0])   # rad per unit duration
        need = max(rate(joints), rate(pitch[:, None] + np.cumsum(flex, axis=1))) / MAX_VOLUNTARY_RATE
        need = max(need, rate(thorax) / thorax_rate)
        return max(duration, 1.02 * need)

    def _lift_plan(self, grip_h: float, load_kg: float, technique: str | None, total_s: float | None, kind: str):
        """Technique, lowest posture and durations for a lift ('lift') or a put-down ('put_down')."""
        rng = self.lift_rng()
        if technique is None:
            technique = self.choose_technique(load_kg)
        if technique not in LIFT_TECHNIQUES:
            raise ValueError(f"unknown lifting technique {technique!r}")
        mean, sd = LIFT_LUMBAR_DEG[technique]
        lumbar = float(np.clip(rng.normal(mean, sd), 10.0, 55.0))
        share = float(np.clip(LIFT_LEG_SHARE[technique] + rng.normal(0, 0.04), 0.02, 0.8))
        low = self.lift_posture(grip_h, technique, lumbar, share)
        stand = {"knee": low["knee_stand"], "pitch": 0.0, "flex": np.zeros(3)}
        start = {"knee": low["knee_stand"], "pitch": self.pose.pitch, "flex": self.pose.flex.copy()}
        leg_scale = np.sqrt(self.height * 0.53 / 0.9)    # Hof time scale for this worker's leg
        up_mean, up_sd = LIFT_UP_S[technique]
        down_s = float(rng.uniform(*LIFT_DOWN_S))
        up_s = float(np.clip(rng.normal(up_mean, up_sd), 1.1, 2.2)) * leg_scale
        load = load_kg if kind == "lift" else 0.0
        lead = LIFT_SPINE_LEAD[0] + (LIFT_SPINE_LEAD[1] - LIFT_SPINE_LEAD[0]) * float(np.clip(load / 25.0, 0.0, 1.0))
        rise = {"legs": LIFT_LEGS_WINDOW, "pitch": (lead / 2, 0.95), "spine": (lead, 1.0), "depart": LIFT_TURN}
        lower = {"legs": (0.0, 1.0), "pitch": (0.0, 1.0), "spine": (0.0, 1.0), "arrive": LIFT_TURN}
        down_s = self._rate_limited(down_s, start, low, lower)
        up_s = self._rate_limited(up_s, low, stand, rise)
        hold_s = float(rng.uniform(*LIFT_HOLD_S))
        settle_s = 0.0
        if total_s is not None:
            # Fit the run's time budget: what is left after the hold is spent standing upright after
            # the rise (settle). A tight budget shortens the hold (to LIFT_MIN_HOLD_S) and may push
            # the thorax past LIFT_THORAX_RATE, never past MAX_VOLUNTARY_RATE.
            spare = total_s - LIFT_MIN_HOLD_S
            if down_s + up_s > spare:
                scale = spare / (down_s + up_s)
                down_s = self._rate_limited(down_s * scale, start, low, lower, MAX_VOLUNTARY_RATE)
                up_s = self._rate_limited(up_s * scale, low, stand, rise, MAX_VOLUNTARY_RATE)
            k_total = int(round(total_s * self.fs))
            k_down, k_up = int(round(down_s * self.fs)), int(round(up_s * self.fs))
            k_free = max(k_total - k_down - k_up, int(LIFT_MIN_HOLD_S * self.fs))
            k_hold = min(max(int(round(hold_s * self.fs)), int(LIFT_MIN_HOLD_S * self.fs)), k_free)
            k_settle = k_free - k_hold
            if k_settle < 2:                             # a segment needs 2 samples: give it to the hold
                k_hold, k_settle = k_free, 0
            hold_s, settle_s = k_hold / self.fs, k_settle / self.fs
        return {"technique": technique, "lumbar_deg": lumbar, "low": low, "start": start, "stand": stand,
                "down_s": down_s, "hold_s": hold_s, "up_s": up_s, "settle_s": settle_s, "rise": rise, "lower": lower}

    def _grip_world(self, grip: tuple) -> np.ndarray:
        """World position of a grip (forward, up) given relative to the ankles, at the current pose."""
        h = self.height
        k_stand = knee_at_stand(self.worker.pelvis_height / h)
        root_f = lift_root(k_stand, self.pose.pitch, h)[0]
        ankle = self.pose.pos - forward_vec(self.pose.heading) * root_f
        point = ankle + forward_vec(self.pose.heading) * grip[0]
        point[1] = self.ground_y + grip[1]
        return point

    def _lift_info(self, plan: dict, load_kg: float, object_height: float, grip: np.ndarray) -> dict:
        """Event fields of a lift or put-down: the planned technique (the worker's habit, which sets
        the leg share and timing), the knee class actually reached (knee_class: where the trunk is
        capped at TRUNK_MAX the knees bend further, so a planned stoop can end as a semi-squat), the
        lowest posture and the world grip point."""
        low = plan["low"]
        knee_deg = round(float(np.rad2deg(low["knee"])), 1)
        return {"technique": plan["technique"], "knee_class": knee_class(knee_deg), "load_kg": float(load_kg),
                "object_height": float(object_height), "knee_deg": knee_deg,
                "knee_stand_deg": round(float(np.rad2deg(low["knee_stand"])), 1),
                "trunk_deg": round(float(np.rad2deg(low["trunk"])), 1), "pelvis_drop_m": round(low["pelvis_drop"], 3),
                "gx": float(grip[0]), "gy": float(grip[1]), "gz": float(grip[2])}

    def lift(self, object_height: float, load_kg: float, technique: str | None = None, total_s: float | None = None,
             phase: str = "activity") -> None:
        """Lift a load gripped `object_height` m above the support, standing with the feet still.

        Lower the body (legs and trunk together), set and grip, then a leg-drive rise: the knees
        extend first, then the hips, then the lumbar spine (Burgess-Limerick 1995). technique is
        'stoop', 'semi_squat' or 'squat' (None: the worker's habit, see choose_technique); the events
        also name the class of the knee angle reached (knee_class, see _lift_info). total_s fixes the
        whole lift's duration (after a short hold, the rest is spent standing upright). Events:
        lift_grasp (hands on the load, with the grip point), lift_off (load leaves the support),
        lift_end (upright).
        """
        plan = self._lift_plan(object_height, load_kg, technique, total_s, "lift")
        low = plan["low"]
        grip = self._grip_world(low["grip"])
        info = self._lift_info(plan, load_kg, object_height, grip)
        y_stand = self.ground_y + self.worker.pelvis_height
        self._lift_segment(plan["down_s"], plan["start"], low, plan["lower"], phase)
        self.event("lift_grasp", **info)
        self.hold(plan["hold_s"], phase, sway=0.5)
        self.event("lift_off", **info)
        self._lift_segment(plan["up_s"], low, {**plan["stand"], "y_end": y_stand}, plan["rise"], phase)
        self.event("lift_end", technique=plan["technique"], knee_class=info["knee_class"])
        if plan["settle_s"] > 0:
            self.hold(plan["settle_s"], phase, sway=1.0)        # stand with the load before walking off

    def put_down(self, object_height: float, load_kg: float, technique: str | None = None,
                 total_s: float | None = None, phase: str = "activity") -> None:
        """Set a carried load down, held `object_height` m above the support at touchdown, then let go
        and stand up (the same technique and geometry as lift()). Events: put_down_touch (load on
        the support, with the grip point), put_down_release (hands off), put_down_end (upright)."""
        plan = self._lift_plan(object_height, load_kg, technique, total_s, "put_down")
        low = plan["low"]
        grip = self._grip_world(low["grip"])
        info = self._lift_info(plan, load_kg, object_height, grip)
        y_stand = self.ground_y + self.worker.pelvis_height
        self._lift_segment(plan["down_s"], plan["start"], low, plan["lower"], phase)
        self.event("put_down_touch", **info)
        self.hold(plan["hold_s"], phase, sway=0.5)
        self.event("put_down_release", **info)
        self._lift_segment(plan["up_s"], low, {**plan["stand"], "y_end": y_stand}, plan["rise"], phase)
        self.event("put_down_end", technique=plan["technique"], knee_class=info["knee_class"])
        if plan["settle_s"] > 0:
            self.hold(plan["settle_s"], phase, sway=1.0)

    def rebar_tie(self, duration: float, phase: str = "activity") -> None:
        """Sustained stoop at 70-100 deg with small tying jerks and short shuffles."""
        depth = np.deg2rad(self.rng.uniform(70, 100))
        hip = depth * 0.5
        self.move(1.5, phase, y=self.ground_y + self.worker.pelvis_height - 0.05, pitch=hip,
                  flex=(depth - hip) * np.array([0.45, 0.35, 0.2]))
        elapsed = 0.0
        while elapsed < duration:
            chunk = float(self.rng.uniform(20, 40))
            s = self.seg(chunk, phase, sway=0.4)
            self.impulses(s, np.arange(1.0, chunk, self.rng.uniform(3, 5)), (0.1, 0.3), pelvis_g=0.05)
            self.commit(s)
            s = self.seg(1.5, phase)
            s.pos += np.outer(0.4 * minjerk(s.k), forward_vec(self.pose.heading))
            self.commit(s)
            elapsed += chunk + 1.5
        self.stand_up(1.5, phase)

    def keyed_rng(self, tag: int) -> np.random.Generator:
        """Generator for extra detail keyed on the worker and the current sample: it never advances
        the main stream, so adding detail leaves every other random number of a run unchanged."""
        key = [tag, int(round(self.worker.height_m * 1e6)), int(round(self.worker.mass_kg * 1e6)), self.n]
        return np.random.default_rng(key)

    def kneel_work(self, duration: float, phase: str = "activity") -> None:
        """Kneeling floor work: kneel to a working trunk depth, then keep moving the trunk (reaching,
        placing, looking up) around it, with small tool jolts."""
        u = (float(self.rng.uniform(45, 90)) - 45.0) / 45.0     # the old main-stream draw, rescaled
        depth = KNEEL_DEPTH_DEG[0] + (KNEEL_DEPTH_DEG[1] - KNEEL_DEPTH_DEG[0]) * u
        self.kneel(depth, 2.0, phase)
        s = self.seg(duration, phase, sway=0.3)
        self.impulses(s, np.arange(2.0, duration, self.rng.uniform(4, 8)), (0.1, 0.4), pelvis_g=0.05)
        # Trunk excursions (deg), kept between about upright and 80 deg, faded in and out.
        moves = KNEEL_MOVE_SD_DEG * band_noise(self.keyed_rng(0x4B4E), s.k, self.fs, 0.08, 0.5)
        trunk = soft_limit(depth + moves, 3.0, 80.0, knee=5.0) - depth
        trunk = np.deg2rad(trunk) * taper(s.k, self.fs, 1.5)
        s.pitch = s.pitch + 0.6 * trunk                               # the same hip share as kneel()
        s.flex = s.flex + np.outer(0.4 * trunk, [0.45, 0.35, 0.20])
        self.commit(s)
        self.stand_up(1.5, phase)

    def sit(self, seat_height: float = 0.45, hard: bool = False, phase: str = "activity") -> None:
        """Sit on a bucket/bench. hard=True: last 0.1-0.3 s near free fall, then a 2-4 g seat impact."""
        seated_y = self.ground_y + seat_height + 0.08
        if not hard:
            self.move(1.5, phase, y=seated_y, pitch=np.deg2rad(-5), flex=np.deg2rad([6, 4, 2]))
            return
        drop = float(self.rng.uniform(0.04, 0.15))
        lean = np.deg2rad(SIT_LEAN_DEG)
        self.move(0.8, phase, y=seated_y + drop, pitch=SIT_PELVIS_SHARE * lean,
                  flex=(1.0 - SIT_PELVIS_SHARE) * lean * SPINE_FLEX_SPLIT)
        dip = float(self.rng.uniform(0.5, 0.8))
        y, v, ys = seated_y + drop, 0.0, []
        k_seat, m = float(self.rng.uniform(2.5e4, 5e4)), 30.0
        c = 2 * 0.4 * np.sqrt(k_seat * m)
        contact = None  # sample of the seat contact
        while contact is None or len(ys) < contact + max(int(SIT_SETTLE_S * self.fs), int(0.6 * self.fs)):
            if y > seated_y:
                a = -(1 - dip) * G  # specific force = dip * g while dropping onto the seat
            else:
                force = max(k_seat * (seated_y - y) - c * v, 0.0)  # the seat only pushes
                a = force / m - G
                if contact is None:
                    contact = len(ys)
            v += a * self.dt
            y += v * self.dt
            ys.append(y)
        s = self.seg(len(ys) / self.fs, phase)
        s.pos[:, 1] = np.array(ys)[: s.k]
        # The trunk is already coming back up as the hips drop onto the seat, and goes on while it settles.
        back = minjerk(s.k)
        s.pitch = s.pitch + (np.deg2rad(SIT_SEATED_PITCH_DEG) - s.pitch) * back
        s.flex = s.flex + (np.deg2rad(SIT_SEATED_FLEX_DEG) - s.flex) * back[:, None]
        self.commit(s)

    def cab_ride(self, duration: float, phase: str = "activity") -> None:
        """Equipment seat: whole-body vibration 1-8 Hz, jolts 1-3 g, dips over bumps, vehicle motion."""
        self.ladder(1.0, up=True, phase=phase)
        self.sit(0.45, phase=phase)
        s = self.seg(duration, phase, sway=0.0)
        t = np.arange(s.k) * self.dt
        wbv = band_noise(self.rng, s.k, self.fs, 1.0, 8.0) * self.rng.uniform(0.3, 1.0)  # m/s^2 rms
        bumps = np.zeros(s.k)
        for c in self.rng.choice(s.k, size=max(int(duration / 8), 1), replace=False):
            width = int(self.rng.uniform(0.08, 0.2) * self.fs)
            g_bump = self.rng.uniform(1.0, 3.0) * G
            end = min(c + width, s.k)
            bumps[c:end] += g_bump * np.sin(np.pi * np.arange(end - c) / width)
            dip_end = min(end + width, s.k)
            bumps[end:dip_end] -= self.rng.uniform(0.1, 0.4) * G * np.sin(np.pi * np.arange(dip_end - end) / width)
        a = wbv + bumps
        a -= a.mean()
        spec = np.fft.rfft(a)
        freqs = np.fft.rfftfreq(s.k, self.dt)
        freqs[0] = 1.0
        disp = np.fft.irfft(-spec / (2 * np.pi * freqs) ** 2, n=s.k)
        disp = signal.sosfiltfilt(signal.butter(2, 0.3, "high", fs=self.fs, output="sos"), disp)
        s.pos[:, 1] += disp * taper(s.k, self.fs, 1.0)
        # Vehicle travel: slow speed changes and turns, starting and ending at rest.
        speed = np.clip(np.cumsum(band_noise(self.rng, s.k, self.fs, None, 0.05)) * self.dt * 0.5 + 2.0, 0, 4)
        speed *= taper(s.k, self.fs, 4.0)
        heading = self.pose.heading + np.cumsum(band_noise(self.rng, s.k, self.fs, None, 0.05)) * self.dt * 0.2
        step = np.stack([-np.sin(heading), np.zeros(s.k), -np.cos(heading)], axis=1)
        s.pos += np.cumsum(step * speed[:, None], axis=0) * self.dt
        s.heading = heading
        self.commit(s)
        self.vel = np.zeros(3)
        self.stand_up(1.2, phase)
        self.ladder(1.0, up=False, phase=phase)

    def jump_down(self, height: float, phase: str = "activity") -> None:
        """Planned jump from a step/truck bed: crouch, take off, ballistic drop, feet-first landing."""
        from .falls import feet_first_landing

        self.move(0.4, phase, y=self.pose.pos[1] - 0.10, pitch=np.deg2rad(15), flex=np.deg2rad([5, 4, 3]))
        stop = float(self.rng.uniform(0.25, 0.4))  # planned landing: deep knee flexion
        feet_first_landing(self, height, forward_speed=float(self.rng.uniform(0.5, 1.0)), stop_distance=stop,
                           takeoff_up=float(self.rng.uniform(0.3, 0.8)), phase=phase, label="jump")
        self.stand_up(0.8, phase)

    def lying_activity(self, duration: float, phase: str = "lying_work", amp_deg: float = 5.0) -> None:
        """Conscious work while lying (mechanic under a vehicle, crawl-space work): arm-driven
        trunk movements every few seconds with short pauses (sin^2 bumps, no velocity steps)."""
        remaining = duration
        while remaining > 0.5:
            seg = self.seg(min(float(self.rng.uniform(1.5, 4.0)), remaining), phase, fidget=2.0)
            shape = np.sin(np.pi * np.linspace(0, 1, seg.k)) ** 2
            seg.flex = seg.flex + np.outer(shape, np.deg2rad(self.rng.uniform(-amp_deg, amp_deg, 3)))
            seg.lat = seg.lat + np.outer(shape, np.deg2rad(self.rng.uniform(-amp_deg, amp_deg, 3)) * 0.6)
            seg.pos[:, 1] += 0.01 * shape * self.rng.uniform(0.2, 1.0)
            self.commit(seg)
            remaining -= seg.k * self.dt
            if remaining > 0.5:
                pause = self.seg(min(float(self.rng.uniform(1.0, 6.0)), remaining), phase, fidget=1.5)
                self.commit(pause)
                remaining -= pause.k * self.dt

    def remove_strip(self, table_height: float = 0.75) -> None:
        """End of shift: take the strip off and leave it on a table (no breathing, no motion)."""
        self.move(2.0, "activity", twist=np.deg2rad([5, 8, 10]), lat=np.deg2rad([2, 3, 4]))
        self.event("strip_removed")
        s = self.seg(3.0, "device_off", fidget=0.0)
        s.worn[:] = False
        s.breath_amp[:] = 0
        s.sway[:] = 0
        w = minjerk(s.k)
        target_pitch = np.deg2rad(-90 if self.rng.random() < 0.5 else 90)
        s.pitch = self.pose.pitch + (target_pitch - self.pose.pitch) * w
        s.flex = self.pose.flex * (1 - w)[:, None]
        s.lat = self.pose.lat * (1 - w)[:, None]
        s.twist = self.pose.twist * (1 - w)[:, None]
        s.pos[:, 1] = self.pose.pos[1] + (self.ground_y + table_height - self.pose.pos[1]) * w + 0.15 * np.sin(np.pi * w)
        s.pos += np.outer(0.4 * w, forward_vec(self.pose.heading))
        self.commit(s)

    def strap_slip(self, slot: int, degrees: float) -> None:
        """The strap shifts on one anchor: a step change in that sensor's mounting rotation."""
        axis = self.rng.normal(size=3)
        axis[1] *= 0.3
        axis /= np.linalg.norm(axis)
        self.mount_steps.append((self.t, slot, axis * np.deg2rad(degrees)))
        self.event("strap_slip", slot=slot, degrees=degrees)

    def hat_off(self, phase: str = "activity") -> None:
        """Hard hat knocked off: cable tug on the upper-thoracic anchor (1-5 g, <50 ms)."""
        s = self.seg(1.0, phase, sway=1.0)
        g_peak = self.rng.uniform(1.0, 5.0)
        pulse = smooth_pulse(self.fs, 25.0, 0.008) * g_peak * G / self.worker.spacings()[2]
        s.flex[100:100 + len(pulse), 2] += pulse
        rot = self.rng.uniform(0.5, 2.0)  # rad/s peak, brief twist of the upper segment
        width = int(0.3 * self.fs)
        # sin^2 bump: zero rate at both ends, peak rate = rot.
        s.twist[100:100 + width, 2] += rot * 0.3 / np.pi * np.sin(np.pi * np.arange(width) / width) ** 2
        self.commit(s)
        self.event("hat_off", peak_g=g_peak)

    # --- posture primitives (used by posture.py; existing primitives are unchanged) -------
    def posture_hold(self, trunk_deg: float, hold_s: float, lat_deg: float = 0.0, twist_deg: float = 0.0,
                     pelvis_share: float = 0.45, down_s: float = 1.5, drift_deg: float = 3.0, sway: float = 0.4,
                     phase: str = "activity", return_up: bool = True) -> None:
        """Move into a trunk posture and hold it, with slow natural drift, then (optionally) stand.

        trunk_deg is the upper-thoracic inclination from upright (forward +): pelvis_share of it at
        the hip (lumbo-pelvic rhythm), the rest as spinal flexion. lat_deg / twist_deg are side bend
        (right +) and axial twist, mostly in the spine. Drift is +/-drift_deg (RMS) at < 0.2 Hz.
        """
        trunk = np.deg2rad(trunk_deg)
        hip = trunk * pelvis_share
        lat, twist = np.deg2rad(lat_deg), np.deg2rad(twist_deg)
        drop = 0.03 + 0.05 * max(np.sin(hip), 0.0)
        self.move(down_s, phase, y=self.ground_y + self.worker.pelvis_height - drop, pitch=hip, roll=0.3 * lat,
                  flex=(trunk - hip) * np.array([0.45, 0.35, 0.20]), lat=0.7 * lat * np.array([0.3, 0.35, 0.35]),
                  twist=twist * np.array([0.2, 0.4, 0.4]))
        s = self.seg(hold_s, phase, sway=sway)
        if drift_deg > 0:
            env = taper(s.k, self.fs, 1.0)
            for arr, share in ((s.flex, np.array([0.4, 0.35, 0.25])), (s.lat, np.array([0.3, 0.35, 0.35]))):
                arr += np.outer(np.deg2rad(drift_deg) * band_noise(self.rng, s.k, self.fs, None, 0.2) * env, share)
        self.commit(s)
        if return_up:
            self.stand_up(1.5, phase)

    def functional_calibration(self, bends: int = 3, depth_deg: float = 40.0) -> None:
        """Three slow forward bends after the neutral stand (spine-model note, Q2 step 3): they
        reveal each sensor's sagittal plane so mounting rotation about the spine can be removed."""
        self.event("functional_cal_start")
        for _ in range(bends):
            self.posture_hold(depth_deg, 1.0, down_s=1.8, drift_deg=0.0, sway=0.2)
            self.hold(1.0, sway=0.5)
        self.event("functional_cal_end")

    def sit_posture(self, duration: float, slouch_deg: float = 0.0, seat_height: float = 0.45,
                    phase: str = "activity") -> None:
        """Sit on a bench or seat; slouch_deg > 0 tilts the pelvis back and flexes the lumbar and
        thoracic spine (a slumped spinal curve with a near-upright trunk)."""
        self.sit(seat_height, phase=phase)
        slouch = np.deg2rad(slouch_deg)
        self.move(2.0, phase, pitch=-0.35 * slouch, flex=slouch * np.array([0.55, 0.45, 0.35]))
        s = self.seg(duration, phase, sway=0.2)
        env = taper(s.k, self.fs, 1.0)
        s.flex += np.outer(np.deg2rad(2.0) * band_noise(self.rng, s.k, self.fs, None, 0.15) * env, [0.4, 0.35, 0.25])
        self.commit(s)
        self.move(1.5, phase, pitch=np.deg2rad(-5), flex=np.deg2rad([6, 4, 2]))
        self.stand_up(1.5, phase)

    # --- finish --------------------------------------------------------------------------
    def build(self) -> PoseTrack:
        cat = lambda name: np.concatenate([getattr(s, name) for s in self.segs])
        pos, heading, pitch, roll = cat("pos"), cat("heading"), cat("pitch"), cat("roll")
        flex, lat, twist = cat("flex"), cat("lat"), cat("twist")
        phase, worn = cat("phase"), cat("worn")
        breath_amp, breath_hz, fidget, sway = cat("breath_amp"), cat("breath_hz"), cat("fidget"), cat("sway")
        n = len(pos)
        # Overlay envelopes change per segment; smooth them (1 Hz) so switching sway, fidget or
        # breathing on/off never creates a position step (which would read as a fake impact).
        env_sos = signal.butter(2, 1.0, fs=self.fs, output="sos")
        if n > 100:
            breath_amp = np.clip(signal.sosfiltfilt(env_sos, breath_amp), 0, None)
            fidget = np.clip(signal.sosfiltfilt(env_sos, fidget), 0, None)
            sway = np.clip(signal.sosfiltfilt(env_sos, sway), 0, None)
        # Breathing: thoracic/upper flexion oscillation (~ +/-0.01 g projection) and 1 mm bob.
        breath = np.sin(2 * np.pi * np.cumsum(breath_hz) * self.dt)
        flex[:, 1] += 0.5 * breath_amp * breath
        flex[:, 2] += breath_amp * breath
        pos[:, 1] += 0.001 * breath * (breath_amp > 0)
        # Postural sway while standing: slow < 0.5 Hz drift of pelvis position and lean.
        for axis in (0, 2):
            pos[:, axis] += SWAY_POS_M * sway * band_noise(self.rng, n, self.fs, None, 0.4)
        pitch += np.deg2rad(0.4) * sway * band_noise(self.rng, n, self.fs, None, 0.4)
        roll += np.deg2rad(0.3) * sway * band_noise(self.rng, n, self.fs, None, 0.4)
        # Fidget: small involuntary joint motion of a conscious wearer.
        for arr in (flex, lat, twist):
            for j in range(3):
                arr[:, j] += np.deg2rad(0.12) * fidget * band_noise(self.rng, n, self.fs, None, 2.0)
        # Light zero-phase smoothing removes numerical kinks at segment joins (keeps impacts < 120 Hz).
        sos = signal.butter(4, 120.0, fs=self.fs, output="sos")
        smooth = lambda x: signal.sosfiltfilt(sos, x, axis=0) if len(x) > 60 else x
        pos, heading, pitch, roll = smooth(pos), smooth(heading), smooth(pitch), smooth(roll)
        flex, lat, twist = smooth(flex), smooth(lat), smooth(twist)
        mount_delta = np.zeros((n, 4, 3))
        for t0, slot, rotvec in self.mount_steps:
            start = int(t0 * self.fs)
            ramp_len = int(0.3 * self.fs)
            w = np.ones(n - start)
            w[: min(ramp_len, n - start)] = minjerk(min(ramp_len, n - start))
            mount_delta[start:, slot] += np.outer(w, rotvec)
        track = PoseTrack(self.fs, pos, heading, pitch, roll, flex, lat, twist, phase, worn, mount_delta)
        clamp_joints(track)
        return track
