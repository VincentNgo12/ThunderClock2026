# File: realism/__init__.py
# Purpose: Realism scorecard for ThunderSim: compares simulated trunk-IMU signals with real
#          recordings (Erciyes UCI 455, VTT-ConIot), like for like, and ranks the gaps.
# Contributors: Unassigned
"""Run as `.venv/bin/python -m realism score --out <scratch dir>` from detection/simulation/.

Modules:
- paths.py    where the real data lives (outside the repo) and dataset sensor specs
- real.py     loads real recordings into the segment frame (read-only, reuses harmonize.py)
- emulate.py  generates simulator runs and emulates each real sensor on the simulated body
- mapping.py  simulator scenario/activity <-> real category mapping (with uncertainty notes)
- features.py event-level (falls) and activity-level (windows) features
- stats.py    distribution distances with bootstrap CIs, real split-half noise floor, C2ST
- score.py    orchestration, downstream (falldet) impact, tables, plots
"""
