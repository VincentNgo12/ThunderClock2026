# File: __init__.py
# Purpose: 3D viewer for ThunderSim runs: a local web page that replays one scenario with a
#          full human model, the 4-IMU strip and the physical scene the scenario implies.
# Contributors: Unassigned
"""ThunderSim 3D viewer.

Only the spine (4 rigid links), the sensor readings and the model events come from the
simulator. Legs, arms, head and the surrounding scene are illustrative: they are posed and
placed from what the scenario script did, so a person can follow the run.

    .venv/bin/python -m thundersim viz          # serves http://127.0.0.1:8765
"""
