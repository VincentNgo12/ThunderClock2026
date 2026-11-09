# File: __init__.py
# Purpose: Detection hub package (docs/DETECTION_ARCHITECTURE.md): one MainModel that consumes the
#          raw sensor queue and runs falldet, the topple trigger, the collapse confirmer, the
#          man-down timer and the arbiter. Makes detection/models (falldet) importable.
# Contributors: Unassigned

import os
import sys

_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
if os.path.isdir(_MODELS) and _MODELS not in sys.path:
    sys.path.insert(0, _MODELS)  # same approach as thundersim.model_api.load_model("falldet")
