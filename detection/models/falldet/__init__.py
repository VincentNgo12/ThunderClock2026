"""ThunderClock 4-IMU spine-strip fall detector (two stages, classical ML).

    from falldet import detect, FallDetectorStream
"""
from .constants import ANCHORS, CLASSES, FS_HZ  # noqa: F401
from .detector import (Detector, FallDetectorStream, detect, fixture_arrays,  # noqa: F401
                       get_detector, load_fixture, parse_sample, window_around)
