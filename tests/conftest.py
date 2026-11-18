"""Make the ROS 2 Python packages importable without colcon installation.

Offline unit tests must work on a plain Linux box with only Python + pytest,
so the package roots are added to sys.path here instead of relying on an
installed ROS 2 workspace.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "ros2_bridge" / "forest_bringup", ROOT / "ros2_bridge"):
    p = str(candidate)
    if p not in sys.path:
        sys.path.insert(0, p)
