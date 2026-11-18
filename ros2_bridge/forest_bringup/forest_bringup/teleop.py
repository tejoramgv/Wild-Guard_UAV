"""Keyboard teleoperation state machine.

The keyboard node owns input capture; this module owns the mapping and the
continuous-command semantics, which are what the offline tests exercise.
"""

import math
from dataclasses import dataclass

from forest_bringup.safety import VelocityCommand

# Canonical key names used by the ROS node after normalising raw key events.
FORWARD = "up"
BACKWARD = "down"
LEFT = "left"
RIGHT = "right"
ASCEND = "w"
DESCEND = "s"
YAW_LEFT = "a"
YAW_RIGHT = "d"
HOVER = "space"
LAND = "l"
TAKEOFF = "t"
ESTOP = "esc"

MOVEMENT_KEYS = {FORWARD, BACKWARD, LEFT, RIGHT, ASCEND, DESCEND, YAW_LEFT, YAW_RIGHT, HOVER}
EVENT_KEYS = {LAND, TAKEOFF, ESTOP}

DEFAULT_KEYMAP = {
    "up": FORWARD,
    "down": BACKWARD,
    "left": LEFT,
    "right": RIGHT,
    "w": ASCEND,
    "s": DESCEND,
    "a": YAW_LEFT,
    "d": YAW_RIGHT,
    " ": HOVER,
    "l": LAND,
    "t": TAKEOFF,
    "esc": ESTOP,
}


@dataclass
class TeleopConfig:
    max_speed: float = 2.0
    max_vertical_speed: float = 1.5
    max_yaw_rate: float = 1.2


class TeleopState:
    """Tracks currently-held keys and emits a continuous velocity command.

    Pressing a key starts a command that persists until the matching release,
    which is the behaviour a real joystick/keyboard has. Discrete actions
    (takeoff/land/estop) are queued and consumed by the ROS node.
    """

    def __init__(self, config=None):
        self.config = config or TeleopConfig()
        self._pressed = set()
        self._events = []
        self._hover_latched = False

    @property
    def pressed(self):
        return frozenset(self._pressed)

    def press(self, key):
        key = DEFAULT_KEYMAP.get(key, key)
        if key in EVENT_KEYS:
            if not self._events or self._events[-1] != key:
                self._events.append(key)
            if key == ESTOP:
                self._pressed.clear()
            return
        if key not in MOVEMENT_KEYS:
            return
        if key == HOVER:
            self._hover_latched = True
            self._pressed.clear()
            return
        self._pressed.add(key)

    def release(self, key):
        key = DEFAULT_KEYMAP.get(key, key)
        if key == HOVER:
            return
        self._pressed.discard(key)

    def pop_events(self):
        events, self._events = self._events, []
        return events

    def compute(self, dt=0.0):
        cfg = self.config
        if self._hover_latched:
            self._hover_latched = False
            return VelocityCommand()
        vx = (1.0 if FORWARD in self._pressed else 0.0) - (1.0 if BACKWARD in self._pressed else 0.0)
        vy = (1.0 if RIGHT in self._pressed else 0.0) - (1.0 if LEFT in self._pressed else 0.0)
        vz = (1.0 if ASCEND in self._pressed else 0.0) - (1.0 if DESCEND in self._pressed else 0.0)
        wz = (1.0 if YAW_LEFT in self._pressed else 0.0) - (1.0 if YAW_RIGHT in self._pressed else 0.0)
        horiz = (vx * vx + vy * vy) ** 0.5
        if horiz > 1.0:
            vx, vy = vx / horiz, vy / horiz
        return VelocityCommand(
            vx=vx * cfg.max_speed,
            vy=vy * cfg.max_speed,
            vz=vz * cfg.max_vertical_speed,
            wz=wz * cfg.max_yaw_rate,
        )

    def all_released(self):
        return not self._pressed and not self._hover_latched


def body_to_world(vx_forward, vy_right, yaw):
    """Heading-relative (forward/right) velocity -> world-frame ENU.

    ``yaw`` is the ENU heading (radians, CCW from +X). Right is a -90 degree
    rotation of forward, so at yaw=0 (facing +X/east) "right" is -Y/south.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    return (vx_forward * c + vy_right * s,
            vx_forward * s - vy_right * c)


def world_to_body(vx, vy, yaw):
    """World-frame ENU velocity -> (forward, right) components."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (vx * c + vy * s,
            vx * s - vy * c)


def project_forward(vx, vy, yaw):
    """Scalar speed along the current heading (can be negative)."""
    return world_to_body(vx, vy, yaw)[0]
