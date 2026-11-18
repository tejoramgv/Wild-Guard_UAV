"""Flight safety: limits, command arbitration and loss-of-command handling.

Pure Python (no ROS), so every rule below is unit-testable offline.
"""

from dataclasses import dataclass
import math


@dataclass
class FlightLimits:
    max_speed: float = 2.0              # horizontal speed, m/s
    max_vertical_speed: float = 1.5     # m/s
    max_yaw_rate: float = 1.2           # rad/s
    max_acceleration: float = 2.5       # m/s^2, applied by the rate limiter
    max_altitude: float = 25.0          # metres above ground_z
    min_altitude: float = 0.3
    ground_z: float = 0.0
    command_timeout: float = 0.5        # seconds without a command -> hover
    heartbeat_timeout: float = 2.0      # seconds without telemetry -> hover


@dataclass
class VelocityCommand:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    wz: float = 0.0

    def as_tuple(self):
        return (self.vx, self.vy, self.vz, self.wz)

    def is_zero(self, tol=1e-6):
        return all(abs(v) <= tol for v in self.as_tuple())


ZERO_COMMAND = VelocityCommand()


@dataclass
class SafetyStatus:
    command: VelocityCommand
    reason: str
    emergency: bool = False
    stale: bool = False
    limited: bool = False


def clamp_command(cmd, limits, position_enu=None):
    """Clamp a velocity command to the configured envelope.

    Horizontal speed is limited as a vector magnitude (not per axis) so a
    diagonal command cannot exceed ``max_speed``. Altitude limits are applied
    only when a position estimate is available.
    """
    vx, vy, vz, wz = cmd.vx, cmd.vy, cmd.vz, cmd.wz
    horiz = math.hypot(vx, vy)
    if horiz > limits.max_speed:
        scale = limits.max_speed / horiz
        vx, vy = vx * scale, vy * scale
    vz = max(-limits.max_vertical_speed, min(limits.max_vertical_speed, vz))
    if position_enu is not None:
        altitude = position_enu[2] - limits.ground_z
        if altitude >= limits.max_altitude and vz > 0.0:
            vz = 0.0
        if altitude <= limits.min_altitude and vz < 0.0:
            vz = 0.0
    wz = max(-limits.max_yaw_rate, min(limits.max_yaw_rate, wz))
    return VelocityCommand(vx, vy, vz, wz)


def rate_limit(current, target, max_acceleration, dt):
    """Limit the change between two commands to ``max_acceleration * dt``."""
    if dt <= 0.0:
        return VelocityCommand(*current.as_tuple())
    max_delta = max_acceleration * dt
    values = []
    for c, t in zip(current.as_tuple(), target.as_tuple()):
        delta = max(-max_delta, min(max_delta, t - c))
        values.append(c + delta)
    return VelocityCommand(*values)


class CommandArbiter:
    """Owns the single authoritative velocity command.

    Priority: emergency stop > obstacle avoidance > manual teleop > hover.
    Commands expire, so a crashed node or a lost key release cannot leave the
    drone flying away.
    """

    def __init__(self, command_timeout=0.5):
        self.command_timeout = command_timeout
        self._manual = ZERO_COMMAND
        self._manual_stamp = -math.inf
        self._avoidance = ZERO_COMMAND
        self._avoidance_active = False
        self._avoidance_stamp = -math.inf
        self._emergency = False

    def set_manual(self, cmd, now):
        self._manual = cmd
        self._manual_stamp = now

    def set_avoidance(self, cmd, active, now):
        self._avoidance = cmd
        self._avoidance_active = active
        self._avoidance_stamp = now

    def emergency_stop(self):
        self._emergency = True

    def clear_emergency_stop(self):
        self._emergency = False

    @property
    def emergency(self):
        return self._emergency

    def _fresh(self, stamp, now):
        return (now - stamp) <= self.command_timeout

    def select(self, now):
        """Return ``(command, source)`` for the current control cycle."""
        if self._emergency:
            return ZERO_COMMAND, "emergency_stop"
        avoidance_fresh = self._fresh(self._avoidance_stamp, now)
        manual_fresh = self._fresh(self._manual_stamp, now)
        if avoidance_fresh and self._avoidance_active:
            return self._avoidance, "obstacle_avoidance"
        if manual_fresh:
            return self._manual, "manual"
        # Avoidance that is active but stale is treated as risk, not free path.
        if self._avoidance_active and not manual_fresh:
            return ZERO_COMMAND, "avoidance_stale_hover"
        return ZERO_COMMAND, "command_timeout_hover"


class SafetySupervisor:
    """Applies limits, timeouts and emergency stop to the selected command."""

    def __init__(self, limits=None):
        self.limits = limits or FlightLimits()
        self.arbiter = CommandArbiter(self.limits.command_timeout)
        self._last_telemetry = -math.inf
        self._last_output = ZERO_COMMAND
        self._last_time = None

    def update_telemetry(self, now):
        self._last_telemetry = now

    def evaluate(self, now, position_enu=None, use_avoidance=True):
        telemetry_lost = (now - self._last_telemetry) > self.limits.heartbeat_timeout
        if telemetry_lost:
            # Without a pose estimate we cannot guarantee safety: hover.
            safe = VelocityCommand()
            self._last_output = safe
            self._last_time = now
            return SafetyStatus(safe, "telemetry_lost_hover", emergency=False, stale=True)

        cmd, source = self.arbiter.select(now)
        if not use_avoidance and source == "obstacle_avoidance":
            cmd, source = ZERO_COMMAND, "avoidance_disabled_hover"

        if self._last_time is None:
            # First cycle: no previous output to ramp from.
            limited = cmd
        else:
            dt = max(0.0, now - self._last_time)
            limited = rate_limit(self._last_output, cmd,
                                 self.limits.max_acceleration, dt)
        clamped = clamp_command(limited, self.limits, position_enu)
        self._last_output = clamped
        self._last_time = now
        return SafetyStatus(
            command=clamped,
            reason=source,
            emergency=self.arbiter.emergency,
            stale=False,
            limited=(clamped.as_tuple() != cmd.as_tuple()),
        )
