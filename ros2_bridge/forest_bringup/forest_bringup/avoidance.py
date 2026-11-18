"""Local obstacle detection and avoidance planner.

Depth-based (primary) and LaserScan-based (optional) sector detection, plus a
committed-direction planner with hysteresis to avoid oscillation. Pure Python
so it can be tested without ROS, a simulator or numpy.
"""

from dataclasses import dataclass
import math


@dataclass
class AvoidanceConfig:
    min_safe_distance: float = 2.0       # start steering below this, m
    emergency_stop_distance: float = 0.9 # never continue forward below this, m
    detection_range: float = 8.0         # ignore returns beyond this, m
    max_turn_rate: float = 1.0           # rad/s
    avoidance_speed: float = 1.0         # m/s cap while avoiding
    replan_period: float = 0.1           # s
    commit_time: float = 0.6             # s a chosen side is held
    side_hysteresis_ratio: float = 1.25  # switch side only if clearly better
    min_valid_depth: float = 0.05

    def validate(self):
        if self.emergency_stop_distance >= self.min_safe_distance:
            raise ValueError("emergency_stop_distance must be < min_safe_distance")
        if self.detection_range <= self.min_safe_distance:
            raise ValueError("detection_range must be > min_safe_distance")
        return self


@dataclass
class AvoidanceResult:
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    wz: float = 0.0
    active: bool = False
    reason: str = "clear"
    sector_ranges: tuple = ()
    side: int = 0

    def as_tuple(self):
        return (self.vx, self.vy, self.vz, self.wz)


def _finite_positive(value, min_valid):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v < min_valid:
        return None
    return v


def sector_min_ranges(depth, num_sectors=7, vertical_fraction=0.5,
                      max_range=None, min_valid_depth=0.05):
    """Reduce a 2D depth image to per-sector minimum ranges.

    Only the middle vertical band is used: ground and sky pixels are noisy and
    irrelevant for horizontal collision avoidance. Columns are split evenly
    across sectors.
    """
    if not depth:
        return tuple()
    height = len(depth)
    width = len(depth[0]) if height else 0
    if height == 0 or width == 0:
        return tuple()
    rows = max(1, int(height * vertical_fraction))
    r0 = (height - rows) // 2
    r1 = r0 + rows
    num_sectors = max(1, int(num_sectors))
    ranges = []
    for s in range(num_sectors):
        c0 = int(width * s / num_sectors)
        c1 = max(c0 + 1, int(width * (s + 1) / num_sectors))
        best = None
        for r in range(r0, r1):
            row = depth[r]
            for c in range(c0, c1):
                value = _finite_positive(row[c], min_valid_depth)
                if value is None:
                    continue
                if max_range is not None and value > max_range:
                    continue
                if best is None or value < best:
                    best = value
        if best is not None:
            ranges.append(best)
    return tuple(ranges)


def sector_ranges_from_scan(angles, ranges, num_sectors=7, max_range=8.0,
                            min_valid_depth=0.05):
    """Reduce a 360-degree LaserScan to forward-facing sector minima.

    ``angles`` and ``ranges`` must be equal length. The forward direction is
    angle 0; sectors span -90..+90 degrees, left to right.
    """
    if len(angles) != len(ranges):
        raise ValueError("angles and ranges must have equal length")
    edges = [math.radians(-90.0 + 180.0 * i / num_sectors) for i in range(num_sectors + 1)]
    out = []
    for i in range(num_sectors):
        lo, hi = edges[i], edges[i + 1]
        best = None
        for angle, value in zip(angles, ranges):
            a = math.atan2(math.sin(angle), math.cos(angle))
            if not (lo <= a < hi):
                continue
            v = _finite_positive(value, min_valid_depth)
            if v is None or v > max_range:
                continue
            if best is None or v < best:
                best = v
        if best is not None:
            out.append(best)
    return tuple(out)


class DepthObstaclePlanner:
    """Committed-side local planner.

    Behavior:
      * clear path  -> passthrough of the requested forward speed, active=False
      * below safe  -> reduce speed proportionally and steer to the freer side
      * below stop  -> zero forward (-backup) and yaw to the freer side
    The chosen side is retained for ``commit_time`` unless the other side is
    better by ``side_hysteresis_ratio``, which prevents left/right oscillation.
    """

    def __init__(self, config=None):
        self.config = (config or AvoidanceConfig()).validate()
        self._side = 0
        self._side_since = None

    def reset(self):
        self._side = 0
        self._side_since = None

    @staticmethod
    def _forward_index(num_sectors):
        return num_sectors // 2

    def plan(self, sectors, now, forward_speed, max_speed=2.0, preferred_side=0):
        cfg = self.config
        sectors = tuple(sectors)
        if not sectors:
            return AvoidanceResult(
                vx=min(max(forward_speed, 0.0), cfg.avoidance_speed),
                active=False, reason="no_depth_data", sector_ranges=sectors)

        mid = self._forward_index(len(sectors))
        # Center sector plus its immediate neighbours define forward clearance.
        window = sectors[max(0, mid - 1):mid + 2]
        center_range = min(window) if window else cfg.detection_range

        if center_range > cfg.detection_range:
            return AvoidanceResult(vx=min(max(forward_speed, 0.0), max_speed),
                                   active=False, reason="clear",
                                   sector_ranges=sectors)

        # Select committed side with hysteresis.
        n = len(sectors)
        left = min(sectors[:mid]) if sectors[:mid] else 0.0
        right = min(sectors[mid + 1:]) if sectors[mid + 1:] else 0.0
        if self._side == 0:
            if preferred_side in (-1, 1):
                self._side, self._side_since = preferred_side, now
            else:
                self._side = -1 if left >= right else 1
                self._side_since = now
        else:
            other = -self._side
            current_clear = left if self._side == -1 else right
            other_clear = left if other == -1 else right
            expired = self._side_since is None or (now - self._side_since) >= cfg.commit_time
            if expired and other_clear > current_clear * cfg.side_hysteresis_ratio:
                self._side, self._side_since = other, now
        side = self._side
        steer = -side * cfg.max_turn_rate  # side -1 (left) -> +wz (CCW)

        if center_range <= cfg.emergency_stop_distance:
            return AvoidanceResult(vx=0.0, wz=steer, active=True,
                                   reason="emergency_stop_and_turn",
                                   sector_ranges=sectors, side=side)

        if center_range <= cfg.min_safe_distance:
            # Scale forward speed down to zero as we approach the stop distance.
            span = cfg.min_safe_distance - cfg.emergency_stop_distance
            scale = (center_range - cfg.emergency_stop_distance) / span
            vx = cfg.avoidance_speed * max(0.0, min(1.0, scale))
            vx = min(vx, max(forward_speed, 0.0))
            return AvoidanceResult(vx=vx, wz=steer, active=True,
                                   reason="slow_and_steer",
                                   sector_ranges=sectors, side=side)

        # Between safe and detection range: cautious approach, no hard steer
        # until inside the safe distance, but still report risk.
        vx = min(max(forward_speed, 0.0), cfg.avoidance_speed)
        return AvoidanceResult(vx=vx, active=True, reason="obstacle_ahead",
                               sector_ranges=sectors)

    def plan_depth(self, depth, now, forward_speed, max_speed=2.0,
                   preferred_side=0):
        sectors = sector_min_ranges(depth, max_range=self.config.detection_range,
                                    min_valid_depth=self.config.min_valid_depth)
        return self.plan(sectors, now, forward_speed, max_speed, preferred_side)
