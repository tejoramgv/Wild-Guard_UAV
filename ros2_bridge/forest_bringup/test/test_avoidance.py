"""Obstacle detection and avoidance planner tests."""

import math

from forest_bringup.avoidance import (
    AvoidanceConfig, DepthObstaclePlanner, sector_min_ranges,
    sector_ranges_from_scan)


def _depth(center=10.0, left=10.0, right=10.0, rows=5, cols=9):
    img = []
    for _ in range(rows):
        row = [10.0] * cols
        row[cols // 2] = center
        row[1] = left
        row[cols - 2] = right
        img.append(row)
    return img


def test_sector_ranges_pick_center_min():
    img = _depth(center=1.0)
    sectors = sector_min_ranges(img, num_sectors=7, vertical_fraction=1.0)
    assert len(sectors) == 7
    assert min(sectors) == 1.0


def test_sector_scan():
    angles = [math.radians(a) for a in range(-90, 91, 10)]
    ranges = [5.0] * len(angles)
    ranges[len(ranges) // 2] = 1.2
    sectors = sector_ranges_from_scan(angles, ranges, num_sectors=7, max_range=8.0)
    assert min(sectors) == 1.2


def test_clear_path_passthrough():
    planner = DepthObstaclePlanner(AvoidanceConfig())
    result = planner.plan((10.0,) * 7, now=0.0, forward_speed=1.5)
    assert result.active is False and result.reason == "clear"
    assert result.vx == 1.5


def test_slow_and_steer():
    planner = DepthObstaclePlanner(AvoidanceConfig())
    result = planner.plan((8.0, 8.0, 1.5, 8.0, 8.0), now=0.0, forward_speed=1.5)
    assert result.active is True
    assert result.reason == "slow_and_steer"
    assert 0.0 < result.vx < 1.5
    assert abs(result.wz) > 0.0


def test_emergency_stop_and_turn():
    planner = DepthObstaclePlanner(AvoidanceConfig())
    result = planner.plan((8.0, 8.0, 0.5, 8.0, 8.0), now=0.0, forward_speed=1.5)
    assert result.vx == 0.0
    assert result.reason == "emergency_stop_and_turn"
    assert result.wz != 0.0


def test_side_choice_and_hysteresis():
    cfg = AvoidanceConfig(commit_time=1.0)
    planner = DepthObstaclePlanner(cfg)
    # Left blocked, right free -> steer right (negative yaw rate).
    first = planner.plan((1.0, 1.0, 1.5, 8.0, 8.0), now=0.0, forward_speed=1.0)
    assert first.side == 1 and first.wz < 0.0
    # Shortly after, the other side looks better, but we hold the commitment.
    held = planner.plan((8.0, 8.0, 1.5, 1.0, 1.0), now=0.2, forward_speed=1.0)
    assert held.side == 1
    # After commit_time expires, switching is allowed.
    switched = planner.plan((8.0, 8.0, 1.5, 1.0, 1.0), now=1.5, forward_speed=1.0)
    assert switched.side == -1


def test_no_depth_data():
    planner = DepthObstaclePlanner()
    result = planner.plan((), now=0.0, forward_speed=1.0)
    assert result.active is False and result.reason == "no_depth_data"


def test_config_validation():
    try:
        AvoidanceConfig(emergency_stop_distance=3.0, min_safe_distance=2.0).validate()
    except ValueError as exc:
        assert "emergency_stop_distance" in str(exc)
    else:
        raise AssertionError("expected ValueError")
