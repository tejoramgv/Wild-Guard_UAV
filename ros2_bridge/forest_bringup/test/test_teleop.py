"""Keyboard teleoperation mapping tests."""

import math

from forest_bringup.teleop import (
    ESTOP, HOVER, LAND, TAKEOFF, TeleopConfig, TeleopState)


def test_forward_continuous_until_release():
    s = TeleopState(TeleopConfig(max_speed=2.0))
    s.press("up")
    cmd = s.compute()
    assert cmd.vx == 2.0
    # Still pressed on the next cycle -> still moving.
    assert s.compute().vx == 2.0
    s.release("up")
    assert s.compute().is_zero()


def test_arrow_directions_and_strafe():
    s = TeleopState(TeleopConfig(max_speed=2.0))
    s.press("down")
    assert s.compute().vx == -2.0
    s.release("down")
    s.press("left")
    assert s.compute().vy == -2.0
    s.release("left")
    s.press("right")
    assert s.compute().vy == 2.0


def test_vertical_and_yaw():
    s = TeleopState(TeleopConfig(max_vertical_speed=1.5, max_yaw_rate=1.2))
    s.press("w")
    assert s.compute().vz == 1.5
    s.release("w")
    s.press("s")
    assert s.compute().vz == -1.5
    s.release("s")
    s.press("a")
    assert s.compute().wz == 1.2
    s.release("a")
    s.press("d")
    assert s.compute().wz == -1.2


def test_diagonal_normalised():
    s = TeleopState(TeleopConfig(max_speed=2.0))
    s.press("up")
    s.press("right")
    cmd = s.compute()
    assert math.isclose(math.hypot(cmd.vx, cmd.vy), 2.0)


def test_hover_stops_and_clears():
    s = TeleopState()
    s.press("up")
    s.press("space")
    assert s.compute().is_zero()
    assert s.all_released()


def test_discrete_events():
    s = TeleopState()
    s.press("t")
    s.press("l")
    s.press("esc")
    assert s.pop_events() == [TAKEOFF, LAND, ESTOP]
    assert s.pop_events() == []


def test_estop_clears_held_keys():
    s = TeleopState()
    s.press("up")
    s.press("esc")
    assert s.compute().is_zero()


def test_unknown_key_ignored():
    s = TeleopState()
    s.press("z")
    assert s.compute().is_zero()
