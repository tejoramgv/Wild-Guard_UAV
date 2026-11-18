"""Tests for heading-relative to world-frame teleop conversion."""

import math

from forest_bringup.teleop import body_to_world, project_forward, world_to_body


def test_forward_at_cardinal_headings():
    assert body_to_world(1.0, 0.0, 0.0) == (1.0, 0.0)
    x, y = body_to_world(1.0, 0.0, math.pi / 2)
    assert math.isclose(x, 0.0, abs_tol=1e-9) and math.isclose(y, 1.0)
    x, y = body_to_world(1.0, 0.0, math.pi)
    assert math.isclose(x, -1.0) and math.isclose(y, 0.0, abs_tol=1e-9)


def test_right_is_minus_ninety_degrees():
    # Facing east (+X), right is south (-Y).
    assert body_to_world(0.0, 1.0, 0.0) == (0.0, -1.0)
    # Facing north (+Y), right is east (+X).
    x, y = body_to_world(0.0, 1.0, math.pi / 2)
    assert math.isclose(x, 1.0) and math.isclose(y, 0.0, abs_tol=1e-9)


def test_roundtrip_and_projection():
    yaw = 0.7
    world = body_to_world(1.5, -0.4, yaw)
    fwd, right = world_to_body(*world, yaw)
    assert math.isclose(fwd, 1.5) and math.isclose(right, -0.4)
    assert math.isclose(project_forward(*world, yaw), 1.5)
    assert math.isclose(project_forward(1.0, 1.0, 0.0), 1.0)
