"""Conversion and camera-geometry tests (no ROS, no sim)."""

import math

from forest_bringup.conversions import (
    camera_point_to_world_enu, camera_ray_to_world_enu, enu_cmd_to_ned,
    enu_to_ned_quaternion, enu_to_ned_point, enu_yaw_to_ned_deg,
    flu_enu_to_frd_ned_quaternion, frd_ned_to_flu_enu_quaternion,
    frd_to_flu_quaternion, frd_to_flu_vec, imu_ned_to_enu, intrinsics,
    ned_to_enu_point, ned_to_enu_quaternion, ned_to_enu_vec,
    ned_yaw_deg_to_enu_deg, ned_yaw_to_enu, pixel_depth_to_camera_point,
    pixel_to_camera_ray, quat_from_euler, quat_mul, quat_normalize,
    quat_rotate, quat_to_yaw, ray_ground_intersection)
import forest_bringup.conversions as conv


def test_ned_enu_point_and_vec():
    assert ned_to_enu_point(1, 0, 0) == (0, 1, 0)
    assert ned_to_enu_point(0, 1, 0) == (1, 0, 0)
    assert ned_to_enu_point(0, 0, 1) == (0, 0, -1)
    assert ned_to_enu_vec(2, 0, -1) == (0, 2, 1)
    # inverse
    for p in [(3, -2, 5), (0, 0, 0), (-1.5, 4, -0.25)]:
        assert enu_to_ned_point(*ned_to_enu_point(*p)) == p


def test_ned_enu_axis_rotation_quaternion():
    q = conv._Q_NED_TO_ENU
    for src, dst in [((1, 0, 0), (0, 1, 0)),
                     ((0, 1, 0), (1, 0, 0)),
                     ((0, 0, 1), (0, 0, -1))]:
        got = quat_rotate(q, src)
        assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(got, dst))


def test_yaw_conversion():
    assert math.isclose(ned_yaw_to_enu(0.0), math.radians(90.0))
    assert math.isclose(ned_yaw_to_enu(90.0), 0.0)
    assert math.isclose(ned_yaw_deg_to_enu_deg(180.0), 270.0)
    assert math.isclose(enu_yaw_to_ned_deg(math.radians(0.0)), 90.0)


def test_orientation_quaternion_roundtrip_and_yaw():
    q_ned = quat_from_euler(0.12, -0.08, math.radians(35.0))
    q_enu = ned_to_enu_quaternion(*q_ned)
    back = enu_to_ned_quaternion(*q_enu)
    dot = sum(a * b for a, b in zip(q_ned, back))
    assert math.isclose(abs(dot), 1.0, abs_tol=1e-9)
    assert math.isclose(math.degrees(quat_to_yaw(q_enu)),
                        ned_yaw_deg_to_enu_deg(35.0), abs_tol=1e-6)


def test_quaternion_helpers():
    assert quat_mul((1, 0, 0, 0), (1, 0, 0, 0)) == (1, 0, 0, 0)
    q = quat_from_euler(0, 0, math.pi / 2)
    got = quat_rotate(q, (1, 0, 0))
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(got, (0, 1, 0)))
    assert math.isclose(quat_normalize((2, 0, 0, 0))[0], 1.0)


def test_cmd_conversion():
    assert enu_cmd_to_ned(1, 0, 0, 0.5) == (0, 1, 0, -0.5)
    assert enu_cmd_to_ned(0, 2, -1, -0.5) == (2, 0, 1, 0.5)


def test_imu_body_vectors_are_frd_to_flu():
    accel, gyro, q = (0.1, -0.2, 9.81), (0.01, 0.02, 0.03), quat_from_euler(0, 0, 0.5)
    a, g, qe = imu_ned_to_enu(accel, gyro, q)
    assert all(math.isclose(x, y) for x, y in zip(a, (0.1, 0.2, -9.81)))
    assert all(math.isclose(x, y) for x, y in zip(g, (0.01, -0.02, -0.03)))
    assert qe != q


def test_frd_flu_roundtrip():
    assert frd_to_flu_vec((1.0, 2.0, 3.0)) == (1.0, -2.0, -3.0)
    q = quat_from_euler(0.1, -0.2, 0.3)
    roundtrip = flu_enu_to_frd_ned_quaternion(
        *frd_ned_to_flu_enu_quaternion(*q))
    assert math.isclose(abs(sum(a * b for a, b in zip(q, roundtrip))), 1.0,
                        abs_tol=1e-9)


def test_level_drone_flu_enu_is_identity_for_east_heading():
    # AirSim FRD body facing east => NED yaw 90. In ROS FLU body that is the
    # identity orientation (x forward east, y left north, z up).
    q_ned = quat_from_euler(0, 0, math.pi / 2)
    q_flu_enu = frd_ned_to_flu_enu_quaternion(*q_ned)
    assert math.isclose(abs(q_flu_enu[0]), 1.0, abs_tol=1e-9)
    derived = frd_to_flu_quaternion(*ned_to_enu_quaternion(*q_ned))
    assert math.isclose(abs(sum(a * b for a, b in zip(q_flu_enu, derived))), 1.0,
                        abs_tol=1e-9)


def test_intrinsics_and_center_ray():
    K = intrinsics(640, 480, 60.0)
    assert math.isclose(K["fx"], 554.256258422, rel_tol=1e-9)
    assert K["cx"] == 320 and K["cy"] == 240
    assert pixel_to_camera_ray(320, 240, K) == (0.0, 0.0, 1.0)


def test_camera_point_to_world_ned_enu():
    # 5 m altitude, facing north (NED yaw 0), 10 m along the optical axis.
    K = intrinsics(640, 480, 60.0)
    world = camera_point_to_world_enu((0, 0, 10), (0, 0, -5),
                                      quat_from_euler(0, 0, 0), 11.5)
    assert math.isclose(world[0], 0.0, abs_tol=1e-6)
    assert math.isclose(world[1], 10 * math.cos(math.radians(11.5)), abs_tol=1e-6)
    assert math.isclose(world[2], 5 - 10 * math.sin(math.radians(11.5)), abs_tol=1e-6)
    # A pixel to the right of center must map to +X ENU when facing north.
    p = pixel_depth_to_camera_point(400, 240, 10.0, K)
    right = camera_point_to_world_enu(p, (0, 0, -5), quat_from_euler(0, 0, 0), 11.5)
    assert right[0] > 0.0


def test_depth_perspective_conversion():
    K = intrinsics(640, 480, 60.0)
    planar = pixel_depth_to_camera_point(400, 300, 5.0, K)
    perspective = pixel_depth_to_camera_point(400, 300, 5.0, K, perspective=True)
    assert math.isclose(math.sqrt(sum(v * v for v in perspective)), 5.0, abs_tol=1e-9)
    assert perspective[2] < 5.0 and planar[2] == 5.0


def test_ground_plane_intersection():
    origin, direction = camera_ray_to_world_enu(
        320, 240, intrinsics(640, 480, 60.0), (0, 0, -5),
        quat_from_euler(0, 0, 0), 11.5)
    hit = ray_ground_intersection(origin, direction, 0.0)
    assert hit is not None
    assert math.isclose(hit[2], 0.0)
    assert math.isclose(hit[1], 5.0 / math.tan(math.radians(11.5)), rel_tol=1e-6)
    assert ray_ground_intersection((0, 0, 5), (0, 0, 1), 0.0) is None
    assert ray_ground_intersection((0, 0, 5), (1, 0, 0), 0.0) is None
