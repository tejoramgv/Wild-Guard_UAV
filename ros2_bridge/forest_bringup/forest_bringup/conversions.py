"""NED <-> ENU conversions and camera geometry.

Pure Python on purpose: no rclpy, numpy or AirSim imports, so the whole
module is unit-testable on a headless Linux machine.

Conventions (documented in COORDINATE-TRANSFORMS.md):
  * AirSim world/vehicle pose is NED, metres, origin at the PlayerStart.
  * ROS 2 uses ENU (REP-103), metres.
  * AirSim IMU gyro/accel are already in the body (FRD) frame, so world-frame
    re-expression does NOT change them. Only orientation is converted.
  * Camera optical frame is x-right, y-down, z-forward (OpenCV/ROS optical).
"""

import math

# ---------------------------------------------------------------------------
# Points and vectors
# ---------------------------------------------------------------------------

def ned_to_enu_point(x, y, z):
    """NED coordinate -> ENU coordinate: (x_n, y_n, z_n) -> (y_n, x_n, -z_n)."""
    return (y, x, -z)


def enu_to_ned_point(x, y, z):
    """ENU coordinate -> NED coordinate. The mapping is its own inverse."""
    return (y, x, -z)


# A translation-free vector uses the same axis permutation as a point.
ned_to_enu_vec = ned_to_enu_point
enu_to_ned_vec = enu_to_ned_point


# ---------------------------------------------------------------------------
# Quaternions (w, x, y, z)
# ---------------------------------------------------------------------------

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)

# 180 deg rotation about (1, 1, 0)/sqrt(2): the NED->ENU axis swap.
_Q_NED_TO_ENU = (0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0)

# 180 deg about X: converts FRD body coordinates (AirSim) to FLU (ROS REP-103).
_Q_FRD_TO_FLU = (0.0, 1.0, 0.0, 0.0)


def quat_normalize(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        raise ValueError("cannot normalize zero quaternion")
    return (w / n, x / n, y / n, z / n)


def quat_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_mul(a, b):
    """Hamilton product a * b (rotations compose right-to-left)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_rotate(q, v):
    """Rotate vector v by unit quaternion q."""
    w, x, y, z = quat_normalize(q)
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def quat_from_euler(roll, pitch, yaw):
    """ZYX intrinsic (yaw-pitch-roll) Euler radians -> quaternion."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return quat_normalize((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ))


def quat_to_yaw(q):
    """Heading about +Z from a quaternion (radians, atan2 convention)."""
    w, x, y, z = quat_normalize(q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_to_rot_matrix(q):
    w, x, y, z = quat_normalize(q)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def ned_to_enu_quaternion(w, x, y, z):
    """Re-express a body->NED orientation as body->ENU.

    R_enu_body = R_ned_to_enu @ R_ned_body, hence q_enu = q_R * q_ned.
    The body frame itself is unchanged (it is FRD, not ENU), which is why no
    left/right conjugation is used.
    """
    return quat_normalize(quat_mul(_Q_NED_TO_ENU, quat_normalize((w, x, y, z))))


def enu_to_ned_quaternion(w, x, y, z):
    """Inverse of :func:`ned_to_enu_quaternion`."""
    return quat_normalize(quat_mul(quat_conjugate(_Q_NED_TO_ENU),
                                   quat_normalize((w, x, y, z))))


def frd_to_flu_vec(v):
    """Body vector FRD (x-forward, y-right, z-down) -> FLU (x-fwd, y-left, z-up)."""
    return (v[0], -v[1], -v[2])


def flu_to_frd_vec(v):
    """Inverse of :func:`frd_to_flu_vec`."""
    return (v[0], -v[1], -v[2])


def frd_to_flu_quaternion(w, x, y, z):
    """Re-express a body->world orientation from FRD body to FLU body."""
    return quat_normalize(quat_mul(quat_normalize((w, x, y, z)),
                                   _Q_FRD_TO_FLU))


def flu_to_frd_quaternion(w, x, y, z):
    """Inverse of :func:`frd_to_flu_quaternion`."""
    return quat_normalize(quat_mul(quat_normalize((w, x, y, z)),
                                   quat_conjugate(_Q_FRD_TO_FLU)))


def frd_ned_to_flu_enu_quaternion(w, x, y, z):
    """AirSim FRD-body->NED orientation -> ROS FLU-body->ENU orientation."""
    return frd_to_flu_quaternion(*ned_to_enu_quaternion(w, x, y, z))


def flu_enu_to_frd_ned_quaternion(w, x, y, z):
    """ROS FLU-body->ENU orientation -> AirSim FRD-body->NED orientation."""
    q_frd_enu = flu_to_frd_quaternion(w, x, y, z)
    return enu_to_ned_quaternion(*q_frd_enu)


# ---------------------------------------------------------------------------
# Yaw convenience helpers
# ---------------------------------------------------------------------------

def ned_yaw_to_enu(yaw_ned_deg):
    """AirSim NED yaw (degrees) -> ROS ENU yaw (radians)."""
    return math.radians(-yaw_ned_deg + 90.0)


def ned_yaw_deg_to_enu_deg(yaw_ned_deg):
    return (-yaw_ned_deg + 90.0) % 360.0


def enu_yaw_to_ned_deg(yaw_enu_rad):
    """ROS ENU yaw (radians) -> AirSim NED yaw (degrees, wrapped to [0, 360))."""
    return (-math.degrees(yaw_enu_rad) + 90.0) % 360.0


# ---------------------------------------------------------------------------
# Twist / IMU
# ---------------------------------------------------------------------------

def enu_cmd_to_ned(vx, vy, vz, wz):
    """ENU velocity command -> NED velocity command (vx, vy, vz, yaw-rate).

    Linear axes swap x/y and negate z; yaw rate negates because ENU yaw and
    NED yaw increase in opposite directions. AirSim's world-frame VelCmd
    consumes this directly (its wrapper converts the yaw rate to degrees).
    """
    return (vy, vx, -vz, -wz)


def ned_cmd_to_enu(vx, vy, vz, wz):
    return enu_cmd_to_ned(vx, vy, vz, wz)


def imu_ned_to_enu(linear_acceleration, angular_velocity, orientation_wxyz):
    """Convert AirSim IMU output to ROS 2 REP-103 fields.

    AirSim gyro/accel are in the body FRD frame. ROS body is FLU, so those
    vectors are converted with :func:`frd_to_flu_vec`. The orientation is
    re-expressed as FLU-body -> ENU.
    """
    return (
        frd_to_flu_vec(linear_acceleration),
        frd_to_flu_vec(angular_velocity),
        frd_ned_to_flu_enu_quaternion(*orientation_wxyz),
    )


# ---------------------------------------------------------------------------
# Camera geometry and 3D target localisation
# ---------------------------------------------------------------------------

def intrinsics(width, height, hfov_deg):
    """Pinhole intrinsics from image size and horizontal FOV (square pixels)."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx
    return {"fx": fx, "fy": fy, "cx": width / 2.0, "cy": height / 2.0,
            "width": width, "height": height}


def pixel_to_camera_ray(u, v, K):
    """Unit ray in camera optical coordinates for pixel (u, v)."""
    x = (u - K["cx"]) / K["fx"]
    y = (v - K["cy"]) / K["fy"]
    z = 1.0
    n = math.sqrt(x * x + y * y + z * z)
    return (x / n, y / n, z / n)


def pixel_depth_to_camera_point(u, v, depth, K, perspective=False):
    """Back-project a pixel plus depth into the camera optical frame.

    ``depth`` is the metric value sampled from the depth image.
    AirSim ``DepthPlanar`` is the distance from the camera plane (camera z),
    while ``DepthPerspective`` is the distance along the optical ray. Set
    ``perspective=True`` for the latter.
    """
    x = (u - K["cx"]) * depth / K["fx"]
    y = (v - K["cy"]) * depth / K["fy"]
    if perspective:
        norm = math.sqrt(x * x + y * y + depth * depth)
        if norm == 0.0:
            return (0.0, 0.0, 0.0)
        scale = depth / norm
        return (x * scale, y * scale, depth * scale)
    return (x, y, depth)


def _mat_mul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _mat_vec(m, v):
    return tuple(sum(m[i][k] * v[k] for k in range(3)) for i in range(3))


def camera_mount_rotation(pitch_down_deg):
    """Body(FRD) <- camera-optical rotation for a forward camera pitched down.

    Optical (x-right, y-down, z-forward) -> body (x-forward, y-right, z-down)
    is P = [[0,0,1],[1,0,0],[0,1,0]]; the mount pitch then rotates about the
    body +Y axis. ``pitch_down_deg`` is positive when the camera looks down;
    for AirSim's ``Pitch`` setting use ``pitch_down_deg = -Pitch``.
    """
    c, s = math.cos(math.radians(pitch_down_deg)), math.sin(math.radians(pitch_down_deg))
    ry = ((c, 0.0, -s), (0.0, 1.0, 0.0), (s, 0.0, c))
    p = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    return _mat_mul(ry, p)


def camera_point_to_world_ned(p_cam, drone_position_ned, drone_orientation_wxyz,
                              pitch_down_deg=11.5, mount_offset_ned=(0.0, 0.0, 0.0)):
    """Camera optical point -> world point in NED."""
    p_body = _mat_vec(camera_mount_rotation(pitch_down_deg), p_cam)
    p_body_world = quat_rotate(drone_orientation_wxyz, p_body)
    return (p_body_world[0] + drone_position_ned[0] + mount_offset_ned[0],
            p_body_world[1] + drone_position_ned[1] + mount_offset_ned[1],
            p_body_world[2] + drone_position_ned[2] + mount_offset_ned[2])


def camera_point_to_world_enu(p_cam, drone_position_ned, drone_orientation_wxyz,
                              pitch_down_deg=11.5, mount_offset_ned=(0.0, 0.0, 0.0)):
    return ned_to_enu_point(*camera_point_to_world_ned(
        p_cam, drone_position_ned, drone_orientation_wxyz,
        pitch_down_deg, mount_offset_ned))


def ray_ground_intersection(origin_enu, direction_enu, ground_z=0.0):
    """Intersect a world ray with a horizontal plane z = ground_z.

    Returns the ENU intersection point, or ``None`` if the ray is parallel to
    the plane or points away from it. Used only as a documented fallback when
    depth is invalid; it is never the primary method.
    """
    dz = direction_enu[2]
    if abs(dz) < 1e-9:
        return None
    t = (ground_z - origin_enu[2]) / dz
    if t <= 0.0:
        return None
    return (origin_enu[0] + t * direction_enu[0],
            origin_enu[1] + t * direction_enu[1],
            ground_z)


def camera_ray_to_world_enu(u, v, K, drone_position_ned, drone_orientation_wxyz,
                            pitch_down_deg=11.5):
    """World-space ENU ray for pixel (u, v) (unit direction)."""
    ray_cam = pixel_to_camera_ray(u, v, K)
    origin = camera_point_to_world_enu((0.0, 0.0, 0.0), drone_position_ned,
                                       drone_orientation_wxyz, pitch_down_deg)
    tip = camera_point_to_world_enu(ray_cam, drone_position_ned,
                                    drone_orientation_wxyz, pitch_down_deg)
    direction = (tip[0] - origin[0], tip[1] - origin[1], tip[2] - origin[2])
    n = math.sqrt(sum(c * c for c in direction))
    if n == 0.0:
        return origin, (0.0, 0.0, 0.0)
    return origin, (direction[0] / n, direction[1] / n, direction[2] / n)
