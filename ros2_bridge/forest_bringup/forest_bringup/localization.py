"""3D target localisation from bounding boxes, depth and drone pose.

A 2D box alone cannot give a world coordinate. The primary method samples the
metric depth image inside the box; the documented fallback intersects the
pixel ray with a ground plane. Both are deterministic and offline-testable.
"""

import math

from forest_bringup.conversions import (
    camera_point_to_world_enu, camera_ray_to_world_enu, intrinsics,
    pixel_depth_to_camera_point, ray_ground_intersection)


def _sample(image, u, v):
    height = len(image)
    if height == 0:
        return None
    width = len(image[0])
    if width == 0:
        return None
    r = int(round(v))
    c = int(round(u))
    if r < 0 or r >= height or c < 0 or c >= width:
        return None
    try:
        value = float(image[r][c])
    except (TypeError, ValueError):
        return None
    return value


def _median(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return None
    mid = n // 2
    return values[mid] if n % 2 else 0.5 * (values[mid - 1] + values[mid])


def sample_box_depth(image, det, patch=5, min_depth=0.05, max_depth=120.0):
    """Robust depth for a detection box.

    Samples a small patch at the box bottom-center (feet/ground contact for
    animals and people) and falls back to the box center. Invalid values
    (0, negative, NaN, infinity, beyond ``max_depth``) are ignored.
    """
    if image is None or len(image) == 0:
        return None
    height, width = len(image), len(image[0]) if len(image[0]) is not None else 0
    if width == 0:
        return None
    candidates = [
        (det.cx, det.cy + det.h * 0.35),  # lower part of the box
        (det.cx, det.cy),                 # center
        (det.cx, det.cy - det.h * 0.2),   # upper body
    ]
    valid = []
    half = max(1, int(patch) // 2)
    for u, v in candidates:
        for du in range(-half, half + 1):
            for dv in range(-half, half + 1):
                value = _sample(image, u + du, v + dv)
                if value is None or not math.isfinite(value):
                    continue
                if value < min_depth or value > max_depth:
                    continue
                valid.append(value)
        if len(valid) >= patch * patch:
            break
    return _median(valid)


def localize_detection(det, depth_image, K, drone_position_ned,
                       drone_orientation_wxyz, pitch_down_deg=11.5,
                       ground_z=0.0, allow_ground_fallback=True,
                       perspective=False, min_depth=0.05, max_depth=120.0):
    """Localise one detection. Returns dict or ``None`` when impossible.

    Result keys: ``x``, ``y``, ``z`` (ENU metres), ``method``
    ("depth" or "ground_plane") and ``pixel``.
    """
    u, v = det.cx, det.cy
    depth = sample_box_depth(depth_image, det, min_depth=min_depth,
                             max_depth=max_depth)
    if depth is not None:
        point_cam = pixel_depth_to_camera_point(u, v, depth, K,
                                                perspective=perspective)
        x, y, z = camera_point_to_world_enu(
            point_cam, drone_position_ned, drone_orientation_wxyz,
            pitch_down_deg)
        return {"x": x, "y": y, "z": z, "method": "depth",
                "pixel": (u, v), "depth": depth}

    if not allow_ground_fallback:
        return None
    origin, direction = camera_ray_to_world_enu(
        u, v, K, drone_position_ned, drone_orientation_wxyz, pitch_down_deg)
    hit = ray_ground_intersection(origin, direction, ground_z=ground_z)
    if hit is None:
        return None
    return {"x": hit[0], "y": hit[1], "z": hit[2], "method": "ground_plane",
            "pixel": (u, v), "depth": None}


def localize_detections(detections, depth_image, K, drone_position_ned,
                        drone_orientation_wxyz, **kwargs):
    """Localise a sequence, dropping entries that cannot be resolved."""
    out = []
    for det in detections:
        result = localize_detection(det, depth_image, K, drone_position_ned,
                                    drone_orientation_wxyz, **kwargs)
        if result is not None:
            out.append((det, result))
    return out


def make_intrinsics(width, height, hfov_deg):
    return intrinsics(width, height, hfov_deg)
