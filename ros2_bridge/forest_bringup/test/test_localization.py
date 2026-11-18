"""3D localisation tests (depth primary, ground-plane fallback)."""

import math

from forest_bringup.conversions import (
    camera_point_to_world_enu, intrinsics, pixel_depth_to_camera_point,
    quat_from_euler)
from forest_bringup.detection_logic import Detection2D
from forest_bringup.localization import (
    localize_detection, localize_detections, sample_box_depth)


def _det(cx=320, cy=300, w=40, h=60):
    return Detection2D("deer", 0.8, cx, cy, w, h, 1.0, "rgb")


def _depth_image(value=10.0, width=640, height=480):
    return [[value] * width for _ in range(height)]


def test_sample_box_depth_ignores_zeros_and_nan():
    img = _depth_image(0.0, width=20, height=20)
    det = _det(cx=10, cy=10, w=4, h=4)
    assert sample_box_depth(img, det) is None
    for r in range(9, 12):
        for c in range(9, 12):
            img[r][c] = 7.5
    assert sample_box_depth(img, det) == 7.5


def test_localize_depth_primary():
    img = _depth_image(10.0)
    K = intrinsics(640, 480, 60.0)
    det = _det()
    out = localize_detection(det, img, K, (0, 0, -5),
                             quat_from_euler(0, 0, 0), pitch_down_deg=11.5)
    expected = camera_point_to_world_enu(
        pixel_depth_to_camera_point(det.cx, det.cy, 10.0, K), (0, 0, -5),
        quat_from_euler(0, 0, 0), 11.5)
    assert out is not None and out["method"] == "depth"
    assert all(math.isclose(a, b, abs_tol=1e-6)
               for a, b in zip((out["x"], out["y"], out["z"]), expected))


def test_localize_falls_back_to_ground_plane():
    img = _depth_image(0.0)  # invalid depth everywhere
    K = intrinsics(640, 480, 60.0)
    det = _det()
    out = localize_detection(det, img, K, (0, 0, -5),
                             quat_from_euler(0, 0, 0),
                             pitch_down_deg=11.5, ground_z=0.0)
    assert out is not None and out["method"] == "ground_plane"
    assert math.isclose(out["z"], 0.0)
    assert out["y"] > 0.0


def test_localize_respects_fallback_switch():
    img = _depth_image(0.0)
    K = intrinsics(640, 480, 60.0)
    det = _det()
    out = localize_detection(det, img, K, (0, 0, -5),
                             quat_from_euler(0, 0, 0),
                             pitch_down_deg=11.5,
                             allow_ground_fallback=False)
    assert out is None


def test_localize_batch_drops_failures():
    img = _depth_image(0.0)
    K = intrinsics(640, 480, 60.0)
    dets = [_det(), _det(cx=700, cy=300)]  # second box is outside the image
    outlined = localize_detections(dets, img, K, (0, 0, -5),
                                   quat_from_euler(0, 0, 0),
                                   pitch_down_deg=11.5,
                                   allow_ground_fallback=False)
    assert outlined == []
