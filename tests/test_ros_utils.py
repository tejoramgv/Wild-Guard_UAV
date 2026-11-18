"""Image decoding/stats tests using duck-typed messages (no rclpy needed)."""

from types import SimpleNamespace

import numpy as np

from forest_bringup.ros_utils import (
    array_to_gray3, brightness_stats, image_msg_to_array, stamp_to_seconds)


def _msg(data, encoding, width, height, step=None):
    return SimpleNamespace(data=data, encoding=encoding, width=width,
                           height=height, step=step if step is not None else
                           width * (3 if encoding in ("rgb8", "bgr8") else 1))


def test_decode_rgb8_with_row_padding():
    pixels = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    padded = np.zeros((1, 8), dtype=np.uint8)  # 6 pixel bytes + 2 pad
    padded[0, :6] = pixels.reshape(-1)
    arr = image_msg_to_array(_msg(padded.tobytes(), "rgb8", 2, 1, step=8))
    assert arr.shape == (1, 2, 3)
    assert arr[0, 1].tolist() == [4, 5, 6]


def test_decode_mono8_and_32fc1():
    mono = image_msg_to_array(_msg(bytes([10, 20, 30, 40]), "mono8", 2, 2))
    assert mono.shape == (2, 2) and mono[1, 1] == 40
    depth = np.array([[1.5, 2.5]], dtype=np.float32)
    got = image_msg_to_array(_msg(depth.tobytes(), "32FC1", 2, 1))
    assert got.dtype == np.float32 and got[0, 0] == 1.5


def test_unsupported_encoding():
    try:
        image_msg_to_array(_msg(b"", "yuv422", 0, 0))
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_array_to_gray3_and_stats():
    gray = np.array([[0, 100], [200, 255]], dtype=np.uint8)
    rgb = array_to_gray3(gray)
    assert rgb.shape == (2, 2, 3)
    stats = brightness_stats(gray)
    assert stats["max"] == 255 and stats["min"] == 0
    assert brightness_stats(np.array([]))["mean"] == 0.0


def test_stamp_to_seconds():
    stamp = SimpleNamespace(sec=10, nanosec=500_000_000)
    assert stamp_to_seconds(stamp) == 10.5
