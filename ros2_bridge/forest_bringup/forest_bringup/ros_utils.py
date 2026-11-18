"""ROS-facing helpers shared by the ForestAirSim nodes.

Importing this module requires rclpy/message packages; the offline core modules
(conversions, safety, avoidance, ...) deliberately do not import it.
"""

import math

import numpy as np


def stamp_to_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quat_msg_to_tuple(q):
    return (q.w, q.x, q.y, q.z)


def tuple_to_quat_msg(q, msg):
    msg.x, msg.y, msg.z, msg.w = q[1], q[2], q[3], q[0]
    return msg


def image_msg_to_array(msg):
    """Decode a sensor_msgs/Image into a numpy array without cv_bridge.

    Supports the encodings this project publishes: rgb8/bgr8, rgba8/bgra8,
    mono8/mono16, 32FC1 and 16UC1. Rows are un-padded using ``step``.
    """
    encoding = (msg.encoding or "").lower()
    if encoding in ("rgb8", "bgr8"):
        dtype, channels = np.uint8, 3
    elif encoding in ("rgba8", "bgra8"):
        dtype, channels = np.uint8, 4
    elif encoding == "mono8":
        dtype, channels = np.uint8, 1
    elif encoding == "mono16":
        dtype, channels = np.uint16, 1
    elif encoding == "32fc1":
        dtype, channels = np.float32, 1
    elif encoding == "16uc1":
        dtype, channels = np.uint16, 1
    else:
        raise ValueError(f"unsupported image encoding: {msg.encoding!r}")

    row_bytes = int(msg.width) * channels * np.dtype(dtype).itemsize
    step = int(msg.step) if int(msg.step) >= row_bytes else row_bytes
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    rows = raw[:step * int(msg.height)].reshape((int(msg.height), step))
    rows = rows[:, :row_bytes]
    flat = rows.reshape(-1).view(dtype)
    if channels == 1:
        return flat.reshape((int(msg.height), int(msg.width)))
    return flat.reshape((int(msg.height), int(msg.width), channels))


def array_to_gray3(array):
    """Convert RGB/BGR/gray/float image data to an 8-bit 3-channel array."""
    a = np.asarray(array)
    if a.ndim == 2:
        if a.dtype != np.uint8:
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
            if hi > lo:
                a = (a - lo) / (hi - lo) * 255.0
            else:
                a = np.zeros_like(a, dtype=np.float32)
        a = a.astype(np.uint8)
        return np.repeat(a[:, :, None], 3, axis=2)
    if a.shape[2] == 4:
        a = a[:, :, :3]
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def brightness_stats(image):
    a = np.asarray(image, dtype=np.float64)
    if a.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def yaw_from_quat_msg(q):
    w, x, y, z = quat_msg_to_tuple(q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
