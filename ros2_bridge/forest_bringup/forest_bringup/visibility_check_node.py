"""Image-quality diagnostics for the RGB and thermal streams.

Beyond brightness, this node checks that the thermal stream is not merely a
copy of the RGB stream. A thermal image that is pixel-identical to RGB is a
configuration bug, not a thermal camera.
"""

import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from forest_bringup.ros_utils import (
    brightness_stats, image_msg_to_array, stamp_to_seconds)


class VisibilityCheck(Node):
    def __init__(self):
        super().__init__("visibility_check")
        p = self.declare_parameter
        p("rgb_topic", "/drone/camera/rgb/image_raw")
        p("thermal_topic", "/drone/camera/thermal/image_raw")
        p("min_mean", 15.0)
        p("expected_width", 640)
        p("expected_height", 480)
        p("report_period", 3.0)
        p("duplicate_tolerance", 1.0)

        g = lambda n: self.get_parameter(n).value
        self.min_mean = float(g("min_mean"))
        self.exp_w, self.exp_h = int(g("expected_width")), int(g("expected_height"))
        self.report_period = float(g("report_period"))
        self.dup_tol = float(g("duplicate_tolerance"))
        self.frames = {"rgb": 0, "thermal": 0}
        self.last = {}
        self.last_report = {"rgb": 0.0, "thermal": 0.0}
        self.status_pub = self.create_publisher(String, "/drone/visibility/status", 10)
        self.create_subscription(Image, g("rgb_topic"),
                                 lambda m: self.on_image("rgb", m),
                                 qos_profile_sensor_data)
        self.create_subscription(Image, g("thermal_topic"),
                                 lambda m: self.on_image("thermal", m),
                                 qos_profile_sensor_data)
        self.get_logger().info("visibility_check ready")

    def _downsample(self, arr):
        a = np.asarray(arr)
        if a.ndim == 3:
            a = a.astype(np.float64).mean(axis=2)
        else:
            a = a.astype(np.float64)
        h, w = a.shape
        return a[::max(1, h // 48), ::max(1, w // 64)]

    def on_image(self, source, msg):
        try:
            arr = image_msg_to_array(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)
            return
        self.frames[source] += 1
        stats = brightness_stats(arr)
        now = stamp_to_seconds(msg.header.stamp)
        small = self._downsample(arr)
        duplicate = False
        other = "thermal" if source == "rgb" else "rgb"
        if other in self.last:
            prev = self.last[other]
            if prev.shape == small.shape:
                duplicate = float(np.mean(np.abs(prev - small))) <= self.dup_tol
        self.last[source] = small

        size_ok = msg.width == self.exp_w and msg.height == self.exp_h
        bright_ok = stats["mean"] >= self.min_mean
        record = {
            "source": source,
            "frame": self.frames[source],
            "width": msg.width,
            "height": msg.height,
            "encoding": msg.encoding,
            "mean": round(stats["mean"], 2),
            "std": round(stats["std"], 2),
            "size_ok": size_ok,
            "bright_ok": bright_ok,
            "duplicate_of_" + other: duplicate,
        }
        if duplicate:
            self.get_logger().error(
                f"{source} stream is pixel-identical to {other}; "
                "the thermal camera is probably configured as Scene/RGB")
        self.status_pub.publish(String(data=json.dumps(record)))
        if self.frames[source] == 1 or \
                (now - self.last_report[source]) >= self.report_period:
            self.last_report[source] = now
            self.get_logger().info(json.dumps(record))


def main():
    rclpy.init()
    node = VisibilityCheck()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
