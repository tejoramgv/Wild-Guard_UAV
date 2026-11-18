"""Depth-based local obstacle avoidance.

Inputs: depth image (primary) or LaserScan (optional), camera info, manual
velocity and odometry. Output: the final arbitrated /drone/cmd_vel. When an
obstacle is inside the safe envelope this node overrides the manual command,
which is what gives collision avoidance priority over teleoperation.
"""

import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String

from forest_bringup.avoidance import (
    AvoidanceConfig, DepthObstaclePlanner, sector_min_ranges,
    sector_ranges_from_scan)
from forest_bringup.ros_utils import image_msg_to_array
from forest_bringup.teleop import body_to_world, project_forward


class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__("obstacle_avoidance")
        p = self.declare_parameter
        p("min_safe_distance", 2.0)
        p("emergency_stop_distance", 0.9)
        p("detection_range", 8.0)
        p("max_turn_rate", 1.0)
        p("avoidance_speed", 1.0)
        p("replan_period", 0.1)
        p("commit_time", 0.6)
        p("side_hysteresis_ratio", 1.25)
        p("num_sectors", 7)
        p("vertical_fraction", 0.5)
        p("min_valid_depth", 0.05)
        p("use_lidar", False)
        p("lidar_topic", "/airsim_node/ForestDrone/lidar/LidarSensor")
        p("depth_timeout", 0.5)
        p("max_speed", 2.0)

        g = lambda n: self.get_parameter(n).value
        self.cfg = AvoidanceConfig(
            min_safe_distance=g("min_safe_distance"),
            emergency_stop_distance=g("emergency_stop_distance"),
            detection_range=g("detection_range"),
            max_turn_rate=g("max_turn_rate"),
            avoidance_speed=g("avoidance_speed"),
            replan_period=g("replan_period"),
            commit_time=g("commit_time"),
            side_hysteresis_ratio=g("side_hysteresis_ratio"),
            min_valid_depth=g("min_valid_depth")).validate()
        self.num_sectors = int(g("num_sectors"))
        self.vertical_fraction = float(g("vertical_fraction"))
        self.max_speed = float(g("max_speed"))
        self.depth_timeout = float(g("depth_timeout"))
        self.use_lidar = bool(g("use_lidar"))
        self.planner = DepthObstaclePlanner(self.cfg)

        self.depth = None
        self.depth_time = None
        self.scan = None
        self.scan_time = None
        self.manual = Twist()
        self.yaw = 0.0

        self.pub = self.create_publisher(Twist, "/drone/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, "/drone/avoidance/status", 10)
        self.create_subscription(Image, "/drone/camera/depth/image_raw",
                                 self.on_depth, qos_profile_sensor_data)
        self.create_subscription(Twist, "/drone/cmd_vel_manual", self.on_manual, 10)
        self.create_subscription(Odometry, "/drone/odom", self.on_odom, 10)
        if self.use_lidar:
            self.create_subscription(LaserScan, g("lidar_topic"), self.on_scan,
                                     qos_profile_sensor_data)

        self.timer = self.create_timer(self.cfg.replan_period, self.on_tick)
        self.get_logger().info(
            f"obstacle_avoidance ready safe={self.cfg.min_safe_distance} "
            f"emergency={self.cfg.emergency_stop_distance} "
            f"range={self.cfg.detection_range} lidar={self.use_lidar}")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_depth(self, msg):
        try:
            arr = image_msg_to_array(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)
            return
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        self.depth = arr
        self.depth_time = self.now()

    def on_scan(self, msg):
        self.scan = msg
        self.scan_time = self.now()

    def on_manual(self, msg):
        self.manual = msg

    def on_odom(self, msg):
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _sectors(self, now):
        if self.use_lidar and self.scan is not None and \
                self.scan_time is not None and now - self.scan_time <= self.depth_timeout:
            angles = np.linspace(self.scan.angle_min, self.scan.angle_max,
                                 len(self.scan.ranges))
            return sector_ranges_from_scan(
                angles.tolist(), self.scan.ranges, self.num_sectors,
                max_range=self.cfg.detection_range,
                min_valid_depth=self.cfg.min_valid_depth), "lidar"
        if self.depth is not None and self.depth_time is not None and \
                now - self.depth_time <= self.depth_timeout:
            arr = self.depth
            step_r = max(1, arr.shape[0] // 120)
            step_c = max(1, arr.shape[1] // 160)
            small = arr[::step_r, ::step_c]
            sectors = sector_min_ranges(
                small, self.num_sectors, self.vertical_fraction,
                max_range=self.cfg.detection_range,
                min_valid_depth=self.cfg.min_valid_depth)
            return sectors, "depth"
        return None, "none"

    def _publish(self, twist, status):
        self.pub.publish(twist)
        self.status_pub.publish(String(data=json.dumps(status)))

    def on_tick(self):
        now = self.now()
        manual = self.manual
        zero = Twist()
        if abs(manual.linear.x) + abs(manual.linear.y) + abs(manual.linear.z) + \
                abs(manual.angular.z) <= 1e-6:
            self._publish(zero, {"active": False, "reason": "idle",
                                 "sectors": [], "source": "none"})
            return

        sectors, source = self._sectors(now)
        if sectors is None:
            self._publish(manual, {"active": False, "reason": "no_depth_passthrough",
                                   "sectors": [], "source": source})
            return

        forward = project_forward(manual.linear.x, manual.linear.y, self.yaw)
        if forward <= 0.05:
            self._publish(manual, {"active": False, "reason": "no_forward_intent",
                                   "sectors": list(sectors), "source": source})
            return

        result = self.planner.plan(sectors, now, forward, self.max_speed)
        if not result.active:
            self._publish(manual, {"active": False, "reason": result.reason,
                                   "sectors": list(sectors), "source": source})
            return

        vx, vy = body_to_world(result.vx, 0.0, self.yaw)
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.linear.z = vx, vy, 0.0
        cmd.angular.z = result.wz
        self._publish(cmd, {
            "active": True,
            "reason": result.reason,
            "sectors": [round(s, 3) for s in sectors],
            "source": source,
            "side": result.side,
            "commanded_forward": round(result.vx, 3),
        })


def main():
    rclpy.init()
    node = ObstacleAvoidance()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
