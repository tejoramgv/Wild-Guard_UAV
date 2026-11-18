"""3D target localisation and RGB/thermal association.

Converts 2D detections into world coordinates using the depth image, the
camera intrinsics/extrinsics and the drone pose. Publishes fused 3D records;
the database node is the only component that persists them.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from forest_interfaces.msg import Detection2DArray, Detection3D, Detection3DArray

from forest_bringup.conversions import (
    enu_to_ned_point, flu_enu_to_frd_ned_quaternion, intrinsics)
from forest_bringup.detection_logic import Detection2D, DetectionHistory, associate
from forest_bringup.localization import localize_detection
from forest_bringup.ros_utils import image_msg_to_array


class TargetLocalization(Node):
    def __init__(self):
        super().__init__("target_localization")
        p = self.declare_parameter
        p("rgb_det_topic", "/drone/detections/rgb")
        p("thermal_det_topic", "/drone/detections/thermal")
        p("depth_topic", "/drone/camera/depth/image_raw")
        p("odom_topic", "/drone/odom")
        p("camera_info_topic", "/drone/camera/depth/camera_info")
        p("img_width", 640)
        p("img_height", 480)
        p("hfov_deg", 60.0)
        p("camera_pitch_deg", -11.5)
        p("depth_is_perspective", False)
        p("max_time_delta", 0.15)
        p("min_iou", 0.2)
        p("output_topic", "/drone/detections/fused")
        p("dedup_radius", 1.5)
        p("dedup_seconds", 5.0)
        p("allow_ground_fallback", True)
        p("ground_z", 0.0)
        p("rate_hz", 10.0)
        p("detection_timeout", 1.0)

        g = lambda n: self.get_parameter(n).value
        self.K = intrinsics(int(g("img_width")), int(g("img_height")),
                            float(g("hfov_deg")))
        self.pitch_down = -float(g("camera_pitch_deg"))
        self.perspective = bool(g("depth_is_perspective"))
        self.max_time_delta = float(g("max_time_delta"))
        self.min_iou = float(g("min_iou"))
        self.allow_ground = bool(g("allow_ground_fallback"))
        self.ground_z = float(g("ground_z"))
        self.det_timeout = float(g("detection_timeout"))
        self.history = DetectionHistory(float(g("dedup_radius")),
                                        float(g("dedup_seconds")))

        self.det_rgb = None
        self.det_thermal = None
        self.det_rgb_time = None
        self.det_thermal_time = None
        self.depth = None
        self.odom = None
        self.camera_info = None

        self.pub = self.create_publisher(Detection3DArray, g("output_topic"), 10)
        self.create_subscription(Detection2DArray, g("rgb_det_topic"),
                                 lambda m: self._on_dets("rgb", m), 10)
        self.create_subscription(Detection2DArray, g("thermal_det_topic"),
                                 lambda m: self._on_dets("thermal", m), 10)
        self.create_subscription(Image, g("depth_topic"), self.on_depth,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, g("odom_topic"), self.on_odom, 10)
        self.create_subscription(CameraInfo, g("camera_info_topic"),
                                 self.on_info, 10)
        self.timer = self.create_timer(1.0 / float(g("rate_hz")), self.on_tick)
        self.get_logger().info(
            f"target_localization ready pitch_down={self.pitch_down} "
            f"output={g('output_topic')}")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_dets(self, source, msg):
        if source == "rgb":
            self.det_rgb, self.det_rgb_time = msg, self.now()
        else:
            self.det_thermal, self.det_thermal_time = msg, self.now()

    def on_depth(self, msg):
        try:
            arr = image_msg_to_array(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)
            return
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        self.depth = arr

    def on_odom(self, msg):
        self.odom = msg

    def on_info(self, msg):
        if len(msg.k) >= 9 and msg.k[0] > 0.0:
            self.camera_info = {"fx": msg.k[0], "fy": msg.k[4],
                                "cx": msg.k[2], "cy": msg.k[5],
                                "width": msg.width, "height": msg.height}
        else:
            self.camera_info = None

    def _fresh(self, stamp, now):
        return stamp is not None and (now - stamp) <= self.det_timeout

    def _to_det2d(self, m):
        source = "rgb" if m.source == "rgb" else "thermal"
        return Detection2D(m.class_name, m.confidence, m.cx, m.cy, m.width,
                           m.height, m.stamp, source, m.image_path)

    def on_tick(self):
        now = self.now()
        if self.odom is None:
            return
        rgb = [self._to_det2d(d) for d in self.det_rgb.detections] \
            if self._fresh(self.det_rgb_time, now) and self.det_rgb else []
        thermal = [self._to_det2d(d) for d in self.det_thermal.detections] \
            if self._fresh(self.det_thermal_time, now) and self.det_thermal else []
        if not rgb and not thermal:
            return

        fused = associate(rgb, thermal, max_time_delta=self.max_time_delta,
                          min_iou=self.min_iou)
        K = self.camera_info or self.K
        odom = self.odom
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        pos_ned = enu_to_ned_point(p.x, p.y, p.z)
        # /drone/odom is REP-103 FLU-body -> ENU; camera math needs FRD -> NED.
        q_ned = flu_enu_to_frd_ned_quaternion(q.w, q.x, q.y, q.z)

        out = Detection3DArray()
        out.header.stamp = odom.header.stamp
        out.header.frame_id = "drone/odom_enu"
        for item in fused:
            primary = item.primary
            result = localize_detection(
                primary, self.depth, K, pos_ned, q_ned,
                pitch_down_deg=self.pitch_down, ground_z=self.ground_z,
                allow_ground_fallback=self.allow_ground,
                perspective=self.perspective)
            if result is None:
                self.get_logger().debug(
                    f"could not localise {item.class_name} (no depth/ground hit)")
                continue
            det = Detection3D()
            det.class_name = item.class_name
            det.confidence = float(item.confidence)
            det.stamp = float(primary.stamp)
            det.x, det.y, det.z = (float(result["x"]), float(result["y"]),
                                   float(result["z"]))
            det.drone_x, det.drone_y, det.drone_z = (float(p.x), float(p.y), float(p.z))
            det.source = "+".join(item.sources)
            bbox = primary.bbox()
            det.bbox_x = int(round(bbox[0]))
            det.bbox_y = int(round(bbox[1]))
            det.bbox_w = int(round(primary.w))
            det.bbox_h = int(round(primary.h))
            det.image_path = primary.image_path
            det.method = result["method"]
            det.is_new = bool(self.history.check_and_add(
                item.class_name, (det.x, det.y, det.z), det.stamp))
            out.detections.append(det)
        if out.detections:
            self.pub.publish(out)


def main():
    rclpy.init()
    node = TargetLocalization()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
