"""YOLO inference node for RGB and thermal image streams.

Model loading is lazy and optional so the rest of the stack runs without
ultralytics installed. The default COCO model knows ``person`` and
``elephant`` (plus bird/bear/cow/horse/sheep) but NOT deer or wild boar;
custom forest weights are required for those classes. This is reported in the
logs and in YOLO-SETUP.md rather than silently assumed.
"""

import json
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from forest_interfaces.msg import Detection2D, Detection2DArray

from forest_bringup.ros_utils import (
    array_to_gray3, image_msg_to_array, stamp_to_seconds)

COCO_ANIMALS = {"person", "bird", "cat", "dog", "horse", "sheep", "cow",
                "elephant", "bear", "zebra", "giraffe"}


class YoloDetection(Node):
    def __init__(self):
        super().__init__("yolo_detection")
        p = self.declare_parameter
        p("rgb_topic", "/drone/camera/rgb/image_raw")
        p("thermal_topic", "/drone/camera/thermal/image_raw")
        p("model_rgb", "yolov8n.pt")
        p("model_thermal", "")
        p("confidence", 0.35)
        p("iou", 0.45)
        p("device", "cpu")
        p("rate_hz", 2.0)
        p("max_det", 50)
        p("allowed_classes", "")          # comma-separated model class names
        p("class_map", "{}")              # JSON: model_name -> canonical_name
        p("thermal_enabled", True)
        p("save_images", False)
        p("image_dir", "detections_images")

        g = lambda n: self.get_parameter(n).value
        self.conf = float(g("confidence"))
        self.iou = float(g("iou"))
        self.device = g("device")
        self.max_det = int(g("max_det"))
        self.thermal_enabled = bool(g("thermal_enabled"))
        self.save_images = bool(g("save_images"))
        self.image_dir = g("image_dir")
        self.allowed = {c.strip() for c in str(g("allowed_classes")).split(",") if c.strip()}
        self.class_map = json.loads(g("class_map") or "{}")
        if self.allowed:
            unknown = self.allowed - COCO_ANIMALS
            if unknown:
                self.get_logger().info(
                    f"custom/non-COCO classes requested: {sorted(unknown)} "
                    "(requires custom weights)")
        else:
            self.get_logger().warn(
                "allowed_classes is empty: every COCO class will be published")
        self.get_logger().info(
            "COCO models do not contain 'deer' or 'wild boar'; provide custom "
            "weights and/or class_map before claiming those detections")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for yolo_detection_node "
                "(pip install ultralytics)") from exc

        self.model_rgb = YOLO(g("model_rgb"))
        thermal_model = g("model_thermal") or g("model_rgb")
        self.model_thermal = self.model_rgb if thermal_model == g("model_rgb") \
            else YOLO(thermal_model)
        if self.thermal_model_is_rgb(thermal_model, g("model_rgb")):
            self.get_logger().warn(
                "thermal inference reuses the RGB model; thermal-trained "
                "weights are recommended (see YOLO-SETUP.md)")

        self.pub_rgb = self.create_publisher(
            Detection2DArray, "/drone/detections/rgb", 10)
        self.pub_thermal = self.create_publisher(
            Detection2DArray, "/drone/detections/thermal", 10)
        self.latest = {"rgb": None, "thermal": None}
        self.create_subscription(Image, g("rgb_topic"),
                                 lambda m: self._store("rgb", m),
                                 qos_profile_sensor_data)
        self.create_subscription(Image, g("thermal_topic"),
                                 lambda m: self._store("thermal", m),
                                 qos_profile_sensor_data)
        self.timer = self.create_timer(1.0 / float(g("rate_hz")), self.on_tick)
        self.frames = {"rgb": 0, "thermal": 0}
        self.get_logger().info("yolo_detection ready")

    @staticmethod
    def thermal_model_is_rgb(thermal_model, rgb_model):
        return thermal_model == rgb_model

    def _store(self, source, msg):
        self.latest[source] = msg

    def _infer(self, msg, model, source):
        try:
            arr = image_msg_to_array(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)
            return None
        # Ultralytics/OpenCV expect BGR; ROS rgb8 is RGB.
        if arr.ndim == 3 and arr.shape[2] >= 3:
            if (msg.encoding or "").lower() in ("rgb8", "rgba8"):
                arr = arr[:, :, [2, 1, 0]]
            rgb_for_save = arr
        else:
            arr = array_to_gray3(arr)
            rgb_for_save = arr
        results = model.predict(arr, conf=self.conf, iou=self.iou,
                                device=self.device, max_det=self.max_det,
                                verbose=False)
        out = Detection2DArray()
        out.header = msg.header
        stamp = stamp_to_seconds(msg.header.stamp)
        image_path = ""
        if self.save_images:
            os.makedirs(self.image_dir, exist_ok=True)
            image_path = os.path.join(
                self.image_dir, f"{source}_{int(stamp * 1000)}.jpg")
            try:
                import cv2
                cv2.imwrite(image_path, rgb_for_save)
            except Exception as exc:  # pragma: no cover - IO dependent
                self.get_logger().warn(f"image save failed: {exc}")
                image_path = ""
        names = results[0].names
        boxes = results[0].boxes
        if boxes is None:
            return out
        for box in boxes:
            cls_id = int(box.cls[0].item())
            model_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) \
                else names[cls_id]
            if self.allowed and model_name not in self.allowed:
                continue
            canonical = self.class_map.get(model_name, model_name)
            xywh = box.xywh[0].tolist()
            det = Detection2D()
            det.class_name = canonical
            det.confidence = float(box.conf[0].item())
            det.stamp = stamp
            det.source = source
            det.cx, det.cy, det.width, det.height = (float(xywh[0]), float(xywh[1]),
                                                     float(xywh[2]), float(xywh[3]))
            det.image_path = image_path
            out.detections.append(det)
        return out

    def on_tick(self):
        for source, publisher, model in (
                ("rgb", self.pub_rgb, self.model_rgb),
                ("thermal", self.pub_thermal, self.model_thermal)):
            if source == "thermal" and not self.thermal_enabled:
                continue
            msg = self.latest[source]
            if msg is None:
                continue
            self.latest[source] = None
            out = self._infer(msg, model, source)
            if out is None:
                continue
            self.frames[source] += 1
            publisher.publish(out)
            if out.detections:
                self.get_logger().info(
                    f"{source}: {len(out.detections)} detection(s): "
                    + ", ".join(f"{d.class_name}:{d.confidence:.2f}"
                                for d in out.detections))


def main():
    rclpy.init()
    node = YoloDetection()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
