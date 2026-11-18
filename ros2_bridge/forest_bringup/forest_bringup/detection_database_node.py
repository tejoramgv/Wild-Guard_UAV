"""Detection database node: SQLite persistence, deduplication and CSV export."""

import json

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from forest_interfaces.msg import Detection3DArray
from forest_bringup.storage import DetectionRecord, DetectionStore


class DetectionDatabase(Node):
    def __init__(self):
        super().__init__("detection_database")
        p = self.declare_parameter
        p("input_topic", "/drone/detections/fused")
        p("db_path", "detections.sqlite3")
        p("csv_path", "detections.csv")
        p("dedup_radius", 1.5)
        p("dedup_seconds", 5.0)
        p("log_period", 10.0)

        g = lambda n: self.get_parameter(n).value
        self.store = DetectionStore(g("db_path"), float(g("dedup_radius")),
                                    float(g("dedup_seconds")))
        self.csv_path = g("csv_path")
        self.accepted = 0
        self.suppressed = 0
        self.create_subscription(Detection3DArray, g("input_topic"),
                                 self.on_detections, 10)
        self.create_service(Trigger, "/drone/detections/export_csv",
                            self.srv_export)
        self.create_service(Trigger, "/drone/detections/stats", self.srv_stats)
        self.create_timer(float(g("log_period")), self.log_stats)
        self.get_logger().info(
            f"detection_database ready db={g('db_path')} "
            f"existing={self.store.count()}")

    def on_detections(self, msg):
        for d in msg.detections:
            rec = DetectionRecord(
                class_name=d.class_name, confidence=float(d.confidence),
                x=float(d.x), y=float(d.y), z=float(d.z), stamp=float(d.stamp),
                drone_x=float(d.drone_x), drone_y=float(d.drone_y),
                drone_z=float(d.drone_z), source=d.source,
                bbox_x=int(d.bbox_x), bbox_y=int(d.bbox_y),
                bbox_w=int(d.bbox_w), bbox_h=int(d.bbox_h),
                image_path=d.image_path)
            row_id = self.store.insert_unique(rec)
            if row_id is None:
                self.suppressed += 1
                continue
            self.accepted += 1
            self.get_logger().info(
                f"stored #{row_id} {rec.class_name} conf={rec.confidence:.2f} "
                f"pos=({rec.x:.1f},{rec.y:.1f},{rec.z:.1f}) src={rec.source}")
        if self.accepted % 25 == 0 and self.accepted > 0:
            self.get_logger().info(
                f"accepted={self.accepted} suppressed={self.suppressed} "
                f"total={self.store.count()}")

    def srv_export(self, request, response):
        del request
        path = self.store.export_csv(self.csv_path)
        rows = self.store.count()
        response.success = True
        response.message = f"exported {rows} rows to {path}"
        self.get_logger().info(response.message)
        return response

    def srv_stats(self, request, response):
        del request
        counts = self.store.counts_by_class()
        response.success = True
        response.message = json.dumps({
            "total": self.store.count(),
            "by_class": counts,
            "accepted": self.accepted,
            "suppressed": self.suppressed,
        })
        return response

    def log_stats(self):
        if self.accepted:
            self.get_logger().info(
                f"detections total={self.store.count()} "
                f"by_class={self.store.counts_by_class()}")

    def destroy_node(self):
        self.store.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = DetectionDatabase()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
