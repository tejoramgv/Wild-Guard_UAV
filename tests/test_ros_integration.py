"""ROS 2 integration tests (in-process).

These instantiate the real nodes and exercise real ROS messages/services. They
deliberately avoid cross-process DDS discovery, which is blocked in some
sandboxed/container environments; in-process communication uses the same
rclpy/DDS stack but does not depend on multicast.

Run with a sourced workspace:

    source /opt/ros/jazzy/setup.bash
    source ~/forest_ws/install/setup.bash   # workspace with forest_interfaces/bringup
    pytest -m integration

Skipped automatically when the workspace is not sourced/built.
"""

import math
import os
import shutil
import sqlite3
import struct
import subprocess

import pytest

pytestmark = pytest.mark.integration


def _ros_ready():
    if shutil.which("ros2") is None:
        return False
    try:
        import rclpy  # noqa: F401
        import forest_interfaces.msg  # noqa: F401
    except Exception:
        return False
    for pkg in ("forest_bringup", "forest_interfaces"):
        try:
            result = subprocess.run(["ros2", "pkg", "prefix", pkg],
                                    capture_output=True, timeout=10)
        except Exception:
            return False
        if result.returncode != 0:
            return False
    return True


requires_ros = pytest.mark.skipif(not _ros_ready(),
                                 reason="ROS 2 workspace not sourced/built")


def _init(args):
    import rclpy
    os.environ.setdefault("ROS_DOMAIN_ID", "77")
    rclpy.init(args=["--ros-args"] + args)
    return rclpy


def _spin(executor, cycles=30, timeout=0.05):
    for _ in range(cycles):
        executor.spin_once(timeout_sec=timeout)


@requires_ros
def test_database_node_receives_and_dedups(tmp_path):
    rclpy = _init(["-p", f"db_path:={tmp_path / 'itest.sqlite3'}",
                   "-p", "log_period:=100.0"])
    from rclpy.executors import SingleThreadedExecutor
    from forest_interfaces.msg import Detection3D, Detection3DArray
    from forest_bringup.detection_database_node import DetectionDatabase

    db = DetectionDatabase()
    pub_node = rclpy.create_node("itest_publisher")
    pub = pub_node.create_publisher(Detection3DArray, "/drone/detections/fused", 10)
    executor = SingleThreadedExecutor()
    executor.add_node(db)
    executor.add_node(pub_node)
    try:
        def publish_once(x, stamp):
            msg = Detection3DArray()
            msg.header.stamp = pub_node.get_clock().now().to_msg()
            det = Detection3D()
            det.class_name = "person"
            det.confidence = 0.9
            det.stamp = stamp
            det.x, det.y, det.z = x, 2.0, 0.0
            det.drone_x, det.drone_y, det.drone_z = 0.0, 0.0, 3.0
            det.source = "fused"
            det.bbox_x, det.bbox_y, det.bbox_w, det.bbox_h = 10, 20, 30, 40
            det.method = "depth"
            det.is_new = True
            msg.detections.append(det)
            pub.publish(msg)
            _spin(executor, 5)

        publish_once(1.0, 100.0)   # stored
        publish_once(1.2, 100.1)   # duplicate -> suppressed
        publish_once(50.0, 100.2)  # far -> stored
        _spin(executor, 20)
        db.store.conn.commit()
        rows = db.store.conn.execute(
            "SELECT class_name, x FROM detections ORDER BY id").fetchall()
        assert len(rows) == 2, rows
        assert rows[0][0] == "person"
    finally:
        executor.shutdown()
        db.destroy_node()
        pub_node.destroy_node()
        rclpy.shutdown()


@requires_ros
def test_database_export_csv_service(tmp_path):
    db_path = tmp_path / "export.sqlite3"
    csv_path = tmp_path / "export.csv"
    rclpy = _init(["-p", f"db_path:={db_path}", "-p", f"csv_path:={csv_path}",
                   "-p", "log_period:=100.0"])
    from rclpy.executors import SingleThreadedExecutor
    from std_srvs.srv import Trigger
    from forest_bringup.detection_database_node import DetectionDatabase

    db = DetectionDatabase()
    client_node = rclpy.create_node("itest_export_client")
    client = client_node.create_client(Trigger, "/drone/detections/export_csv")
    executor = SingleThreadedExecutor()
    executor.add_node(db)
    executor.add_node(client_node)
    try:
        for _ in range(50):
            if client.service_is_ready():
                break
            executor.spin_once(timeout_sec=0.1)
        assert client.service_is_ready()
        future = client.call_async(Trigger.Request())
        for _ in range(50):
            if future.done():
                break
            executor.spin_once(timeout_sec=0.1)
        assert future.done() and future.result().success
        assert csv_path.exists()
        assert csv_path.read_text().splitlines()[0].startswith("class_name")
    finally:
        executor.shutdown()
        db.destroy_node()
        client_node.destroy_node()
        rclpy.shutdown()


@requires_ros
def test_localization_pipeline_depth_to_world(tmp_path):
    rclpy = _init(["-p", "camera_pitch_deg:=-11.5", "-p", "rate_hz:=20.0",
                   "-p", "img_width:=640", "-p", "img_height:=480",
                   "-p", "hfov_deg:=60.0"])
    from rclpy.executors import SingleThreadedExecutor
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CameraInfo, Image
    from forest_interfaces.msg import (Detection2D, Detection2DArray,
                                       Detection3DArray)
    from forest_bringup.target_localization_node import TargetLocalization

    loc = TargetLocalization()
    driver = rclpy.create_node("itest_perception_driver")
    listener = rclpy.create_node("itest_fused_listener")
    received = []
    listener.create_subscription(
        Detection3DArray, "/drone/detections/fused",
        lambda m: received.append(m), 10)
    pub_odom = driver.create_publisher(Odometry, "/drone/odom", 10)
    pub_info = driver.create_publisher(CameraInfo, "/drone/camera/depth/camera_info", 10)
    pub_depth = driver.create_publisher(Image, "/drone/camera/depth/image_raw", 10)
    pub_det = driver.create_publisher(Detection2DArray, "/drone/detections/rgb", 10)

    width, height = 640, 480
    fx = (width / 2.0) / math.tan(math.radians(60.0) / 2.0)

    odom = Odometry()
    odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = 0.0, 0.0, 5.0
    odom.pose.pose.orientation.w = 1.0

    info = CameraInfo()
    info.width, info.height = width, height
    info.k = [fx, 0.0, width / 2.0, 0.0, fx, height / 2.0, 0.0, 0.0, 1.0]

    depth = Image()
    depth.width, depth.height = width, height
    depth.encoding = "32FC1"
    depth.step = width * 4
    depth.data = struct.pack("<" + "f" * (width * height), *([10.0] * (width * height)))

    dets = Detection2DArray()
    dets.header.stamp = driver.get_clock().now().to_msg()
    det = Detection2D()
    det.class_name, det.confidence, det.stamp = "deer", 0.8, 1.0
    det.source = "rgb"
    det.cx, det.cy, det.width, det.height = 320.0, 240.0, 40.0, 60.0
    dets.detections.append(det)

    executor = SingleThreadedExecutor()
    for node in (loc, driver, listener):
        executor.add_node(node)
    try:
        for _ in range(80):
            pub_odom.publish(odom)
            pub_info.publish(info)
            pub_depth.publish(depth)
            pub_det.publish(dets)
            _spin(executor, 5)
            if received:
                break
        assert received, "no fused detections received"
        fused = received[-1].detections[0]
        assert fused.method == "depth"
        # Odom identity = level FLU body facing east (+X ENU). The depth ray
        # pitched down 11.5 deg lands 10 m forward and 1.99 m below.
        assert math.isclose(fused.x, 10.0 * math.cos(math.radians(11.5)),
                            abs_tol=1e-3)
        assert math.isclose(fused.y, 0.0, abs_tol=1e-3)
        assert math.isclose(fused.z, 5.0 - 10.0 * math.sin(math.radians(11.5)),
                            abs_tol=1e-3)
        assert fused.class_name == "deer" and fused.source == "rgb"
    finally:
        executor.shutdown()
        for node in (loc, driver, listener):
            node.destroy_node()
        rclpy.shutdown()
