"""AirSim -> ROS 2 bridge: NED to ENU, camera and depth publishing.

Subscribes to the unmodified AirSim ROS 2 wrapper and republishes a
ROS-native interface under /drone/*. No AirSim Python API is used here.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Image, CameraInfo
from geometry_msgs.msg import Quaternion

from forest_bringup.conversions import (
    ned_to_enu_point, ned_to_enu_vec, frd_ned_to_flu_enu_quaternion,
    frd_to_flu_vec, intrinsics, imu_ned_to_enu)


class ForestBridge(Node):
    def __init__(self):
        super().__init__("forest_bridge")
        p = self.declare_parameter
        p("vehicle_name", "ForestDrone")
        p("rgb_camera", "front_center")
        p("thermal_camera", "thermal")
        p("depth_camera", "front_center")
        p("odom_in", "")
        p("imu_in", "")
        p("rgb_in", "")
        p("depth_in", "")
        p("thermal_in", "")
        p("img_width", 640)
        p("img_height", 480)
        p("hfov_deg", 60.0)
        p("camera_pitch_deg", -11.5)
        p("rgb_frame_id", "drone/camera_rgb_optical")
        p("thermal_frame_id", "drone/camera_thermal_optical")
        p("depth_frame_id", "drone/camera_depth_optical")

        g = lambda n: self.get_parameter(n).value
        self.vehicle = g("vehicle_name")
        v = self.vehicle
        base = f"/airsim_node/{v}"
        self.odom_in = g("odom_in") or f"{base}/odom_local_ned"
        self.imu_in = g("imu_in") or f"{base}/imu/Imu"
        self.rgb_in = g("rgb_in") or f"{base}/{g('rgb_camera')}/Scene"
        self.depth_in = g("depth_in") or f"{base}/{g('depth_camera')}/DepthPlanar"
        self.thermal_in = g("thermal_in") or f"{base}/{g('thermal_camera')}/Infrared"

        width, height = int(g("img_width")), int(g("img_height"))
        K = intrinsics(width, height, float(g("hfov_deg")))
        self.rgb_info = self._camera_info(K, width, height, g("rgb_frame_id"))
        self.thermal_info = self._camera_info(K, width, height, g("thermal_frame_id"))
        self.depth_info = self._camera_info(K, width, height, g("depth_frame_id"))

        self.pub_odom = self.create_publisher(Odometry, "/drone/odom", 10)
        self.pub_imu = self.create_publisher(Imu, "/drone/imu", 10)
        self.pub_rgb = self.create_publisher(Image, "/drone/camera/rgb/image_raw",
                                             qos_profile_sensor_data)
        self.pub_rgb_info = self.create_publisher(
            CameraInfo, "/drone/camera/rgb/camera_info", 10)
        self.pub_depth = self.create_publisher(
            Image, "/drone/camera/depth/image_raw", qos_profile_sensor_data)
        self.pub_depth_info = self.create_publisher(
            CameraInfo, "/drone/camera/depth/camera_info", 10)
        self.pub_thermal = self.create_publisher(
            Image, "/drone/camera/thermal/image_raw", qos_profile_sensor_data)
        self.pub_thermal_info = self.create_publisher(
            CameraInfo, "/drone/camera/thermal/camera_info", 10)

        self.create_subscription(Odometry, self.odom_in, self.on_odom, 10)
        self.create_subscription(Imu, self.imu_in, self.on_imu, 10)
        self.create_subscription(Image, self.rgb_in, self.on_rgb,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_in, self.on_depth,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, self.thermal_in, self.on_thermal,
                                 qos_profile_sensor_data)

        self.get_logger().info(
            f"forest_bridge ready vehicle={v} odom_in={self.odom_in} "
            f"rgb_in={self.rgb_in} depth_in={self.depth_in} thermal_in={self.thermal_in}")

    @staticmethod
    def _camera_info(K, width, height, frame_id):
        info = CameraInfo()
        info.width, info.height = width, height
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [K["fx"], 0.0, K["cx"], 0.0, K["fy"], K["cy"], 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [K["fx"], 0.0, K["cx"], 0.0, 0.0, K["fy"], K["cy"], 0.0,
                  0.0, 0.0, 1.0, 0.0]
        info.header.frame_id = frame_id
        return info

    def on_odom(self, m):
        pos = m.pose.pose.position
        vel = m.twist.twist.linear
        ang = m.twist.twist.angular
        q = m.pose.pose.orientation
        x, y, z = ned_to_enu_point(pos.x, pos.y, pos.z)
        vx, vy, vz = ned_to_enu_vec(vel.x, vel.y, vel.z)
        # AirSim angular velocity is body FRD; ROS expects body FLU.
        wx, wy, wz = frd_to_flu_vec((ang.x, ang.y, ang.z))
        qw, qx, qy, qz = frd_ned_to_flu_enu_quaternion(q.w, q.x, q.y, q.z)
        out = Odometry()
        out.header.stamp = m.header.stamp
        out.header.frame_id = "drone/odom"
        out.child_frame_id = "drone/base_link"
        out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z = x, y, z
        out.pose.pose.orientation = Quaternion(w=qw, x=qx, y=qy, z=qz)
        out.twist.twist.linear.x, out.twist.twist.linear.y, out.twist.twist.linear.z = vx, vy, vz
        out.twist.twist.angular.x, out.twist.twist.angular.y, out.twist.twist.angular.z = wx, wy, wz
        self.pub_odom.publish(out)

    def on_imu(self, m):
        q = m.orientation
        accel = (m.linear_acceleration.x, m.linear_acceleration.y,
                 m.linear_acceleration.z)
        gyro = (m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z)
        a, g, qe = imu_ned_to_enu(accel, gyro, (q.w, q.x, q.y, q.z))
        out = Imu()
        out.header = m.header
        out.header.frame_id = "drone/imu"
        out.orientation = Quaternion(w=qe[0], x=qe[1], y=qe[2], z=qe[3])
        out.angular_velocity.x, out.angular_velocity.y, out.angular_velocity.z = g
        out.linear_acceleration.x, out.linear_acceleration.y, out.linear_acceleration.z = a
        out.angular_velocity_covariance = m.angular_velocity_covariance
        out.linear_acceleration_covariance = m.linear_acceleration_covariance
        self.pub_imu.publish(out)

    def _publish_image(self, m, publisher, info_publisher, info, frame_id):
        out = m
        out.header.frame_id = frame_id
        publisher.publish(out)
        info.header.stamp = m.header.stamp
        info_publisher.publish(info)

    def on_rgb(self, m):
        self._publish_image(m, self.pub_rgb, self.pub_rgb_info,
                            self.rgb_info, self.rgb_info.header.frame_id)

    def on_depth(self, m):
        self._publish_image(m, self.pub_depth, self.pub_depth_info,
                            self.depth_info, self.depth_info.header.frame_id)

    def on_thermal(self, m):
        self._publish_image(m, self.pub_thermal, self.pub_thermal_info,
                            self.thermal_info, self.thermal_info.header.frame_id)


def main():
    rclpy.init()
    node = ForestBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
