"""Flight executor: state machine, AirSim services and velocity forwarding.

Subscribes to the final arbitrated /drone/cmd_vel and publishes
airsim_interfaces/VelCmd to the AirSim wrapper's world-frame command topic.
The wrapper expects NED world velocity and a NED yaw rate in rad/s (it
converts the rate to degrees internally).
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from std_srvs.srv import Trigger

from forest_bringup.conversions import enu_cmd_to_ned
from forest_bringup.flight_state import FlightState, FlightStateMachine
from forest_bringup.safety import (
    FlightLimits, SafetySupervisor, VelocityCommand)


def _load_airsim_types():
    """Import AirSim wrapper interfaces lazily with a clear error message."""
    try:
        from airsim_interfaces.msg import VelCmd
        from airsim_interfaces.srv import Land, Takeoff
        return VelCmd, Takeoff, Land
    except ImportError as exc:  # pragma: no cover - depends on Windows/ROS ws
        raise RuntimeError(
            "airsim_interfaces is required. Build the AirSim ROS 2 wrapper "
            "(ros2/src/airsim_ros_pkgs) in the same colcon workspace first."
        ) from exc


class FlightExecutor(Node):
    def __init__(self):
        super().__init__("flight_executor")
        VelCmd, Takeoff, Land = _load_airsim_types()
        self.VelCmd, self.Takeoff, self.Land = VelCmd, Takeoff, Land

        p = self.declare_parameter
        p("vehicle_name", "ForestDrone")
        p("hover_alt", 3.0)
        p("ground_z", 0.0)
        p("max_speed", 2.0)
        p("max_vertical_speed", 1.5)
        p("max_yaw_rate", 1.2)
        p("max_acceleration", 2.5)
        p("max_altitude", 25.0)
        p("min_altitude", 0.3)
        p("cmd_timeout", 0.5)
        p("heartbeat_timeout", 2.0)
        p("control_rate_hz", 20.0)

        g = lambda n: self.get_parameter(n).value
        self.vehicle = g("vehicle_name")
        self.limits = FlightLimits(
            max_speed=g("max_speed"),
            max_vertical_speed=g("max_vertical_speed"),
            max_yaw_rate=g("max_yaw_rate"),
            max_acceleration=g("max_acceleration"),
            max_altitude=g("max_altitude"),
            min_altitude=g("min_altitude"),
            ground_z=g("ground_z"),
            command_timeout=g("cmd_timeout"),
            heartbeat_timeout=g("heartbeat_timeout"),
        )
        self.supervisor = SafetySupervisor(self.limits)
        self.state = FlightStateMachine(hover_altitude=g("hover_alt"))
        self.position_enu = None

        base = f"/airsim_node/{self.vehicle}"
        self.vel_pub = self.create_publisher(VelCmd, f"{base}/vel_cmd_world_frame", 10)
        self.state_pub = self.create_publisher(String, "/drone/state", 10)

        self.create_subscription(Twist, "/drone/cmd_vel", self.on_cmd, 10)
        self.create_subscription(Odometry, "/drone/odom", self.on_odom, 10)

        self.airsim_takeoff = self.create_client(Takeoff, f"{base}/takeoff")
        self.airsim_land = self.create_client(Land, f"{base}/land")

        self.create_service(Trigger, "/drone/takeoff", self.srv_takeoff)
        self.create_service(Trigger, "/drone/land", self.srv_land)
        self.create_service(Trigger, "/drone/hover", self.srv_hover)
        self.create_service(Trigger, "/drone/emergency_stop", self.srv_estop)
        self.create_service(Trigger, "/drone/clear_estop", self.srv_clear_estop)

        self.last_cmd = VelocityCommand()
        self.timer = self.create_timer(1.0 / float(g("control_rate_hz")), self.on_tick)
        self.get_logger().info(
            f"flight_executor ready vehicle={self.vehicle} "
            f"vel_topic={base}/vel_cmd_world_frame")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, m):
        p = m.pose.pose.position
        self.position_enu = (p.x, p.y, p.z)
        self.supervisor.update_telemetry(self.now())
        altitude = p.z - self.limits.ground_z
        previous = self.state.state
        self.state.on_odometry(altitude)
        if previous != self.state.state:
            self.get_logger().info(f"state {previous} -> {self.state.state}")

    def on_cmd(self, m):
        self.last_cmd = VelocityCommand(m.linear.x, m.linear.y,
                                        m.linear.z, m.angular.z)
        self.supervisor.arbiter.set_manual(self.last_cmd, self.now())
        moving = not self.last_cmd.is_zero()
        self.state.on_command(moving)

    def _publish_zero(self):
        msg = self.VelCmd()
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.z = 0.0
        self.vel_pub.publish(msg)

    def _publish_command(self, cmd):
        vx, vy, vz, wz = enu_cmd_to_ned(cmd.vx, cmd.vy, cmd.vz, cmd.wz)
        msg = self.VelCmd()
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        msg.twist.angular.z = wz
        self.vel_pub.publish(msg)

    def on_tick(self):
        now = self.now()
        if self.state.state == FlightState.EMERGENCY_STOP:
            self._publish_zero()
            status = "emergency_stop"
        elif self.state.command_allowed():
            status_obj = self.supervisor.evaluate(now, self.position_enu)
            self._publish_command(status_obj.command)
            status = status_obj.reason
        else:
            self._publish_zero()
            status = self.state.state.value
        self.state_pub.publish(String(data=status))

    def _takeoff_done(self, future):
        if future.result() is not None and future.result().success:
            self.get_logger().info("AirSim takeoff accepted")
        else:
            self.get_logger().error("AirSim takeoff failed; returning to LANDED")
            self.state.state = FlightState.LANDED

    def srv_takeoff(self, request, response):
        del request
        if self.state.is_landed():
            self.state.request_takeoff()
            if self.airsim_takeoff.service_is_ready():
                req = self.Takeoff.Request()
                req.wait_on_last_task = False
                self.airsim_takeoff.call_async(req).add_done_callback(self._takeoff_done)
                response.success, response.message = True, "takeoff requested"
            else:
                self.state.state = FlightState.LANDED
                response.success, response.message = False, "AirSim takeoff service unavailable"
        else:
            response.success, response.message = False, f"cannot take off from {self.state.state.value}"
        return response

    def _land_done(self, future):
        if future.result() is not None and future.result().success:
            self.get_logger().info("AirSim land accepted")

    def srv_land(self, request, response):
        del request
        if self.state.request_land() or self.state.state == FlightState.LANDED:
            if self.airsim_land.service_is_ready():
                req = self.Land.Request()
                req.wait_on_last_task = False
                self.airsim_land.call_async(req).add_done_callback(self._land_done)
                response.success, response.message = True, "land requested"
            else:
                response.success, response.message = False, "AirSim land service unavailable"
        else:
            response.success, response.message = False, "land not allowed in current state"
        return response

    def srv_hover(self, request, response):
        del request
        if self.state.request_hover():
            response.success, response.message = True, "hover"
        else:
            response.success, response.message = False, f"hover not allowed in {self.state.state.value}"
        return response

    def srv_estop(self, request, response):
        del request
        self.state.emergency_stop()
        self.supervisor.arbiter.emergency_stop()
        self._publish_zero()
        response.success, response.message = True, "emergency stop latched"
        return response

    def srv_clear_estop(self, request, response):
        del request
        self.state.clear_emergency()
        self.supervisor.arbiter.clear_emergency_stop()
        response.success, response.message = True, "emergency stop cleared"
        return response


def main():
    rclpy.init()
    node = FlightExecutor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
