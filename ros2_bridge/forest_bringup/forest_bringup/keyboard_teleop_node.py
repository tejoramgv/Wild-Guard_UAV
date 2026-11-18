"""Keyboard teleoperation: continuous velocity commands while keys are held.

The node publishes world-frame ENU velocity on /drone/cmd_vel_manual; the
obstacle-avoidance node owns the final /drone/cmd_vel. Heading-relative
forward/strafe input is rotated by the current ENU yaw, so "up" always means
"forward" regardless of where the drone is looking.
"""

import math
import select
import sys
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger

from forest_bringup.teleop import (
    ESTOP, LAND, TAKEOFF, TeleopConfig, TeleopState, body_to_world)

try:  # pynput is the only backend with real key-release events
    from pynput import keyboard as pynput_keyboard
except ImportError:  # pragma: no cover - headless machines
    pynput_keyboard = None


class _TerminalKeys(threading.Thread):
    """Degraded fallback: a key press is held for ``pulse_seconds``.

    Terminals do not deliver key-release events, so this backend cannot meet
    the hold-until-release requirement. It is opt-in and documented as such.
    """

    def __init__(self, on_press, on_release, pulse_seconds=0.25):
        super().__init__(daemon=True)
        self.on_press = on_press
        self.on_release = on_release
        self.pulse_seconds = pulse_seconds
        self._stop = threading.Event()
        self._map = {
            "\x1b[A": "up", "\x1b[B": "down", "\x1b[C": "right", "\x1b[D": "left",
            "w": "w", "s": "s", "a": "a", "d": "d", " ": "space",
            "l": "l", "t": "t", "x": "esc",
        }

    def run(self):
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue
                char = sys.stdin.read(1)
                if char == "\x1b":
                    # Distinguish an arrow escape sequence from a lone ESC.
                    more, _, _ = select.select([sys.stdin], [], [], 0.02)
                    if more:
                        char += sys.stdin.read(2)
                key = self._map.get(char)
                if not key:
                    continue
                self.on_press(key)
                threading.Timer(self.pulse_seconds,
                                lambda k=key: self.on_release(k)).start()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def stop(self):
        self._stop.set()


class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__("keyboard_teleop")
        p = self.declare_parameter
        p("max_speed", 2.0)
        p("max_vertical_speed", 1.5)
        p("max_yaw_rate", 1.2)
        p("input_backend", "pynput")  # pynput | terminal
        p("pulse_seconds", 0.25)
        p("publish_rate_hz", 20.0)
        p("enabled", True)

        g = lambda n: self.get_parameter(n).value
        self.teleop = TeleopState(TeleopConfig(
            max_speed=g("max_speed"),
            max_vertical_speed=g("max_vertical_speed"),
            max_yaw_rate=g("max_yaw_rate")))
        self.yaw = 0.0
        self.enabled = bool(g("enabled"))
        self.pub = self.create_publisher(Twist, "/drone/cmd_vel_manual", 10)
        self.create_subscription(Odometry, "/drone/odom", self.on_odom, 10)
        self.takeoff_cli = self.create_client(Trigger, "/drone/takeoff")
        self.land_cli = self.create_client(Trigger, "/drone/land")
        self.estop_cli = self.create_client(Trigger, "/drone/emergency_stop")
        self.timer = self.create_timer(1.0 / float(g("publish_rate_hz")),
                                       self.on_tick)
        self._listener = None
        self._terminal = None
        if not self.enabled:
            self.get_logger().warn("keyboard_teleop disabled by parameter")
            return
        if g("input_backend") == "terminal":
            self._terminal = _TerminalKeys(self._handle_press, self.teleop.release,
                                           float(g("pulse_seconds")))
            self._terminal.start()
            self.get_logger().warn(
                "terminal input backend active: keys are pulsed, not held; "
                "install pynput for true press/release semantics")
        elif pynput_keyboard is not None:
            self._listener = pynput_keyboard.Listener(
                on_press=self.on_key_press, on_release=self.on_key_release)
            self._listener.start()
            self.get_logger().info("keyboard_teleop ready (pynput)")
        else:
            self.get_logger().error(
                "pynput is not installed and input_backend!=terminal; "
                "install with 'pip install pynput' or set input_backend:=terminal")

    def on_odom(self, m):
        q = m.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _resolve(self, key):
        if pynput_keyboard is None:
            return key
        special = {
            pynput_keyboard.Key.up: "up",
            pynput_keyboard.Key.down: "down",
            pynput_keyboard.Key.left: "left",
            pynput_keyboard.Key.right: "right",
            pynput_keyboard.Key.space: "space",
            pynput_keyboard.Key.esc: "esc",
        }
        if key in special:
            return special[key]
        char = getattr(key, "char", None)
        if isinstance(char, str):
            return char.lower()
        return None

    def on_key_press(self, key):
        self._handle_press(self._resolve(key))

    def on_key_release(self, key):
        name = self._resolve(key)
        if name:
            self.teleop.release(name)

    def _handle_press(self, name):
        if not name:
            return
        self.teleop.press(name)
        for event in self.teleop.pop_events():
            if event == TAKEOFF:
                self._call_service(self.takeoff_cli, "takeoff")
            elif event == LAND:
                self._call_service(self.land_cli, "land")
            elif event == ESTOP:
                self._call_service(self.estop_cli, "emergency_stop")

    def _call_service(self, client, label):
        if not client.service_is_ready():
            self.get_logger().warn(f"{label} service not ready")
            return
        client.call_async(Trigger.Request())
        self.get_logger().info(f"{label} requested")

    def on_tick(self):
        cmd = self.teleop.compute()
        vx, vy = body_to_world(cmd.vx, cmd.vy, self.yaw)
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = vx, vy, cmd.vz
        msg.angular.z = cmd.wz
        self.pub.publish(msg)

    def destroy_node(self):
        if self._listener is not None:
            self._listener.stop()
        if self._terminal is not None:
            self._terminal.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = KeyboardTeleop()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
