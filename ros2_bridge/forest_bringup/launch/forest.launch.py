"""Full ForestAirSim runtime graph.

AirSim runs on Windows; this launch connects to it over the network. Each
node is independent and can be launched alone for debugging.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("forest_bringup")
    params = os.path.join(share, "config", "forest_params.yaml")

    args = [
        DeclareLaunchArgument("start_airsim", default_value="true"),
        DeclareLaunchArgument("airsim_host", default_value="localhost"),
        DeclareLaunchArgument("publish_clock", default_value="False"),
        DeclareLaunchArgument("vehicle_name", default_value="ForestDrone"),
        DeclareLaunchArgument("enable_keyboard", default_value="true"),
        DeclareLaunchArgument("enable_yolo", default_value="true"),
        DeclareLaunchArgument("enable_thermal", default_value="true"),
        DeclareLaunchArgument("input_backend", default_value="pynput"),
        DeclareLaunchArgument("use_lidar", default_value="false"),
        DeclareLaunchArgument("yolo_model_rgb", default_value="yolov8n.pt"),
        DeclareLaunchArgument("yolo_model_thermal", default_value=""),
        DeclareLaunchArgument("db_path", default_value="detections.sqlite3"),
        DeclareLaunchArgument("csv_path", default_value="detections.csv"),
    ]

    airsim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("airsim_ros_pkgs"),
                         "launch", "airsim_node.launch.py")),
        launch_arguments={"host": LaunchConfiguration("airsim_host"),
                          "publish_clock": LaunchConfiguration("publish_clock")}.items(),
        condition=IfCondition(LaunchConfiguration("start_airsim")),
    )

    bridge = Node(
        package="forest_bringup", executable="forest_bridge_node",
        name="forest_bridge", output="screen",
        parameters=[params, {"vehicle_name": LaunchConfiguration("vehicle_name")}])

    executor = Node(
        package="forest_bringup", executable="flight_executor_node",
        name="flight_executor", output="screen",
        parameters=[params, {"vehicle_name": LaunchConfiguration("vehicle_name")}])

    keyboard = Node(
        package="forest_bringup", executable="keyboard_teleop_node",
        name="keyboard_teleop", output="screen",
        parameters=[params, {"input_backend": LaunchConfiguration("input_backend")}],
        condition=IfCondition(LaunchConfiguration("enable_keyboard")))

    avoidance = Node(
        package="forest_bringup", executable="obstacle_avoidance_node",
        name="obstacle_avoidance", output="screen",
        parameters=[params, {"use_lidar": LaunchConfiguration("use_lidar")}])

    yolo = Node(
        package="forest_bringup", executable="yolo_detection_node",
        name="yolo_detection", output="screen",
        parameters=[params, {
            "model_rgb": LaunchConfiguration("yolo_model_rgb"),
            "model_thermal": LaunchConfiguration("yolo_model_thermal"),
            "thermal_enabled": LaunchConfiguration("enable_thermal")}],
        condition=IfCondition(LaunchConfiguration("enable_yolo")))

    localization = Node(
        package="forest_bringup", executable="target_localization_node",
        name="target_localization", output="screen", parameters=[params])

    database = Node(
        package="forest_bringup", executable="detection_database_node",
        name="detection_database", output="screen",
        parameters=[params, {"db_path": LaunchConfiguration("db_path"),
                             "csv_path": LaunchConfiguration("csv_path")}])

    visibility = Node(
        package="forest_bringup", executable="visibility_check_node",
        name="visibility_check", output="screen", parameters=[params])

    return LaunchDescription(args + [airsim, bridge, executor, keyboard,
                                     avoidance, yolo, localization, database,
                                     visibility])
