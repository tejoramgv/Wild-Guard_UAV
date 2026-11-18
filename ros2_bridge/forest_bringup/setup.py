from setuptools import setup

package_name = "forest_bringup"

setup(
    name=package_name,
    version="0.2.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/forest.launch.py"]),
        ("share/" + package_name + "/config", ["config/forest_params.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "forest_bridge_node = forest_bringup.forest_bridge_node:main",
            "flight_executor_node = forest_bringup.flight_executor_node:main",
            "keyboard_teleop_node = forest_bringup.keyboard_teleop_node:main",
            "obstacle_avoidance_node = forest_bringup.obstacle_avoidance_node:main",
            "yolo_detection_node = forest_bringup.yolo_detection_node:main",
            "target_localization_node = forest_bringup.target_localization_node:main",
            "detection_database_node = forest_bringup.detection_database_node:main",
            "visibility_check_node = forest_bringup.visibility_check_node:main",
        ],
    },
)
