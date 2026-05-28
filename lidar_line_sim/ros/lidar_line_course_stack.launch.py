#!/usr/bin/env python3
"""Launch the canonical ROS lidar-line course simulation stack.

Run from a sourced AutoNav ROS environment:
  ros2 launch /path/to/lidar_line_course_stack.launch.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_autonav_repo() -> str:
    return os.environ.get(
        "AUTONAV_REPO",
        str(Path.home() / "code/git/AutoNav_25-26"),
    )


def _robot_description_candidates(autonav_repo: str) -> list[Path]:
    candidates = [
        Path(autonav_repo)
        / "isaac_ros-dev"
        / "install"
        / "bringup"
        / "share"
        / "bringup"
        / "description"
        / "shogi.urdf",
        Path(autonav_repo)
        / "isaac_ros-dev"
        / "src"
        / "bringup"
        / "description"
        / "shogi.urdf",
    ]
    try:
        candidates.append(
            Path(get_package_share_directory("bringup"))
            / "description"
            / "shogi.urdf"
        )
    except Exception:
        pass
    return candidates


def _load_robot_description(autonav_repo: str) -> str:
    for model_path in _robot_description_candidates(autonav_repo):
        if model_path.is_file():
            return model_path.read_text(encoding="utf-8")
    searched = "\n  ".join(str(path) for path in _robot_description_candidates(autonav_repo))
    raise FileNotFoundError(f"Could not find shogi.urdf. Searched:\n  {searched}")


def _robot_state_publisher(context, *args, **kwargs):
    autonav_repo = LaunchConfiguration("autonav_repo").perform(context)
    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{
                "robot_description": _load_robot_description(autonav_repo),
                "use_sim_time": False,
            }],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    sim_root = Path(__file__).resolve().parents[1]
    harness = sim_root / "simulated_world" / "ros_lidar_line_course.py"
    autonav_repo = LaunchConfiguration("autonav_repo")
    scenario = LaunchConfiguration("scenario")
    course_config = LaunchConfiguration("course_config")
    nav2_params = LaunchConfiguration("nav2_params")
    bt_xml = LaunchConfiguration("bt_xml")
    launch_nav2 = LaunchConfiguration("launch_nav2")
    ground_truth_pca = LaunchConfiguration("ground_truth_pca")

    detection_launch = os.path.join(
        get_package_share_directory("autonav_detection"),
        "launch",
        "detection.launch.py",
    )
    nav2_launch = os.path.join(
        get_package_share_directory("nav2_bringup"),
        "launch",
        "navigation_launch.py",
    )

    harness_node = ExecuteProcess(
        cmd=[
            sys.executable,
            str(harness),
            "--ros-args",
            "-p",
            ["scenario:=", scenario],
            "-p",
            ["course_config:=", course_config],
            "-p",
            ["publish_ground_truth_pca:=", ground_truth_pca],
        ],
        output="screen",
    )

    detection_real_pca = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(detection_launch),
        launch_arguments={
            "enable_line": "false",
            "enable_grade": "true",
            "enable_lidar_line": "true",
        }.items(),
        condition=UnlessCondition(ground_truth_pca),
    )
    detection_ground_truth_pca = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(detection_launch),
        launch_arguments={
            "enable_line": "false",
            "enable_grade": "false",
            "enable_lidar_line": "true",
        }.items(),
        condition=IfCondition(ground_truth_pca),
    )

    pca_scan = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pca_cloud_to_laserscan",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "target_frame": "base_link",
            "min_height": -0.10,
            "max_height": 1.50,
            "angle_min": -1.5708,
            "angle_max": 1.5708,
            "angle_increment": 0.0087,
            "scan_time": 0.1,
            "range_min": 0.30,
            "range_max": 25.0,
            "use_inf": True,
        }],
        remappings=[
            ("cloud_in", "/scan_pca_filtered_points"),
            ("scan", "/scan_pca_filtered"),
        ],
    )

    pca_scan_clear = Node(
        package="pointcloud_to_laserscan",
        executable="pointcloud_to_laserscan_node",
        name="pca_cloud_to_laserscan_clear",
        output="screen",
        parameters=[{
            "use_sim_time": False,
            "target_frame": "base_link",
            "min_height": -0.10,
            "max_height": 1.50,
            "angle_min": -1.2217,
            "angle_max": 1.2217,
            "angle_increment": 0.0087,
            "scan_time": 0.1,
            "range_min": 0.30,
            "range_max": 25.0,
            "use_inf": True,
        }],
        remappings=[
            ("cloud_in", "/scan_pca_filtered_points"),
            ("scan", "/scan_pca_filtered_clear"),
        ],
    )

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_launch),
        launch_arguments={
            "params_file": nav2_params,
            "use_sim_time": "false",
            "default_bt_xml_filename": bt_xml,
        }.items(),
        condition=IfCondition(launch_nav2),
    )
    return LaunchDescription([
        DeclareLaunchArgument("autonav_repo", default_value=_default_autonav_repo()),
        DeclareLaunchArgument("scenario", default_value="canonical_5ft_gap"),
        DeclareLaunchArgument("course_config", default_value="__auto__"),
        DeclareLaunchArgument(
            "nav2_params",
            default_value=[
                autonav_repo,
                "/isaac_ros-dev/src/slam/config/nav2_paramsv2.yaml",
            ],
        ),
        DeclareLaunchArgument(
            "bt_xml",
            default_value=[
                autonav_repo,
                "/isaac_ros-dev/install/slam/share/slam/behavior_trees/bt_nav.xml",
            ],
        ),
        DeclareLaunchArgument("launch_nav2", default_value="true"),
        DeclareLaunchArgument("ground_truth_pca", default_value="false"),
        OpaqueFunction(function=_robot_state_publisher),
        harness_node,
        detection_real_pca,
        detection_ground_truth_pca,
        pca_scan,
        pca_scan_clear,
        nav2,
    ])
