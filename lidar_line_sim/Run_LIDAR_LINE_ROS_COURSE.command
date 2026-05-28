#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTONAV_REPO="${AUTONAV_REPO:-$HOME/code/git/AutoNav_25-26}"

cleanup_pattern() {
  local pattern="$1"
  pkill -f "$pattern" >/dev/null 2>&1 || true
}

cleanup_stale_stack() {
  echo "Cleaning stale lidar-line ROS course processes before launch..."
  echo "Set AUTONAV_SIM_CLEAN_START=0 to skip this cleanup."

  cleanup_pattern "$SCRIPT_DIR/ros/lidar_line_course_stack.launch.py"
  sleep 0.5

  cleanup_pattern "$SCRIPT_DIR/simulated_world/ros_lidar_line_course.py"
  cleanup_pattern "$AUTONAV_REPO/isaac_ros-dev/install/autonav_detection/lib/autonav_detection/grade_detector"
  cleanup_pattern "$AUTONAV_REPO/isaac_ros-dev/install/autonav_detection/lib/autonav_detection/lidar_line_detector"
  cleanup_pattern "pointcloud_to_laserscan_node.*scan_pca_filtered_points"

  cleanup_pattern "/opt/ros/humble/lib/nav2_controller/controller_server"
  cleanup_pattern "/opt/ros/humble/lib/nav2_planner/planner_server"
  cleanup_pattern "/opt/ros/humble/lib/nav2_behaviors/behavior_server"
  cleanup_pattern "/opt/ros/humble/lib/nav2_bt_navigator/bt_navigator"
  cleanup_pattern "/opt/ros/humble/lib/nav2_waypoint_follower/waypoint_follower"
  cleanup_pattern "/opt/ros/humble/lib/nav2_velocity_smoother/velocity_smoother"
  cleanup_pattern "/opt/ros/humble/lib/nav2_smoother/smoother_server"
  cleanup_pattern "/opt/ros/humble/lib/nav2_lifecycle_manager/lifecycle_manager"
  cleanup_pattern "/opt/ros/humble/lib/robot_state_publisher/robot_state_publisher"
  sleep 0.5
}

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "ROS Humble setup not found at /opt/ros/humble/setup.bash" >&2
  exit 1
fi

if [[ ! -f "$AUTONAV_REPO/isaac_ros-dev/install/setup.bash" ]]; then
  echo "AutoNav install setup not found:" >&2
  echo "  $AUTONAV_REPO/isaac_ros-dev/install/setup.bash" >&2
  echo "Build/source AutoNav_25-26 first, or set AUTONAV_REPO." >&2
  exit 1
fi

set +u
source /opt/ros/humble/setup.bash
source "$AUTONAV_REPO/isaac_ros-dev/install/setup.bash"
set -u

if [[ "${AUTONAV_SIM_CLEAN_START:-1}" != "0" ]]; then
  cleanup_stale_stack
fi

ros2 launch "$SCRIPT_DIR/ros/lidar_line_course_stack.launch.py" \
  autonav_repo:="$AUTONAV_REPO"
