#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTONAV_REPO="${AUTONAV_REPO:-$HOME/code/git/AutoNav_25-26}"

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

ros2 launch "$SCRIPT_DIR/ros/lidar_line_course_stack.launch.py" \
  autonav_repo:="$AUTONAV_REPO"
