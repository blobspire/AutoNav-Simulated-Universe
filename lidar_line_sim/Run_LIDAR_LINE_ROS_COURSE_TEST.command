#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTONAV_REPO="${AUTONAV_REPO:-$HOME/code/git/AutoNav_25-26}"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/ros_course_runs/$(date +%Y%m%d_%H%M%S)}"

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

mkdir -p "$RUN_DIR"

stop_process() {
  local pid="$1"
  local name="$2"
  local scope="${3:-process}"
  local target="$pid"

  if [[ "$scope" == "group" ]]; then
    target="-$pid"
  fi

  is_running() {
    if [[ "$scope" == "group" ]]; then
      pgrep -g "$pid" >/dev/null 2>&1
    else
      kill -0 "$pid" 2>/dev/null
    fi
  }

  if [[ -z "$pid" ]] || ! is_running; then
    return
  fi

  kill -INT -- "$target" 2>/dev/null || true
  for _ in {1..20}; do
    if ! is_running; then
      wait "$pid" 2>/dev/null || true
      return
    fi
    sleep 0.25
  done

  echo "Timed out waiting for $name to stop after SIGINT; sending SIGTERM." >&2
  kill -TERM -- "$target" 2>/dev/null || true
  for _ in {1..20}; do
    if ! is_running; then
      wait "$pid" 2>/dev/null || true
      return
    fi
    sleep 0.25
  done

  echo "Timed out waiting for $name to stop after SIGTERM; sending SIGKILL." >&2
  kill -KILL -- "$target" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

cleanup() {
  set +e
  if [[ -n "${bag_pid:-}" ]]; then
    stop_process "$bag_pid" "rosbag recorder"
  fi
  if [[ -n "${stack_pid:-}" ]]; then
    stop_process "$stack_pid" "ROS/Nav2 stack" "group"
  fi
}
trap cleanup EXIT INT TERM

setsid ros2 launch "$SCRIPT_DIR/ros/lidar_line_course_stack.launch.py" \
  autonav_repo:="$AUTONAV_REPO" &
stack_pid=$!

sleep 8

ros2 bag record --include-hidden-topics \
  -o "$RUN_DIR/bag" \
  /tf \
  /tf_static \
  /local_ekf/odom \
  /odom \
  /cmd_vel \
  /cmd_vel_nav \
  /autonomous_mode \
  /scan_fullframe \
  /scan_pca_filtered \
  /scan_pca_filtered_clear \
  /lidar_line_points \
  /lidar_line_costmap \
  /lidar_line_detection/diagnostics \
  /scan_pca_filtered_points \
  /terrain/grade_map \
  /pca/surface_normal \
  /local_costmap/costmap \
  /local_costmap/costmap_raw \
  /global_costmap/costmap \
  /global_costmap/costmap_raw \
  /plan \
  /local_plan \
  /trajectories \
  /transformed_global_plan \
  /evaluation \
  /navigate_to_pose/_action/status \
  /follow_path/_action/status \
  /compute_path_to_pose/_action/status &
bag_pid=$!

sleep 6

ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 2.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}" \
  --feedback | tee "$RUN_DIR/goal.log"

sleep 2
cleanup
set -e
trap - EXIT INT TERM

if [[ -x "$AUTONAV_REPO/scripts/run_lidar_line_bag_analysis.sh" ]]; then
  "$AUTONAV_REPO/scripts/run_lidar_line_bag_analysis.sh" "$RUN_DIR/bag" \
    | tee "$RUN_DIR/analysis.log"
else
  echo "Bag saved at $RUN_DIR/bag"
  echo "Analysis script not executable: $AUTONAV_REPO/scripts/run_lidar_line_bag_analysis.sh"
fi
