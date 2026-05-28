#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTONAV_REPO="${AUTONAV_REPO:-$HOME/code/git/AutoNav_25-26}"
SCENARIO="${SCENARIO:-canonical_5ft_gap}"
COURSE_CONFIG="${COURSE_CONFIG:-}"
RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/ros_course_runs/${SCENARIO}_$(date +%Y%m%d_%H%M%S)}"

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

read -r default_goal_x default_goal_y default_goal_qz default_goal_qw resolved_course_config scenario_id < <(
  python3 - "$SCRIPT_DIR" "$SCENARIO" "$COURSE_CONFIG" <<'PY'
import math
from pathlib import Path
import sys

script_dir = Path(sys.argv[1])
scenario = sys.argv[2]
course_config = sys.argv[3].strip()
sys.path.insert(0, str(script_dir / "simulated_world"))
from lidar_line_course import load_lidar_line_course  # noqa: E402

course = load_lidar_line_course(course_config or None, scenario_id=scenario)
half = 0.5 * course.goal_yaw_rad
print(
    f"{course.goal[0]} {course.goal[1]} {math.sin(half)} "
    f"{math.cos(half)} {course.config_path} {course.scenario_id}"
)
PY
)
GOAL_X="${GOAL_X:-$default_goal_x}"
GOAL_Y="${GOAL_Y:-$default_goal_y}"
GOAL_Z="${GOAL_Z:-0.0}"
GOAL_QZ="${GOAL_QZ:-$default_goal_qz}"
GOAL_YAW_W="${GOAL_YAW_W:-$default_goal_qw}"
GROUND_TRUTH_PCA="${GROUND_TRUTH_PCA:-false}"
STRICT_SCENARIO_GEOMETRY="${STRICT_SCENARIO_GEOMETRY:-0}"
GOAL_TIMEOUT="${GOAL_TIMEOUT:-${GOAL_TIMEOUT_SEC:-180s}}"
STARTUP_WAIT_SEC="${STARTUP_WAIT_SEC:-8}"
PRE_GOAL_WAIT_SEC="${PRE_GOAL_WAIT_SEC:-6}"
if [[ "$GOAL_TIMEOUT" =~ ^[0-9]+$ ]]; then
  GOAL_TIMEOUT="${GOAL_TIMEOUT}s"
fi

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

launch_args=(
  autonav_repo:="$AUTONAV_REPO"
  scenario:="$SCENARIO"
  ground_truth_pca:="$GROUND_TRUTH_PCA"
)
if [[ -n "$COURSE_CONFIG" ]]; then
  launch_args+=(course_config:="$COURSE_CONFIG")
fi

setsid ros2 launch "$SCRIPT_DIR/ros/lidar_line_course_stack.launch.py" "${launch_args[@]}" &
stack_pid=$!

sleep "$STARTUP_WAIT_SEC"

ros2 bag record --include-hidden-topics \
  -o "$RUN_DIR/bag" \
  /rosout \
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

sleep "$PRE_GOAL_WAIT_SEC"

echo "Sending NavigateToPose goal: scenario=$scenario_id x=$GOAL_X y=$GOAL_Y z=$GOAL_Z qz=$GOAL_QZ w=$GOAL_YAW_W ground_truth_pca=$GROUND_TRUTH_PCA timeout=$GOAL_TIMEOUT"
goal_status=0
set +e
timeout --foreground "$GOAL_TIMEOUT" ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: $GOAL_X, y: $GOAL_Y, z: $GOAL_Z}, orientation: {z: $GOAL_QZ, w: $GOAL_YAW_W}}}}" \
  --feedback | tee "$RUN_DIR/goal.log"
goal_status="${PIPESTATUS[0]}"
set -e
if [[ "$goal_status" -eq 124 ]]; then
  echo "NavigateToPose goal command timed out after $GOAL_TIMEOUT" | tee -a "$RUN_DIR/goal.log"
elif [[ "$goal_status" -ne 0 ]]; then
  echo "NavigateToPose goal command exited with status $goal_status" | tee -a "$RUN_DIR/goal.log"
fi

sleep 2
cleanup
set -e
trap - EXIT INT TERM

analysis_status=0
if [[ -x "$AUTONAV_REPO/scripts/run_lidar_line_bag_analysis.sh" ]]; then
  analysis_args=(--scenario-config "$resolved_course_config")
  if [[ "$STRICT_SCENARIO_GEOMETRY" == "1" ]]; then
    analysis_args+=(--strict-scenario-geometry)
  fi
  set +e
  "$AUTONAV_REPO/scripts/run_lidar_line_bag_analysis.sh" "$RUN_DIR/bag" "${analysis_args[@]}" \
    | tee "$RUN_DIR/analysis.log"
  analysis_status="${PIPESTATUS[0]}"
  set -e
else
  echo "Bag saved at $RUN_DIR/bag"
  echo "Analysis script not executable: $AUTONAV_REPO/scripts/run_lidar_line_bag_analysis.sh"
fi

if [[ "$goal_status" -ne 0 ]]; then
  exit "$goal_status"
fi
exit "$analysis_status"
