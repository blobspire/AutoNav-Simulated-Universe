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

read -r default_goal_x default_goal_y < <(
  python3 - "$SCRIPT_DIR/config/lidar_line_course.yaml" <<'PY'
from pathlib import Path
import sys


def parse_scalar(raw):
    value = raw.split("#", 1)[0].strip()
    if not value:
        return ""
    try:
        return float(value)
    except ValueError:
        return value


values = {}
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or ":" not in stripped:
        continue
    key, raw = stripped.split(":", 1)
    values[key.strip()] = parse_scalar(raw)

goal_x = values.get("through_gap_goal_forward_m", values["goal_forward_m"])
goal_y = values["nominal_centerline_y_m"]
print(f"{goal_x} {goal_y}")
PY
)
GOAL_X="${GOAL_X:-$default_goal_x}"
GOAL_Y="${GOAL_Y:-$default_goal_y}"
GOAL_Z="${GOAL_Z:-0.0}"
GOAL_YAW_W="${GOAL_YAW_W:-1.0}"
GROUND_TRUTH_PCA="${GROUND_TRUTH_PCA:-false}"

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
  autonav_repo:="$AUTONAV_REPO" \
  ground_truth_pca:="$GROUND_TRUTH_PCA" &
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

echo "Sending NavigateToPose goal: x=$GOAL_X y=$GOAL_Y z=$GOAL_Z w=$GOAL_YAW_W ground_truth_pca=$GROUND_TRUTH_PCA"
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: $GOAL_X, y: $GOAL_Y, z: $GOAL_Z}, orientation: {w: $GOAL_YAW_W}}}}" \
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
