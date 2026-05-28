#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTONAV_REPO="${AUTONAV_REPO:-$HOME/code/git/AutoNav_25-26}"
BASE_RUN_DIR="${RUN_DIR:-$SCRIPT_DIR/ros_course_runs/suite_$(date +%Y%m%d_%H%M%S)}"
GROUND_TRUTH_PCA="${GROUND_TRUTH_PCA:-false}"
STRICT_SCENARIO_GEOMETRY="${STRICT_SCENARIO_GEOMETRY:-0}"
GOAL_TIMEOUT="${GOAL_TIMEOUT:-180s}"
STARTUP_WAIT_SEC="${STARTUP_WAIT_SEC:-8}"
PRE_GOAL_WAIT_SEC="${PRE_GOAL_WAIT_SEC:-6}"
SCENARIO_SETTLE_SEC="${SCENARIO_SETTLE_SEC:-8}"

DEFAULT_SCENARIOS=(
  canonical_5ft_gap
  open_10ft_lane_centering
  center_obstacle_dual_passage
  edge_obstacle_single_5ft_route
  narrow_decoy_gap_plus_legal_gap
  internal_line_no_cross
  minimum_turn_radius_curve
  canonical_5ft_gap_pose_offset
)

if [[ -n "${SCENARIOS:-}" ]]; then
  # shellcheck disable=SC2206
  scenario_list=($SCENARIOS)
else
  scenario_list=("${DEFAULT_SCENARIOS[@]}")
fi

mkdir -p "$BASE_RUN_DIR"

passed=()
failed=()
scenario_index=0
for scenario in "${scenario_list[@]}"; do
  if [[ "$scenario_index" -gt 0 ]]; then
    echo
    echo "Waiting ${SCENARIO_SETTLE_SEC}s for ROS graph cleanup before the next scenario."
    sleep "$SCENARIO_SETTLE_SEC"
  fi
  scenario_index=$((scenario_index + 1))

  echo
  echo "================================================================================"
  echo "Running lidar-line ROS scenario: $scenario"
  echo "Run directory: $BASE_RUN_DIR/$scenario"
  echo "================================================================================"
  if AUTONAV_REPO="$AUTONAV_REPO" \
     SCENARIO="$scenario" \
     RUN_DIR="$BASE_RUN_DIR/$scenario" \
     GROUND_TRUTH_PCA="$GROUND_TRUTH_PCA" \
     STRICT_SCENARIO_GEOMETRY="$STRICT_SCENARIO_GEOMETRY" \
     GOAL_TIMEOUT="$GOAL_TIMEOUT" \
     STARTUP_WAIT_SEC="$STARTUP_WAIT_SEC" \
     PRE_GOAL_WAIT_SEC="$PRE_GOAL_WAIT_SEC" \
     "$SCRIPT_DIR/Run_LIDAR_LINE_ROS_COURSE_TEST.command"; then
    passed+=("$scenario")
  else
    failed+=("$scenario")
  fi
done

echo
echo "================================================================================"
echo "Lidar-line ROS scenario suite summary"
echo "Run directory: $BASE_RUN_DIR"
echo "Passed (${#passed[@]}): ${passed[*]:-none}"
echo "Failed (${#failed[@]}): ${failed[*]:-none}"
echo "================================================================================"

if [[ "${#failed[@]}" -gt 0 ]]; then
  exit 1
fi
