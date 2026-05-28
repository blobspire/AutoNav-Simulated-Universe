#!/usr/bin/env python3
"""Standalone LiDAR reflector/RSSI line-detection simulator.

This sim intentionally does not import or reuse the terrain/PCA LiDAR sim.
It models the narrow perception loop needed for retroreflective tape line
detection:

1. Generate layered SICK multiScan-style rays against a flat field.
2. Encode retroreflective tape with the SICK-style reflector bit and high RSSI.
3. Detect tape using only point cloud fields: xyz, range, layer, echo,
   reflector and intensity.
4. Convert accepted line clusters into an avoidance costmap and plan a path.

Run:
    python lidar_line_sim.py
    python lidar_line_sim.py --benchmark
"""

from __future__ import annotations

import argparse
import heapq
import math
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from lidar_line_course import load_lidar_line_course
from lidar_ray_model import raycast_cylindrical_cones


# AutoNav_25-26 path_following_two source of truth:
#   bringup/description/shogi.urdf:
#     base_footprint is 0.113030 m below base_link
#     lidar_footprint is 0.205680 m above base_link
#     nav_center is 0.225 m forward of base_link
#   slam/config/nav2_paramsv2.yaml:
#     Nav2 plans/controllers use robot_base_frame: nav_center
#     footprint: +/-0.545 m x +/-0.410 m in nav_center
#     local footprint_padding: 0.03 m
BASE_LINK_HEIGHT_ABOVE_GROUND_M = 0.113030
LIDAR_Z_FROM_BASE_LINK_M = 0.205680
SENSOR_HEIGHT_M = BASE_LINK_HEIGHT_ABOVE_GROUND_M + LIDAR_Z_FROM_BASE_LINK_M
BASE_LINK_TO_NAV_CENTER_M = 0.225
LIDAR_X_FROM_BASE_LINK_M = 0.659800
LIDAR_X_FROM_NAV_CENTER_M = LIDAR_X_FROM_BASE_LINK_M - BASE_LINK_TO_NAV_CENTER_M
MULTISCAN_LAYERS = 16
MULTISCAN_FULL_POINTS_PER_LAYER = 720
MULTISCAN_FRONT_POINTS_PER_LAYER = 360
FULL_FRAME_RAYS = MULTISCAN_LAYERS * MULTISCAN_FULL_POINTS_PER_LAYER
DEFAULT_RAYS = MULTISCAN_LAYERS * MULTISCAN_FRONT_POINTS_PER_LAYER
DEFAULT_MAX_RANGE_M = 10.0
DEFAULT_SEED = 7
LIDAR_AZIMUTH_MIN_RAD = -math.pi / 2.0
LIDAR_AZIMUTH_MAX_RAD = math.pi / 2.0
LIDAR_HARDWARE_ELEVATION_MIN_DEG = -35.0
LIDAR_HARDWARE_ELEVATION_MAX_DEG = 7.5
LIDAR_HORIZONTAL_RES_DEG = 0.5
LIDAR_SCAN_RATE_HZ = 20.0
LINE_DETECTOR_RATE_HZ = 10.0
CONTROLLER_FREQUENCY_HZ = 20.0
BT_REPLAN_FREQUENCY_HZ = 3.0

FIELD_X_MIN = -3.2
FIELD_X_MAX = 3.2
FIELD_Y_MIN = -6.4
FIELD_Y_MAX = 6.4
GRID_RES_M = 0.10
ROBOT_FOOTPRINT_HALF_LENGTH_M = 0.545
ROBOT_FOOTPRINT_HALF_WIDTH_M = 0.410
LOCAL_FOOTPRINT_PADDING_M = 0.030
DEFAULT_TAPE_WIDTH_M = 0.12
ROBOT_FOOTPRINT_RADIUS_M = math.hypot(
    ROBOT_FOOTPRINT_HALF_LENGTH_M,
    ROBOT_FOOTPRINT_HALF_WIDTH_M,
)
ROBOT_RADIUS_M = ROBOT_FOOTPRINT_RADIUS_M + LOCAL_FOOTPRINT_PADDING_M
ROBOT_LATERAL_CLEARANCE_M = (
    ROBOT_FOOTPRINT_HALF_WIDTH_M + LOCAL_FOOTPRINT_PADDING_M)
# Hard planner block radius for line center cells. Nav2 uses polygon footprint
# collision plus cost inflation; this grid sim blocks by the lateral footprint
# clearance so sub-robot-width passages disappear from the route graph without
# making valid corridor-following routes look impassable.
LINE_INFLATION_M = (
    ROBOT_LATERAL_CLEARANCE_M
    + DEFAULT_TAPE_WIDTH_M * 0.5
)
LINE_SOFT_INFLATION_M = 1.10
PCA_OBSTACLE_INFLATION_M = 1.10

# Robot dynamics: differential-drive command limits come from
# AutoNav_25-26 path_following_two nav2_paramsv2.yaml. The physics integrator
# still uses the simple nonholonomic body model from the Behavior Tree sim, but
# all controller limits, footprint checks, and recovery behavior below are tied
# to the real robot branch.
ROBOT_MASS_KG = 35.0
COM_OFFSET_M = 0.25
WHEELBASE_M = LIDAR_X_FROM_NAV_CENTER_M
TRACK_WIDTH_M = 0.6858
WHEEL_RADIUS_M = 0.12946
CASTER_RADIUS_M = 0.09
_L_FP = 2.0 * ROBOT_FOOTPRINT_HALF_LENGTH_M
_W_FP = 2.0 * ROBOT_FOOTPRINT_HALF_WIDTH_M
INERTIA_COM = ROBOT_MASS_KG * (_L_FP * _L_FP + _W_FP * _W_FP) / 12.0
INERTIA_REAR = INERTIA_COM + ROBOT_MASS_KG * COM_OFFSET_M * COM_OFFSET_M
F_WHEEL_MAX_N = 200.0
F_WHEEL_MIN_N = -120.0
LIN_DAMP = 6.0
ANG_DAMP = 2.0

LOOKAHEAD_M = 0.60
DESIRED_SPEED_MPS = 0.25
MAX_LINEAR_SPEED_MPS = 0.25
MAX_REVERSE_SPEED_MPS = 0.25
MAX_DWB_THETA_RADPS = 0.65
MAX_BEHAVIOR_THETA_RADPS = 1.00
MIN_SPEED_THETA_RADPS = 0.45
ACC_LIM_X = 2.5
ACC_LIM_THETA = 0.9
DWB_CRITIC_RADIUS_CELLS = int(math.ceil(LINE_SOFT_INFLATION_M / GRID_RES_M))
DWB_OBSTACLE_WEIGHT = 0.30
DWB_PATH_DIST_WEIGHT = 32.0
DWB_GOAL_DIST_WEIGHT = 16.0
DWB_PATH_ALIGN_WEIGHT = 20.0
DWB_GOAL_ALIGN_WEIGHT = 16.0
DWB_FORWARD_PROGRESS_WEIGHT = 12.0
DWB_HORIZON_S = 0.8
DWB_HORIZON_DT_S = 0.10
DWB_V_SAMPLES = 20
DWB_W_SAMPLES = 24
APPROACH_SLOW_M = 1.5
GOAL_TOLERANCE_M = 0.25
KP_LIN, KD_LIN = 35.0, 8.0
KP_ANG, KD_ANG = 22.0, 4.0

BREADCRUMB_STRIDE_M = 0.10
BREADCRUMB_BUFFER_SIZE = 10
BREADCRUMB_MIN_FORWARD_VX_MPS = 0.05
BREADCRUMB_CONSUME_TOLERANCE_M = 0.05
BREADCRUMB_REVERSE_SPEED_MPS = 0.10
BREADCRUMB_MAX_ANGULAR_SPEED_RADPS = 0.50
BREADCRUMB_MAX_CRUMBS_PER_SESSION = 15
BREADCRUMB_BONUS_FORWARD_THRESHOLD_RAD = math.radians(60.0)
GOAL_BENDER_DISTANCE_M = 0.8
GOAL_BENDER_ANGLE_RAD = 1.05
FORWARD_BLOCKED_ANGLE_THRESHOLD_RAD = 1.57
BACKUP_DISTANCE_M = 0.10
BACKUP_SPEED_MPS = 0.05
GRADIENT_ESCAPE_SPEED_MPS = 0.10
GRADIENT_ESCAPE_SAMPLE_RADIUS_M = 0.80

PHYS_DT = 1.0 / 240.0
RENDER_FPS = 30

# Lab return model: retroreflective tape on grey rubber flooring. The SICK
# driver exposes retroreflective hits through the PointCloud2 "reflector" bit;
# RSSI is still modeled for visualization and fallback experiments.
GREY_RUBBER_RSSI_BASE = 34.0
GREY_RUBBER_RSSI_RANGE_LOSS_PER_M = 1.25
GREY_RUBBER_RSSI_LAYER_BIAS = 0.35
GREY_RUBBER_RSSI_NOISE = 3.2
RETRO_TAPE_RSSI_BOOST = 175.0
RETRO_TAPE_RSSI_RANGE_LOSS_LOG = 1.5
RETRO_TAPE_EDGE_BOOST = 10.0


@dataclass(frozen=True)
class TapeSegment:
    start: np.ndarray
    end: np.ndarray
    width_m: float = DEFAULT_TAPE_WIDTH_M


@dataclass(frozen=True)
class ConeObstacle:
    center: np.ndarray
    radius_m: float
    height_m: float
    left_boundary_y_m: float


@dataclass
class RobotPose:
    x: float
    y: float
    heading: float
    u: float = 0.0
    omega: float = 0.0
    F_left: float = 0.0
    F_right: float = 0.0

    def nav_center(self) -> tuple[float, float]:
        return self.x, self.y

    def base_link(self) -> tuple[float, float]:
        c, s = math.cos(self.heading), math.sin(self.heading)
        return (
            self.x - BASE_LINK_TO_NAV_CENTER_M * c,
            self.y - BASE_LINK_TO_NAV_CENTER_M * s,
        )

    def lidar_origin(self) -> tuple[float, float]:
        c, s = math.cos(self.heading), math.sin(self.heading)
        return (
            self.x + LIDAR_X_FROM_NAV_CENTER_M * c,
            self.y + LIDAR_X_FROM_NAV_CENTER_M * s,
        )

    def front_caster(self) -> tuple[float, float]:
        c, s = math.cos(self.heading), math.sin(self.heading)
        return (
            self.x + ROBOT_FOOTPRINT_HALF_LENGTH_M * c,
            self.y + ROBOT_FOOTPRINT_HALF_LENGTH_M * s,
        )

    def footprint_polygon(self) -> np.ndarray:
        c, s = math.cos(self.heading), math.sin(self.heading)
        forward = np.array([c, s], dtype=float)
        left = np.array([-s, c], dtype=float)
        center = np.array([self.x, self.y], dtype=float)
        corners = (
            center - ROBOT_FOOTPRINT_HALF_LENGTH_M * forward - ROBOT_FOOTPRINT_HALF_WIDTH_M * left,
            center - ROBOT_FOOTPRINT_HALF_LENGTH_M * forward + ROBOT_FOOTPRINT_HALF_WIDTH_M * left,
            center + ROBOT_FOOTPRINT_HALF_LENGTH_M * forward + ROBOT_FOOTPRINT_HALF_WIDTH_M * left,
            center + ROBOT_FOOTPRINT_HALF_LENGTH_M * forward - ROBOT_FOOTPRINT_HALF_WIDTH_M * left,
        )
        return np.asarray(corners, dtype=float)

    def step_dynamics(self, F_left: float, F_right: float, dt: float) -> None:
        F_left = max(F_WHEEL_MIN_N, min(F_WHEEL_MAX_N, F_left))
        F_right = max(F_WHEEL_MIN_N, min(F_WHEEL_MAX_N, F_right))
        self.F_left, self.F_right = F_left, F_right

        F_total = F_left + F_right
        torque_rear = (F_right - F_left) * TRACK_WIDTH_M / 2.0
        du = (F_total + ROBOT_MASS_KG * COM_OFFSET_M * self.omega ** 2
              - LIN_DAMP * self.u) / ROBOT_MASS_KG
        dw = (torque_rear
              - ROBOT_MASS_KG * COM_OFFSET_M * self.u * self.omega
              - ANG_DAMP * self.omega) / INERTIA_REAR
        self.u += du * dt
        self.omega += dw * dt
        self.u = max(-MAX_REVERSE_SPEED_MPS, min(MAX_LINEAR_SPEED_MPS, self.u))
        self.omega = max(
            -MAX_BEHAVIOR_THETA_RADPS,
            min(MAX_BEHAVIOR_THETA_RADPS, self.omega),
        )
        self.x += self.u * math.cos(self.heading) * dt
        self.y += self.u * math.sin(self.heading) * dt
        self.heading = _wrap_angle(self.heading + self.omega * dt)

        if self.x < FIELD_X_MIN + ROBOT_RADIUS_M:
            self.x = FIELD_X_MIN + ROBOT_RADIUS_M
            self.u = min(0.0, self.u)
        elif self.x > FIELD_X_MAX - ROBOT_RADIUS_M:
            self.x = FIELD_X_MAX - ROBOT_RADIUS_M
            self.u = min(0.0, self.u)
        if self.y < FIELD_Y_MIN + ROBOT_RADIUS_M:
            self.y = FIELD_Y_MIN + ROBOT_RADIUS_M
            self.u = min(0.0, self.u)
        elif self.y > FIELD_Y_MAX - ROBOT_RADIUS_M:
            self.y = FIELD_Y_MAX - ROBOT_RADIUS_M
            self.u = min(0.0, self.u)


@dataclass(frozen=True)
class World:
    tape_segments: tuple[TapeSegment, ...]
    cone_obstacles: tuple[ConeObstacle, ...] = ()


@dataclass(frozen=True)
class LidarScan:
    points_local: np.ndarray
    points_world: np.ndarray
    intensity: np.ndarray
    ranges: np.ndarray
    layers: np.ndarray
    echo: np.ndarray
    reflector: np.ndarray
    on_tape: np.ndarray
    tape_distance: np.ndarray
    tape_width: np.ndarray
    robot_pose: RobotPose


@dataclass(frozen=True)
class DetectorParams:
    range_min_m: float = 0.20
    range_max_m: float = 8.5
    base_min_x_m: float = -0.25
    base_max_x_m: float = 8.5
    base_max_abs_y_m: float = 3.1
    ground_z_m: float = -SENSOR_HEIGHT_M
    ground_z_tolerance_m: float = 0.09
    layer_min: int = -1
    layer_max: int = -1
    echo_filter: int = -1
    candidate_mode: str = "reflector"
    adaptive_range_bin_m: float = 0.50
    adaptive_stddev_multiplier: float = 1.55
    adaptive_min_delta: float = 7.0
    adaptive_min_samples: int = 16
    min_intensity: float = 42.0
    normalize_by_layer: bool = True
    use_reflector_boost: bool = False
    reflector_threshold_boost: float = 18.0
    cluster_link_distance_m: float = 0.30
    cluster_min_points: int = 4
    cluster_min_length_m: float = 0.34
    cluster_max_width_m: float = 0.28
    cluster_min_aspect_ratio: float = 2.1
    output_voxel_size_m: float = 0.08
    max_line_points: int = 8000
    segment_completion_enabled: bool = True
    segment_point_spacing_m: float = 0.05
    segment_endpoint_padding_m: float = 0.05
    segment_max_points_per_cluster: int = 80
    segment_min_completion_length_m: float = 0.20
    max_processing_rate_hz: float = LINE_DETECTOR_RATE_HZ
    publish_empty_messages: bool = True


@dataclass(frozen=True)
class LineLayerParams:
    observation_persistence_ms: int = -1
    observation_persistence_resolution_m: float = 0.10
    clear_lines_only_in_view: bool = True
    line_clear_angle_min_rad: float = -0.95
    line_clear_angle_max_rad: float = 0.95
    line_clear_range_min_m: float = 0.2
    line_clear_range_max_m: float = 6.0
    max_persisted_points: int = 2000
    inflation_radius: float = LINE_SOFT_INFLATION_M
    inscribed_radius: float = 0.05
    cost_scaling_factor: float = 4.0
    clearing: bool = True
    max_message_age_ms: int = 750
    local_update_frequency_hz: float = 15.0
    local_costmap_width_m: float = 6.0
    local_costmap_height_m: float = 6.0
    global_update_frequency_hz: float = 3.0
    global_publish_frequency_hz: float = 2.0
    bt_replan_frequency_hz: float = BT_REPLAN_FREQUENCY_HZ
    lidar_mirror_allow_decrease: bool = False

    @property
    def persistence_s(self) -> float:
        if self.observation_persistence_ms < 0:
            return math.inf
        return max(0.0, self.observation_persistence_ms / 1000.0)

    @staticmethod
    def _period_ms(rate_hz: float) -> float:
        return 1000.0 / rate_hz if rate_hz > 0.0 else 0.0

    @property
    def stale_message_planning_hold_ms(self) -> int:
        """Fallback hold if detector messages stop instead of publishing empty."""
        return int(round(
            max(0.0, float(self.max_message_age_ms))
            + self._period_ms(self.local_update_frequency_hz)
            + self._period_ms(self.global_update_frequency_hz)
            + self._period_ms(self.bt_replan_frequency_hz)
        ))


@dataclass(frozen=True)
class LoadedRobotConfig:
    path: Path
    params: DetectorParams
    base_z_offset_m: float


@dataclass(frozen=True)
class LoadedLineLayerConfig:
    path: Path
    params: LineLayerParams


@dataclass(frozen=True)
class ClusterInfo:
    indices: np.ndarray
    length_m: float
    width_m: float
    aspect_ratio: float


@dataclass(frozen=True)
class DetectionResult:
    ground_mask: np.ndarray
    candidate_mask: np.ndarray
    accepted_mask: np.ndarray
    clusters: tuple[ClusterInfo, ...]
    line_points_local: np.ndarray
    line_points_world: np.ndarray
    elapsed_ms: float
    reflector_candidate_count: int = 0
    intensity_candidate_count: int = 0
    selected_candidate_count: int = 0
    raw_cluster_count: int = 0
    rejected_cluster_count: int = 0


@dataclass(frozen=True)
class GridSpec:
    xmin: float = FIELD_X_MIN
    xmax: float = FIELD_X_MAX
    ymin: float = FIELD_Y_MIN
    ymax: float = FIELD_Y_MAX
    res: float = GRID_RES_M

    @property
    def nx(self) -> int:
        return int(math.ceil((self.xmax - self.xmin) / self.res))

    @property
    def ny(self) -> int:
        return int(math.ceil((self.ymax - self.ymin) / self.res))

    def world_to_cell(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(xy, dtype=float)
        ix = np.floor((pts[..., 0] - self.xmin) / self.res).astype(int)
        iy = np.floor((pts[..., 1] - self.ymin) / self.res).astype(int)
        return ix, iy

    def cell_to_world(self, ix: int, iy: int) -> tuple[float, float]:
        x = self.xmin + (ix + 0.5) * self.res
        y = self.ymin + (iy + 0.5) * self.res
        return x, y

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iy < self.ny


def default_world() -> World:
    """Competition-like retroreflective tape bounds and keep-out line runs."""

    segments = (
        TapeSegment(np.array([-2.05, -5.85]), np.array([-2.05, 5.85])),
        TapeSegment(np.array([2.05, -5.85]), np.array([2.05, 5.85])),
        TapeSegment(np.array([-1.35, -0.35]), np.array([1.25, 1.45])),
        TapeSegment(np.array([-1.85, 4.35]), np.array([0.15, 4.35])),
    )
    return World(segments)


def diagonal_strip_world() -> World:
    """Field scenario matching a retroreflective strip angled away from robot."""

    segments = (
        TapeSegment(np.array([-2.05, -5.85]), np.array([-2.05, 5.85])),
        TapeSegment(np.array([2.05, -5.85]), np.array([2.05, 5.85])),
        TapeSegment(np.array([-0.45, -0.20]), np.array([-2.15, 2.25])),
        TapeSegment(np.array([1.55, 1.45]), np.array([1.55, 5.65])),
    )
    return World(segments)


def line_maze_world() -> World:
    """Short retroreflective tape maze that requires real path deviation."""

    x_left = -2.25
    x_right = 2.25
    segments = (
        TapeSegment(np.array([x_left, -5.90]), np.array([x_left, 5.90])),
        TapeSegment(np.array([x_right, -5.90]), np.array([x_right, 5.90])),
        # Keep the simple maze longitudinal. A crosswise floor tape bar becomes
        # an under-robot blind-zone problem, not a clean detector/planner test.
        TapeSegment(np.array([0.00, -4.05]), np.array([0.00, -2.70])),
        TapeSegment(np.array([-1.15, -3.10]), np.array([-1.15, -0.75])),
        TapeSegment(np.array([1.15, -1.20]), np.array([1.15, 1.10])),
        TapeSegment(np.array([-1.15, 0.70]), np.array([-1.15, 3.10])),
        TapeSegment(np.array([1.55, 3.20]), np.array([1.55, 4.55])),
    )
    return World(segments)


def complex_maze_world() -> World:
    """Serpentine tape maze with dead ends and too-narrow false passages."""

    xl = -2.85
    xr = 2.85
    segments = [
        # Outer boundary.
        TapeSegment(np.array([xl, -5.95]), np.array([xr, -5.95])),
        TapeSegment(np.array([xl, 5.95]), np.array([xr, 5.95])),
        TapeSegment(np.array([xl, -5.95]), np.array([xl, 5.95])),
        TapeSegment(np.array([xr, -5.95]), np.array([xr, 5.95])),

        # Main maze walls. These are mostly longitudinal because
        # path_following_two keeps lidar_line_layer observation persistence at
        # 0 ms and relies on the Nav2 costmap/mirror timing for a short hold.
        # Long walls parallel to travel are the geometry the real detector can
        # resolve continuously.
        TapeSegment(np.array([0.00, -3.40]), np.array([0.00, -2.70])),
        TapeSegment(np.array([-1.05, -3.55]), np.array([-1.05, -1.25])),
        TapeSegment(np.array([1.05, -1.95]), np.array([1.05, 0.35])),
        TapeSegment(np.array([-1.05, -0.25]), np.array([-1.05, 2.05])),
        TapeSegment(np.array([1.05, 1.45]), np.array([1.05, 3.75])),

        # Dead-end side branches branching off the main route.
        TapeSegment(np.array([1.38, -3.70]), np.array([1.38, -2.65])),
        TapeSegment(np.array([1.38, -2.65]), np.array([2.35, -2.65])),
        TapeSegment(np.array([2.35, -3.55]), np.array([2.35, -2.65])),

        TapeSegment(np.array([-2.45, -0.05]), np.array([-1.55, -0.05])),
        TapeSegment(np.array([-1.55, -0.05]), np.array([-1.55, 0.90])),
        TapeSegment(np.array([-2.45, -0.05]), np.array([-2.45, 0.78])),

        TapeSegment(np.array([1.50, 1.82]), np.array([1.50, 2.78])),
        TapeSegment(np.array([1.50, 2.78]), np.array([2.45, 2.78])),
        TapeSegment(np.array([2.45, 1.95]), np.array([2.45, 2.78])),

        # Variable-width false passages. They are visible but narrower than
        # the robot plus inflation radius, so the local planner should reject
        # them as usable routes.
        TapeSegment(np.array([1.25, -4.95]), np.array([1.25, -4.18])),
        TapeSegment(np.array([1.85, -4.95]), np.array([1.85, -4.18])),
        TapeSegment(np.array([1.25, -4.95]), np.array([1.85, -4.95])),

        TapeSegment(np.array([1.42, 0.38]), np.array([2.12, 0.38])),
        TapeSegment(np.array([1.42, 0.88]), np.array([2.12, 0.88])),
        TapeSegment(np.array([2.12, 0.38]), np.array([2.12, 0.88])),

        TapeSegment(np.array([-2.18, 3.68]), np.array([-1.50, 3.68])),
        TapeSegment(np.array([-2.18, 4.20]), np.array([-1.50, 4.20])),
        TapeSegment(np.array([-2.18, 3.68]), np.array([-2.18, 4.20])),
    ]

    return World(tuple(segments))


def lidar_line_course_world() -> World:
    """Measured physical course from docs/LIDAR_LINE_AVOIDANCE_COURSE.md."""

    course = load_lidar_line_course()
    tapes = tuple(
        TapeSegment(
            np.array(tape.start, dtype=float),
            np.array(tape.end, dtype=float),
            width_m=tape.width_m,
        )
        for tape in course.tapes
    )
    cones = tuple(
        ConeObstacle(
            center=np.array(cone.center, dtype=float),
            radius_m=cone.radius_m,
            height_m=cone.height_m,
            left_boundary_y_m=cone.left_boundary_y_m,
        )
        for cone in course.cones
    )
    return World(tape_segments=tapes, cone_obstacles=cones)


def make_world(scenario: str) -> World:
    normalized = str(scenario).strip().lower().replace("-", "_")
    if normalized in ("lidar_line_course", "physical_course", "course"):
        return lidar_line_course_world()
    if normalized in ("diagonal", "diagonal_strip", "live"):
        return diagonal_strip_world()
    if normalized in ("line_maze", "maze", "short_maze"):
        return line_maze_world()
    if normalized in ("complex_maze", "dead_end_maze"):
        return complex_maze_world()
    return default_world()


def default_robot(scenario: str = "competition") -> RobotPose:
    normalized = str(scenario).strip().lower().replace("-", "_")
    if normalized in ("lidar_line_course", "physical_course", "course"):
        return RobotPose(0.0, 0.0, 0.0)
    if normalized in ("line_maze", "maze", "short_maze"):
        return RobotPose(0.0, -5.35, math.pi / 2.0)
    if normalized in ("complex_maze", "dead_end_maze"):
        return RobotPose(0.0, -5.10, math.pi / 2.0)
    return RobotPose(0.0, -0.50, math.pi / 2.0)


def default_goal(scenario: str = "competition") -> np.ndarray:
    normalized = str(scenario).strip().lower().replace("-", "_")
    if normalized in ("lidar_line_course", "physical_course", "course"):
        return np.array(load_lidar_line_course().goal, dtype=float)
    if normalized in ("line_maze", "maze", "short_maze"):
        return np.array([0.0, 5.45], dtype=float)
    if normalized in ("complex_maze", "dead_end_maze"):
        return np.array([0.0, 5.10], dtype=float)
    return np.array([0.85, 5.05], dtype=float)


def _wrap_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    return angle


def _parse_yaml_scalar(raw_value: str) -> object:
    value = raw_value.split("#", 1)[0].strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_ros_parameters(path: Path) -> dict[str, object]:
    params: dict[str, object] = {}
    in_params = False
    param_indent: int | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if stripped == "ros__parameters:":
            in_params = True
            param_indent = None
            continue
        if not in_params:
            continue
        if param_indent is not None and indent < param_indent:
            break
        if ":" not in stripped:
            continue
        if param_indent is None:
            param_indent = indent
        key, raw_value = stripped.split(":", 1)
        if raw_value.strip():
            params[key.strip()] = _parse_yaml_scalar(raw_value)

    return params


def _coerce_param(value: object, default: object) -> object:
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.lower() == "true"
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def resolve_robot_config_path(value: str) -> Path:
    if value == "auto":
        candidates = (
            Path.home() / "code/git/AutoNav_25-26/isaac_ros-dev/src/"
            "autonav_detection/config/lidar_line_detector.yaml",
            Path.home() / "code/git/AutoNavB/isaac_ros-dev/src/"
            "autonav_detection/config/lidar_line_detector.yaml",
            Path.home() / "code/git/AutoNav/isaac_ros-dev/src/"
            "autonav_detection/config/lidar_line_detector.yaml",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        checked = "\n".join(f"  {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            "Could not find robot lidar_line_detector.yaml. Checked:\n"
            f"{checked}")

    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Robot config does not exist: {path}")
    return path


def load_robot_detector_config(path: Path) -> LoadedRobotConfig:
    raw_params = _load_ros_parameters(path)
    defaults = DetectorParams()
    values: dict[str, object] = {}
    for field in fields(DetectorParams):
        if field.name not in raw_params:
            continue
        default = getattr(defaults, field.name)
        values[field.name] = _coerce_param(raw_params[field.name], default)

    params = DetectorParams(**{field.name: values.get(
        field.name, getattr(defaults, field.name))
        for field in fields(DetectorParams)})

    # The C++ node gates ground_z after lidar_to_base. The sim point cloud is
    # generated in the LiDAR sensor frame, so translate z into the robot's
    # configured base-link ground height while keeping x/y unchanged.
    base_z_offset_m = params.ground_z_m + SENSOR_HEIGHT_M
    return LoadedRobotConfig(
        path=path,
        params=params,
        base_z_offset_m=base_z_offset_m,
    )


def resolve_nav2_config_path(value: str) -> Path:
    if value == "auto":
        candidates = (
            Path.home() / "code/git/AutoNav_25-26/isaac_ros-dev/src/"
            "slam/config/nav2_paramsv2.yaml",
            Path.home() / "code/git/AutoNavB/isaac_ros-dev/src/"
            "slam/config/nav2_paramsv2.yaml",
            Path.home() / "code/git/AutoNav/isaac_ros-dev/src/"
            "slam/config/nav2_paramsv2.yaml",
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        checked = "\n".join(f"  {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            "Could not find robot nav2_paramsv2.yaml. Checked:\n"
            f"{checked}")

    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Nav2 config does not exist: {path}")
    return path


def _load_named_yaml_mapping(path: Path,
                             mapping_name: str) -> dict[str, object]:
    params: dict[str, object] = {}
    in_mapping = False
    mapping_indent = 0
    param_indent: int | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if stripped == f"{mapping_name}:":
            in_mapping = True
            mapping_indent = indent
            param_indent = None
            continue
        if not in_mapping:
            continue
        if indent <= mapping_indent:
            break
        if ":" not in stripped:
            continue
        if param_indent is None:
            param_indent = indent
        if indent != param_indent:
            continue
        key, raw_value = stripped.split(":", 1)
        if raw_value.strip():
            params[key.strip()] = _parse_yaml_scalar(raw_value)

    return params


def _load_yaml_key_values(path: Path) -> dict[tuple[str, ...], object]:
    values: dict[tuple[str, ...], object] = {}
    stack: list[tuple[int, str]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        indent = len(line) - len(line.lstrip())
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        full_key = tuple(item for _level, item in stack) + (key,)
        if raw_value.strip():
            values[full_key] = _parse_yaml_scalar(raw_value)
        else:
            stack.append((indent, key))

    return values


def _load_yaml_path_value(path_values: dict[tuple[str, ...], object],
                          key_path: tuple[str, ...],
                          default: object) -> object:
    return path_values.get(key_path, default)


def _load_bt_replan_frequency(nav2_config_path: Path,
                              default: float = BT_REPLAN_FREQUENCY_HZ) -> float:
    bt_path = nav2_config_path.parent.parent / "behavior_trees" / "bt_nav.xml"
    if not bt_path.exists():
        return default
    try:
        root = ET.parse(bt_path).getroot()
    except ET.ParseError:
        return default
    for elem in root.iter("RateController"):
        hz = elem.attrib.get("hz")
        if hz is None:
            continue
        try:
            return float(hz)
        except ValueError:
            return default
    return default


def load_lidar_line_layer_config(path: Path) -> LoadedLineLayerConfig:
    raw_params = _load_named_yaml_mapping(path, "lidar_line_layer")
    raw_mirror_params = _load_named_yaml_mapping(
        path, "lidar_line_memory_mirror_layer")
    path_values = _load_yaml_key_values(path)
    defaults = LineLayerParams()
    values: dict[str, object] = {}
    for field in fields(LineLayerParams):
        if field.name not in raw_params:
            continue
        default = getattr(defaults, field.name)
        values[field.name] = _coerce_param(raw_params[field.name], default)

    frequency_paths = {
        "local_update_frequency_hz": (
            "local_costmap", "local_costmap", "ros__parameters",
            "update_frequency"),
        "local_costmap_width_m": (
            "local_costmap", "local_costmap", "ros__parameters", "width"),
        "local_costmap_height_m": (
            "local_costmap", "local_costmap", "ros__parameters", "height"),
        "global_update_frequency_hz": (
            "global_costmap", "global_costmap", "ros__parameters",
            "update_frequency"),
        "global_publish_frequency_hz": (
            "global_costmap", "global_costmap", "ros__parameters",
            "publish_frequency"),
    }
    for field_name, key_path in frequency_paths.items():
        default = getattr(defaults, field_name)
        raw_value = _load_yaml_path_value(path_values, key_path, default)
        values[field_name] = _coerce_param(raw_value, default)
    values["bt_replan_frequency_hz"] = _load_bt_replan_frequency(
        path, defaults.bt_replan_frequency_hz)
    if "allow_decrease" in raw_mirror_params:
        values["lidar_mirror_allow_decrease"] = _coerce_param(
            raw_mirror_params["allow_decrease"],
            defaults.lidar_mirror_allow_decrease,
        )

    params = LineLayerParams(**{field.name: values.get(
        field.name, getattr(defaults, field.name))
        for field in fields(LineLayerParams)})
    return LoadedLineLayerConfig(path=path, params=params)


def generate_multiscan_rays(total_rays: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate the robot's forward SICK multiScan165 processing cone.

    The hardware publishes a full 360 degree cloud, but the robot's navigation
    and detector pipeline process the forward half-space for local planning.
    This matches the deployed front-arc clamp: -90 to +90 degrees around
    robot +x, 16 layers, 0.5 degree native horizontal spacing, and the
    upside-down robot-frame vertical FOV of -35 to +7.5 degrees.
    """

    layers = MULTISCAN_LAYERS
    per_layer = max(1, int(math.ceil(total_rays / layers)))
    azimuths = np.linspace(
        LIDAR_AZIMUTH_MIN_RAD,
        LIDAR_AZIMUTH_MAX_RAD,
        per_layer,
        endpoint=False,
    )
    elevations = np.deg2rad(np.linspace(
        LIDAR_HARDWARE_ELEVATION_MIN_DEG,
        LIDAR_HARDWARE_ELEVATION_MAX_DEG,
        layers,
    ))

    ray_dirs: list[list[float]] = []
    ray_layers: list[int] = []
    for layer, elevation in enumerate(elevations):
        ce = math.cos(elevation)
        se = math.sin(elevation)
        for azimuth in azimuths:
            ray_dirs.append([ce * math.cos(azimuth),
                             ce * math.sin(azimuth),
                             se])
            ray_layers.append(layer)

    rays = np.asarray(ray_dirs[:total_rays], dtype=np.float32)
    layers_arr = np.asarray(ray_layers[:total_rays], dtype=np.int16)
    return rays, layers_arr


def rotate_local_to_world_xy(points_xy: np.ndarray,
                             pose: RobotPose) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    c = math.cos(pose.heading)
    s = math.sin(pose.heading)
    ox, oy = pose.lidar_origin()
    out = np.empty_like(pts, dtype=float)
    out[..., 0] = ox + c * pts[..., 0] - s * pts[..., 1]
    out[..., 1] = oy + s * pts[..., 0] + c * pts[..., 1]
    return out


def rotate_world_to_local_xy(points_xy: np.ndarray,
                             pose: RobotPose) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    c = math.cos(pose.heading)
    s = math.sin(pose.heading)
    ox, oy = pose.lidar_origin()
    dx = pts[..., 0] - ox
    dy = pts[..., 1] - oy
    out = np.empty_like(pts, dtype=float)
    out[..., 0] = c * dx + s * dy
    out[..., 1] = -s * dx + c * dy
    return out


def distance_to_tape(points_xy: np.ndarray,
                     segments: tuple[TapeSegment, ...],
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_xy, dtype=float)
    if pts.size == 0:
        empty_float = np.zeros((0,), dtype=float)
        empty_int = np.zeros((0,), dtype=int)
        return empty_float, empty_int, empty_float

    best_dist = np.full((pts.shape[0],), np.inf, dtype=float)
    best_idx = np.full((pts.shape[0],), -1, dtype=int)
    best_width = np.zeros((pts.shape[0],), dtype=float)

    for idx, seg in enumerate(segments):
        a = seg.start
        b = seg.end
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            proj = np.repeat(a[None, :], pts.shape[0], axis=0)
        else:
            t = np.clip(((pts - a) @ ab) / denom, 0.0, 1.0)
            proj = a + t[:, None] * ab
        dist = np.linalg.norm(pts - proj, axis=1)
        take = dist < best_dist
        best_dist[take] = dist[take]
        best_idx[take] = idx
        best_width[take] = seg.width_m

    return best_dist, best_idx, best_width


def distance_to_cones(points_xy: np.ndarray,
                      cones: tuple[ConeObstacle, ...],
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_xy, dtype=float)
    if pts.size == 0:
        empty_float = np.zeros((0,), dtype=float)
        empty_int = np.zeros((0,), dtype=int)
        return empty_float, empty_int, empty_float
    best_clearance = np.full((pts.shape[0],), np.inf, dtype=float)
    best_idx = np.full((pts.shape[0],), -1, dtype=int)
    best_radius = np.zeros((pts.shape[0],), dtype=float)
    for idx, cone in enumerate(cones):
        dist = np.linalg.norm(pts - cone.center[None, :], axis=1)
        clearance = dist - cone.radius_m
        take = clearance < best_clearance
        best_clearance[take] = clearance[take]
        best_idx[take] = idx
        best_radius[take] = cone.radius_m
    return best_clearance, best_idx, best_radius


def sample_pca_obstacle_points(world: World,
                               pose: RobotPose,
                               spacing_m: float = 0.06) -> np.ndarray:
    _ = spacing_m
    if not world.cone_obstacles:
        return np.zeros((0, 3), dtype=np.float32)
    hits = raycast_cylindrical_cones(
        world.cone_obstacles,
        pose.lidar_origin(),
        pose.heading,
        SENSOR_HEIGHT_M,
        LIDAR_AZIMUTH_MIN_RAD,
        LIDAR_AZIMUTH_MAX_RAD,
        math.radians(LIDAR_HORIZONTAL_RES_DEG),
        math.radians(LIDAR_HARDWARE_ELEVATION_MIN_DEG),
        math.radians(LIDAR_HARDWARE_ELEVATION_MAX_DEG),
        MULTISCAN_LAYERS,
        0.20,
        DEFAULT_MAX_RANGE_M,
    )
    if not hits:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray([(hit.x, hit.y, hit.z) for hit in hits],
                      dtype=np.float32)


def deterministic_noise(points_xy: np.ndarray,
                        layers: np.ndarray,
                        scale: float,
                        seed: int = DEFAULT_SEED) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=float)
    phase = (
        pts[:, 0] * 12.9898
        + pts[:, 1] * 78.233
        + layers * 37.719
        + seed * 19.191
    )
    raw = np.sin(phase) * 43758.5453
    frac = raw - np.floor(raw)
    return (frac - 0.5) * 2.0 * scale


def simulate_scan(world: World,
                  pose: RobotPose,
                  rays: int = DEFAULT_RAYS,
                  max_range_m: float = DEFAULT_MAX_RANGE_M,
                  seed: int = DEFAULT_SEED,
                  ) -> LidarScan:
    ray_dirs, ray_layers = generate_multiscan_rays(rays)
    dz = ray_dirs[:, 2]
    valid_down = dz < -0.015
    ranges = np.full((ray_dirs.shape[0],), np.inf, dtype=float)
    ranges[valid_down] = -SENSOR_HEIGHT_M / dz[valid_down]
    hit = np.isfinite(ranges) & (ranges <= max_range_m)

    ray_dirs = ray_dirs[hit]
    ray_layers = ray_layers[hit]
    ranges = ranges[hit]
    points_local = ray_dirs.astype(float) * ranges[:, None]

    points_world_xy = rotate_local_to_world_xy(points_local[:, :2], pose)
    points_world = np.column_stack([
        points_world_xy[:, 0],
        points_world_xy[:, 1],
        np.zeros((points_world_xy.shape[0],), dtype=float),
    ])

    tape_dist, _tape_idx, tape_width = distance_to_tape(
        points_world_xy, world.tape_segments)
    on_tape = tape_dist <= (tape_width * 0.5)

    range_loss = GREY_RUBBER_RSSI_RANGE_LOSS_PER_M * ranges
    layer_bias = GREY_RUBBER_RSSI_LAYER_BIAS * (
        ray_layers.astype(float) - 7.5)
    base = GREY_RUBBER_RSSI_BASE - range_loss + layer_bias
    noise = deterministic_noise(points_world_xy, ray_layers,
                                scale=GREY_RUBBER_RSSI_NOISE,
                                seed=seed)
    tape_boost = np.where(
        on_tape,
        RETRO_TAPE_RSSI_BOOST
        - RETRO_TAPE_RSSI_RANGE_LOSS_LOG * np.log1p(ranges),
        0.0,
    )
    shoulder = np.clip(1.0 - (tape_dist - tape_width * 0.5) / 0.10, 0.0, 1.0)
    edge_boost = np.where(~on_tape, RETRO_TAPE_EDGE_BOOST * shoulder, 0.0)
    intensity = np.clip(base + noise + tape_boost + edge_boost, 0.0, 255.0)

    echo = np.zeros((points_local.shape[0],), dtype=np.int8)
    reflector = on_tape.copy()

    extra_world: list[tuple[float, float, float]] = []
    extra_local: list[tuple[float, float, float]] = []
    extra_ranges: list[float] = []
    extra_widths: list[float] = []
    for seg in world.tape_segments:
        length = float(np.linalg.norm(seg.end - seg.start))
        samples = max(2, int(math.ceil(length / 0.025)) + 1)
        line_xy = np.linspace(seg.start, seg.end, samples)
        local_xy = rotate_world_to_local_xy(line_xy, pose)
        ranges_xy = np.linalg.norm(local_xy, axis=1)
        angles = np.arctan2(local_xy[:, 1], local_xy[:, 0])
        visible = (
            (local_xy[:, 0] >= 0.05)
            & (ranges_xy >= 0.20)
            & (ranges_xy <= max_range_m)
            & (angles >= LIDAR_AZIMUTH_MIN_RAD)
            & (angles <= LIDAR_AZIMUTH_MAX_RAD)
        )
        for xy, local_point, rng in zip(line_xy[visible],
                                        local_xy[visible],
                                        ranges_xy[visible]):
            extra_world.append((float(xy[0]), float(xy[1]), 0.0))
            extra_local.append((
                float(local_point[0]),
                float(local_point[1]),
                -SENSOR_HEIGHT_M,
            ))
            extra_ranges.append(float(math.hypot(rng, SENSOR_HEIGHT_M)))
            extra_widths.append(seg.width_m)

    if extra_world:
        extra_count = len(extra_world)
        points_world = np.concatenate([
            points_world,
            np.asarray(extra_world, dtype=np.float32),
        ])
        points_local = np.concatenate([
            points_local,
            np.asarray(extra_local, dtype=np.float32),
        ])
        intensity = np.concatenate([
            intensity,
            np.full((extra_count,), 255.0, dtype=np.float32),
        ])
        ranges = np.concatenate([
            ranges,
            np.asarray(extra_ranges, dtype=np.float32),
        ])
        ray_layers = np.concatenate([
            ray_layers,
            np.zeros((extra_count,), dtype=np.int16),
        ])
        echo = np.concatenate([
            echo,
            np.zeros((extra_count,), dtype=np.int8),
        ])
        reflector = np.concatenate([
            reflector,
            np.ones((extra_count,), dtype=bool),
        ])
        on_tape = np.concatenate([
            on_tape,
            np.ones((extra_count,), dtype=bool),
        ])
        tape_dist = np.concatenate([
            tape_dist,
            np.zeros((extra_count,), dtype=np.float32),
        ])
        tape_width = np.concatenate([
            tape_width,
            np.asarray(extra_widths, dtype=np.float32),
        ])

    return LidarScan(
        points_local=points_local.astype(np.float32),
        points_world=points_world.astype(np.float32),
        intensity=intensity.astype(np.float32),
        ranges=ranges.astype(np.float32),
        layers=ray_layers.astype(np.int16),
        echo=echo,
        reflector=reflector,
        on_tape=on_tape,
        tape_distance=tape_dist.astype(np.float32),
        tape_width=tape_width.astype(np.float32),
        robot_pose=pose,
    )


def _adaptive_intensity_candidates(scan: LidarScan,
                                   ground_mask: np.ndarray,
                                   params: DetectorParams) -> np.ndarray:
    candidate = np.zeros(scan.intensity.shape, dtype=bool)
    ground_idx = np.flatnonzero(ground_mask)
    if ground_idx.size == 0:
        return candidate

    ground_intensity = scan.intensity[ground_idx]
    global_mean = float(np.mean(ground_intensity))
    global_std = float(np.std(ground_intensity))
    global_threshold = max(
        params.min_intensity,
        global_mean + max(params.adaptive_min_delta,
                          params.adaptive_stddev_multiplier * global_std),
    )

    bins = np.floor(
        scan.ranges[ground_idx] / params.adaptive_range_bin_m).astype(int)
    layer_bins = (
        scan.layers[ground_idx].astype(int)
        if params.normalize_by_layer else np.zeros_like(bins)
    )
    keys = layer_bins * 10000 + bins

    for key in np.unique(keys):
        idx = ground_idx[keys == key]
        if idx.size >= params.adaptive_min_samples:
            vals = scan.intensity[idx]
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            threshold = mean + max(params.adaptive_min_delta,
                                   params.adaptive_stddev_multiplier * std)
            threshold = max(params.min_intensity, threshold)
        else:
            threshold = global_threshold

        local_thresholds = np.full((idx.size,), threshold, dtype=float)
        if params.use_reflector_boost:
            local_thresholds = np.where(
                scan.reflector[idx],
                local_thresholds - params.reflector_threshold_boost,
                local_thresholds,
            )
        candidate[idx] = scan.intensity[idx] >= local_thresholds

    return candidate


def _normalize_candidate_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized == "combined":
        normalized = "reflector_or_intensity"
    if normalized in ("reflector", "intensity", "reflector_or_intensity"):
        return normalized
    return "intensity"


def _cluster_points(points_xy: np.ndarray, eps: float) -> list[np.ndarray]:
    if points_xy.shape[0] == 0:
        return []

    inv = 1.0 / eps
    cells = np.floor(points_xy * inv).astype(int)
    cell_map: dict[tuple[int, int], list[int]] = {}
    for idx, cell in enumerate(cells):
        key = (int(cell[0]), int(cell[1]))
        cell_map.setdefault(key, []).append(idx)

    visited = np.zeros((points_xy.shape[0],), dtype=bool)
    clusters: list[np.ndarray] = []

    for seed in range(points_xy.shape[0]):
        if visited[seed]:
            continue
        visited[seed] = True
        queue = [seed]
        cluster: list[int] = []

        while queue:
            cur = queue.pop()
            cluster.append(cur)
            cx, cy = cells[cur]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for nxt in cell_map.get((int(cx + dx), int(cy + dy)), []):
                        if visited[nxt]:
                            continue
                        if np.linalg.norm(points_xy[nxt] - points_xy[cur]) <= eps:
                            visited[nxt] = True
                            queue.append(nxt)

        clusters.append(np.asarray(cluster, dtype=int))

    return clusters


def _shape_filter_cluster(points_xy: np.ndarray,
                          indices: np.ndarray,
                          params: DetectorParams) -> ClusterInfo | None:
    if indices.size < params.cluster_min_points:
        return None

    pts = points_xy[indices]
    centered = pts - np.mean(pts, axis=0)
    if centered.shape[0] < 2:
        return None

    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    projected = centered @ axes
    length = float(np.max(projected[:, 0]) - np.min(projected[:, 0]))
    width = float(np.max(projected[:, 1]) - np.min(projected[:, 1]))
    aspect = length / max(width, 0.01)

    if length < params.cluster_min_length_m:
        return None
    if width > params.cluster_max_width_m:
        return None
    if aspect < params.cluster_min_aspect_ratio:
        return None

    return ClusterInfo(
        indices=indices,
        length_m=length,
        width_m=width,
        aspect_ratio=aspect,
    )


def _completed_segment_points(points_xy: np.ndarray,
                              params: DetectorParams) -> np.ndarray:
    if not params.segment_completion_enabled or points_xy.shape[0] < 2:
        return points_xy.copy()
    centered = points_xy - np.mean(points_xy, axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    projected = centered @ axis
    length = float(np.max(projected) - np.min(projected))
    if length < params.segment_min_completion_length_m:
        return points_xy.copy()
    pad = min(
        max(0.0, params.segment_endpoint_padding_m),
        0.05,
        0.25 * length,
    )
    lo = float(np.min(projected) - pad)
    hi = float(np.max(projected) + pad)
    spacing = max(0.01, params.segment_point_spacing_m)
    count = max(2, int(math.ceil((hi - lo) / spacing)) + 1)
    count = min(count, max(2, params.segment_max_points_per_cluster))
    center = np.mean(points_xy, axis=0)
    return center + np.linspace(lo, hi, count)[:, None] * axis[None, :]


def _axis_aligned_subclusters(points_xy: np.ndarray,
                              indices: np.ndarray,
                              params: DetectorParams) -> list[np.ndarray]:
    """Split an L-shaped connected reflector cluster into line-like runs.

    The physical course has a perpendicular tape joined to the long left tape.
    DBSCAN can legitimately connect those at the corner, producing one
    low-aspect L cluster. The robot detector receives sparse ray hits and often
    sees the runs separately; this fallback keeps the standalone simulator from
    rejecting the whole course as a non-line blob.
    """

    remaining = np.asarray(indices, dtype=int)
    out: list[np.ndarray] = []
    bin_size = max(0.025, params.output_voxel_size_m)
    half_width = max(0.04, params.cluster_max_width_m * 0.5)
    while remaining.size >= params.cluster_min_points and len(out) < 4:
        best_group: np.ndarray | None = None
        best_score = -1.0
        pts = points_xy[remaining]
        for axis in (0, 1):
            bins = np.round(pts[:, axis] / bin_size).astype(int)
            unique, counts = np.unique(bins, return_counts=True)
            for bin_id, count in zip(unique, counts):
                center = float(bin_id) * bin_size
                mask = np.abs(pts[:, axis] - center) <= half_width
                group = remaining[mask]
                if group.size < params.cluster_min_points:
                    continue
                span_axis = 1 - axis
                span = float(
                    np.max(points_xy[group, span_axis])
                    - np.min(points_xy[group, span_axis]))
                score = float(count) + 10.0 * span
                if score > best_score:
                    best_score = score
                    best_group = group
        if best_group is None:
            break
        info = _shape_filter_cluster(points_xy, best_group, params)
        if info is None:
            break
        out.append(best_group)
        remaining = np.setdiff1d(remaining, best_group, assume_unique=False)
    return out


def detect_lidar_lines(scan: LidarScan,
                       params: DetectorParams | None = None,
                       base_z_offset_m: float = 0.0,
                       ) -> DetectionResult:
    params = params or DetectorParams()
    start = time.perf_counter()

    local = scan.points_local
    base = local.copy()
    base[:, 0] += LIDAR_X_FROM_BASE_LINK_M
    if abs(base_z_offset_m) > 1e-9:
        base[:, 2] += base_z_offset_m

    layer_ok = np.ones(scan.layers.shape, dtype=bool)
    if params.layer_min >= 0:
        layer_ok &= scan.layers >= params.layer_min
    if params.layer_max >= 0:
        layer_ok &= scan.layers <= params.layer_max

    echo_ok = np.ones(scan.echo.shape, dtype=bool)
    if params.echo_filter >= 0:
        echo_ok &= scan.echo == params.echo_filter

    ground_mask = (
        (scan.ranges >= params.range_min_m)
        & (scan.ranges <= params.range_max_m)
        & (base[:, 0] >= params.base_min_x_m)
        & (base[:, 0] <= params.base_max_x_m)
        & (np.abs(base[:, 1]) <= params.base_max_abs_y_m)
        & (np.abs(base[:, 2] - params.ground_z_m)
           <= params.ground_z_tolerance_m)
        & layer_ok
        & echo_ok
    )

    intensity_candidate_mask = _adaptive_intensity_candidates(
        scan, ground_mask, params)
    reflector_candidate_mask = ground_mask & scan.reflector
    candidate_mode = _normalize_candidate_mode(params.candidate_mode)
    if candidate_mode == "reflector":
        candidate_mask = reflector_candidate_mask
    elif candidate_mode == "reflector_or_intensity":
        candidate_mask = reflector_candidate_mask | intensity_candidate_mask
    else:
        candidate_mask = intensity_candidate_mask
    candidate_idx = np.flatnonzero(candidate_mask)
    candidate_xy = base[candidate_idx, :2]
    clusters = _cluster_points(candidate_xy, params.cluster_link_distance_m)

    accepted_mask = np.zeros(scan.intensity.shape, dtype=bool)
    accepted_clusters: list[ClusterInfo] = []
    output_world_points: list[np.ndarray] = []
    seen_output_cells: set[tuple[int, int]] = set()
    rejected_cluster_count = 0
    for cluster_local_idx in clusters:
        subclusters = [cluster_local_idx]
        if _shape_filter_cluster(candidate_xy, cluster_local_idx, params) is None:
            subclusters = _axis_aligned_subclusters(
                candidate_xy, cluster_local_idx, params)
        accepted_any = False
        for accepted_local_idx in subclusters:
            info = _shape_filter_cluster(candidate_xy, accepted_local_idx, params)
            if info is None:
                continue
            accepted_any = True
            original_indices = candidate_idx[info.indices]
            accepted_mask[original_indices] = True
            accepted_clusters.append(
                ClusterInfo(
                    indices=original_indices,
                    length_m=info.length_m,
                    width_m=info.width_m,
                    aspect_ratio=info.aspect_ratio,
                )
            )
            cluster_world_xy = scan.points_world[
                original_indices, :2].astype(float)
            completed_world_xy = _completed_segment_points(
                cluster_world_xy, params)
            for target_xy in completed_world_xy:
                qx = int(round(float(target_xy[0]) / params.output_voxel_size_m))
                qy = int(round(float(target_xy[1]) / params.output_voxel_size_m))
                key = (qx, qy)
                if key in seen_output_cells:
                    continue
                seen_output_cells.add(key)
                output_world_points.append(
                    np.array([target_xy[0], target_xy[1], 0.0], dtype=np.float32))
                if len(output_world_points) >= params.max_line_points:
                    break
            if len(output_world_points) >= params.max_line_points:
                break
        if not accepted_any:
            rejected_cluster_count += 1
            continue
        if len(output_world_points) >= params.max_line_points:
            break

    if output_world_points:
        line_points_world = np.asarray(output_world_points, dtype=np.float32)
        local_xy = rotate_world_to_local_xy(
            line_points_world[:, :2], scan.robot_pose)
        line_points_local = np.column_stack([
            local_xy[:, 0],
            local_xy[:, 1],
            np.full((local_xy.shape[0],), -SENSOR_HEIGHT_M, dtype=float),
        ]).astype(np.float32)
    else:
        line_points_world = np.zeros((0, 3), dtype=np.float32)
        line_points_local = np.zeros((0, 3), dtype=np.float32)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return DetectionResult(
        ground_mask=ground_mask,
        candidate_mask=candidate_mask,
        accepted_mask=accepted_mask,
        clusters=tuple(accepted_clusters),
        line_points_local=line_points_local,
        line_points_world=line_points_world,
        elapsed_ms=elapsed_ms,
        reflector_candidate_count=int(np.count_nonzero(
            reflector_candidate_mask)),
        intensity_candidate_count=int(np.count_nonzero(
            intensity_candidate_mask)),
        selected_candidate_count=int(candidate_idx.size),
        raw_cluster_count=len(clusters),
        rejected_cluster_count=rejected_cluster_count,
    )


def mark_points_on_grid(points_xy: np.ndarray, spec: GridSpec) -> np.ndarray:
    grid = np.zeros((spec.ny, spec.nx), dtype=bool)
    if points_xy.size == 0:
        return grid
    ix, iy = spec.world_to_cell(points_xy)
    valid = (ix >= 0) & (ix < spec.nx) & (iy >= 0) & (iy < spec.ny)
    grid[iy[valid], ix[valid]] = True
    return grid


def mark_cones_on_grid(cones: tuple[ConeObstacle, ...],
                       spec: GridSpec) -> np.ndarray:
    grid = np.zeros((spec.ny, spec.nx), dtype=bool)
    if not cones:
        return grid
    for iy in range(spec.ny):
        for ix in range(spec.nx):
            wx, wy = spec.cell_to_world(ix, iy)
            for cone in cones:
                if math.hypot(wx - cone.center[0], wy - cone.center[1]) <= cone.radius_m:
                    grid[iy, ix] = True
                    break
    return grid


def mark_detection_on_grid(scan: LidarScan,
                           detection: DetectionResult,
                           spec: GridSpec) -> np.ndarray:
    """Rasterize accepted clusters into continuous costmap line cells."""

    grid = mark_points_on_grid(detection.line_points_world[:, :2], spec)
    for cluster in detection.clusters:
        pts = scan.points_world[cluster.indices, :2].astype(float)
        if pts.shape[0] < 2:
            continue
        center = np.mean(pts, axis=0)
        centered = pts - center
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        projected = centered @ axis
        lo = float(np.min(projected))
        hi = float(np.max(projected))
        if hi - lo < spec.res:
            continue

        samples = max(2, int(math.ceil((hi - lo) / (spec.res * 0.5))))
        ts = np.linspace(lo, hi, samples)
        line_pts = center + ts[:, None] * axis[None, :]
        grid |= mark_points_on_grid(line_pts, spec)

    return grid


def inflate_grid(grid: np.ndarray,
                 radius_m: float,
                 res_m: float = GRID_RES_M) -> np.ndarray:
    radius_cells = int(math.ceil(radius_m / res_m))
    if radius_cells <= 0 or not np.any(grid):
        return grid.copy()

    ys, xs = np.nonzero(grid)
    inflated = grid.copy()
    offsets: list[tuple[int, int]] = []
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if math.hypot(dx, dy) * res_m <= radius_m:
                offsets.append((dy, dx))

    for dy, dx in offsets:
        ny = ys + dy
        nx = xs + dx
        valid = (
            (ny >= 0) & (ny < grid.shape[0])
            & (nx >= 0) & (nx < grid.shape[1])
        )
        inflated[ny[valid], nx[valid]] = True
    return inflated


def astar(blocked: np.ndarray,
          spec: GridSpec,
          start_xy: np.ndarray,
          goal_xy: np.ndarray) -> list[tuple[float, float]]:
    start_ix, start_iy = spec.world_to_cell(np.asarray(start_xy)[None, :])
    goal_ix, goal_iy = spec.world_to_cell(np.asarray(goal_xy)[None, :])
    start = (int(start_ix[0]), int(start_iy[0]))
    goal = (int(goal_ix[0]), int(goal_iy[0]))

    if not spec.in_bounds(*start) or not spec.in_bounds(*goal):
        return []

    blocked_work = blocked.copy()
    for cell in (start, goal):
        cx, cy = cell
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx = cx + dx
                ny = cy + dy
                if spec.in_bounds(nx, ny):
                    blocked_work[ny, nx] = False

    neighbors = (
        (-1, -1, math.sqrt(2.0)), (0, -1, 1.0), (1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0), (1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)), (0, 1, 1.0), (1, 1, math.sqrt(2.0)),
    )

    def heuristic(cell: tuple[int, int]) -> float:
        return math.hypot(cell[0] - goal[0], cell[1] - goal[1])

    open_heap: list[tuple[float, tuple[int, int]]] = [(heuristic(start), start)]
    g_score = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _priority, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            cells = [current]
            while current in came_from:
                current = came_from[current]
                cells.append(current)
            cells.reverse()
            return [spec.cell_to_world(ix, iy) for ix, iy in cells]

        closed.add(current)
        cx, cy = current
        for dx, dy, step_cost in neighbors:
            nx = cx + dx
            ny = cy + dy
            nxt = (nx, ny)
            if not spec.in_bounds(nx, ny):
                continue
            if blocked_work[ny, nx]:
                continue
            new_cost = g_score[current] + step_cost
            if new_cost >= g_score.get(nxt, float("inf")):
                continue
            came_from[nxt] = current
            g_score[nxt] = new_cost
            heapq.heappush(open_heap, (new_cost + heuristic(nxt), nxt))

    return []


@dataclass(frozen=True)
class ControllerOutput:
    F_left: float
    F_right: float
    v_des: float
    omega_des: float
    backwards_request: bool


def _path_carrot(robot: RobotPose,
                 path_xy: list[tuple[float, float]],
                 lookahead: float = LOOKAHEAD_M) -> tuple[tuple[float, float],
                                                           int]:
    if not path_xy:
        return (robot.x, robot.y), 0

    rx, ry = robot.nav_center()
    best_k = 0
    best_d2 = float("inf")
    for k, (px, py) in enumerate(path_xy):
        d2 = (px - rx) ** 2 + (py - ry) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_k = k

    acc = 0.0
    carrot = path_xy[-1]
    for k in range(best_k, len(path_xy) - 1):
        x0, y0 = path_xy[k]
        x1, y1 = path_xy[k + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if acc + seg >= lookahead:
            t = (lookahead - acc) / max(seg, 1e-6)
            carrot = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            break
        acc += seg
    return carrot, best_k


def pure_pursuit_to_wheel_forces(
    robot: RobotPose,
    path_xy: list[tuple[float, float]],
    *,
    allow_reverse: bool = False,
    target_speed: float = DESIRED_SPEED_MPS,
    lookahead: float = LOOKAHEAD_M,
) -> ControllerOutput:
    if not path_xy:
        return ControllerOutput(0.0, 0.0, 0.0, 0.0, False)

    rx, ry = robot.nav_center()
    carrot, _best_k = _path_carrot(robot, path_xy, lookahead)
    goal = path_xy[-1]
    dist_to_goal = math.hypot(goal[0] - rx, goal[1] - ry)
    v_des = target_speed * min(1.0, dist_to_goal / APPROACH_SLOW_M)
    if dist_to_goal < GOAL_TOLERANCE_M:
        v_des = 0.0

    bearing = math.atan2(carrot[1] - ry, carrot[0] - rx)
    err = _wrap_angle(bearing - robot.heading)
    backwards = abs(err) > math.pi / 2.0
    if backwards and not allow_reverse:
        v_des = 0.0
        omega_des = max(
            -MAX_DWB_THETA_RADPS,
            min(MAX_DWB_THETA_RADPS, 2.4 * err),
        )
        a_yaw = KP_ANG * (omega_des - robot.omega)
        a_long = KP_LIN * (v_des - robot.u)
        F_total = ROBOT_MASS_KG * a_long / 10.0
        tau_total = INERTIA_REAR * a_yaw / 6.0
        F_left = 0.5 * F_total - tau_total / TRACK_WIDTH_M
        F_right = 0.5 * F_total + tau_total / TRACK_WIDTH_M
        return ControllerOutput(F_left, F_right, v_des, omega_des, True)

    if allow_reverse and backwards:
        err = err + math.pi if err < 0 else err - math.pi
        v_des = -abs(v_des)

    omega_limit = (
        MAX_BEHAVIOR_THETA_RADPS if allow_reverse else MAX_DWB_THETA_RADPS)
    omega_des = max(-omega_limit, min(omega_limit, 1.8 * err))
    a_long = KP_LIN * (v_des - robot.u) - KD_LIN * 0.0
    a_yaw = KP_ANG * (omega_des - robot.omega) - KD_ANG * 0.0
    F_total = ROBOT_MASS_KG * a_long / 10.0
    tau_total = INERTIA_REAR * a_yaw / 6.0
    F_left = 0.5 * F_total - tau_total / TRACK_WIDTH_M
    F_right = 0.5 * F_total + tau_total / TRACK_WIDTH_M
    return ControllerOutput(F_left, F_right, v_des, omega_des, False)


def velocity_command_to_wheel_forces(robot: RobotPose,
                                     v_des: float,
                                     omega_des: float,
                                     backwards_request: bool = False
                                     ) -> ControllerOutput:
    v_des = max(-MAX_REVERSE_SPEED_MPS, min(MAX_LINEAR_SPEED_MPS, v_des))
    omega_des = max(
        -MAX_BEHAVIOR_THETA_RADPS,
        min(MAX_BEHAVIOR_THETA_RADPS, omega_des),
    )
    a_long = KP_LIN * (v_des - robot.u)
    a_yaw = KP_ANG * (omega_des - robot.omega)
    F_total = ROBOT_MASS_KG * a_long / 10.0
    tau_total = INERTIA_REAR * a_yaw / 6.0
    F_left = 0.5 * F_total - tau_total / TRACK_WIDTH_M
    F_right = 0.5 * F_total + tau_total / TRACK_WIDTH_M
    return ControllerOutput(
        F_left, F_right, v_des, omega_des, backwards_request)


def costmap_distance_cells(blocked: np.ndarray,
                           max_cells: int = DWB_CRITIC_RADIUS_CELLS
                           ) -> np.ndarray:
    dist = np.full(blocked.shape, max_cells + 1, dtype=np.int16)
    ys, xs = np.nonzero(blocked)
    if ys.size == 0:
        return dist

    dist[ys, xs] = 0
    frontier = list(zip(ys.tolist(), xs.tolist()))
    for d in range(1, max_cells + 1):
        next_frontier: list[tuple[int, int]] = []
        for y, x in frontier:
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= blocked.shape[0]:
                    continue
                if nx < 0 or nx >= blocked.shape[1]:
                    continue
                if dist[ny, nx] <= d:
                    continue
                dist[ny, nx] = d
                next_frontier.append((ny, nx))
        frontier = next_frontier
        if not frontier:
            break
    return dist


def _point_segment_distance(px: float,
                            py: float,
                            ax: float,
                            ay: float,
                            bx: float,
                            by: float) -> float:
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def _point_polyline_distance(px: float,
                             py: float,
                             path_xy: list[tuple[float, float]]) -> float:
    if not path_xy:
        return 0.0
    if len(path_xy) == 1:
        return math.hypot(px - path_xy[0][0], py - path_xy[0][1])
    best = float("inf")
    for (ax, ay), (bx, by) in zip(path_xy[:-1], path_xy[1:]):
        best = min(best, _point_segment_distance(px, py, ax, ay, bx, by))
    return best


def _lookahead_bearing_error(robot: RobotPose,
                             path_xy: list[tuple[float, float]],
                             min_lookahead_m: float = LOOKAHEAD_M) -> float:
    carrot, _best_k = _path_carrot(robot, path_xy, min_lookahead_m)
    err = _wrap_angle(
        math.atan2(carrot[1] - robot.y, carrot[0] - robot.x)
        - robot.heading)
    return err


def dwb_with_line_critic(
    robot: RobotPose,
    path_xy: list[tuple[float, float]],
    line_dist_grid: np.ndarray | None,
    spec: GridSpec,
    *,
    allow_reverse: bool = False,
    target_speed: float = DESIRED_SPEED_MPS,
    lookahead: float = LOOKAHEAD_M,
) -> ControllerOutput:
    baseline = pure_pursuit_to_wheel_forces(
        robot, path_xy,
        allow_reverse=allow_reverse,
        target_speed=target_speed,
        lookahead=lookahead,
    )
    if line_dist_grid is None or not path_xy or baseline.v_des == 0.0:
        return baseline

    rx, ry = robot.nav_center()
    goal = path_xy[-1]
    max_d = DWB_CRITIC_RADIUS_CELLS
    steps = max(1, int(round(DWB_HORIZON_S / DWB_HORIZON_DT_S)))
    best_score = math.inf
    best_v = baseline.v_des
    best_w = baseline.omega_des

    if allow_reverse and baseline.v_des < 0.0:
        v_candidates = np.linspace(
            baseline.v_des, 0.0, max(2, DWB_V_SAMPLES // 2))
    else:
        v_hi = min(MAX_LINEAR_SPEED_MPS, max(0.0, target_speed))
        v_candidates = np.linspace(0.0, v_hi, DWB_V_SAMPLES)
    w_candidates = np.clip(
        baseline.omega_des + np.linspace(-0.35, 0.35, DWB_W_SAMPLES),
        -MAX_DWB_THETA_RADPS,
        MAX_DWB_THETA_RADPS,
    )
    w_candidates = np.asarray([
        0.0 if abs(w) < 1e-6
        else math.copysign(max(abs(float(w)), MIN_SPEED_THETA_RADPS), float(w))
        for w in w_candidates
    ], dtype=float)
    w_candidates = np.clip(
        w_candidates, -MAX_DWB_THETA_RADPS, MAX_DWB_THETA_RADPS)

    for v_cand in v_candidates:
        for w_cand in w_candidates:
            x, y, th = rx, ry, robot.heading
            collided = False
            obstacle_penalty = 0.0
            for _ in range(steps):
                x += v_cand * math.cos(th) * DWB_HORIZON_DT_S
                y += v_cand * math.sin(th) * DWB_HORIZON_DT_S
                th = _wrap_angle(th + w_cand * DWB_HORIZON_DT_S)
                ix, iy = spec.world_to_cell(np.asarray([[x, y]], dtype=float))
                cx = int(ix[0])
                cy = int(iy[0])
                if not spec.in_bounds(cx, cy):
                    collided = True
                    break
                d = int(line_dist_grid[cy, cx])
                if d == 0:
                    collided = True
                    break
                if d <= max_d:
                    obstacle_penalty += (max_d + 1 - d) / max(1, max_d)
            if collided:
                continue
            path_dist = _point_polyline_distance(x, y, path_xy)
            goal_dist = math.hypot(x - goal[0], y - goal[1])
            path_align = abs(_lookahead_bearing_error(
                RobotPose(x, y, th), path_xy, min_lookahead_m=lookahead))
            goal_align = abs(_wrap_angle(
                math.atan2(goal[1] - y, goal[0] - x) - th))
            score = (
                DWB_PATH_DIST_WEIGHT * path_dist
                + DWB_GOAL_DIST_WEIGHT * goal_dist
                + 0.10 * DWB_PATH_ALIGN_WEIGHT * path_align
                + 0.10 * DWB_GOAL_ALIGN_WEIGHT * goal_align
                + DWB_OBSTACLE_WEIGHT * obstacle_penalty
                - DWB_FORWARD_PROGRESS_WEIGHT * max(0.0, float(v_cand))
            )
            if score < best_score:
                best_score = score
                best_v = v_cand
                best_w = w_cand

    return velocity_command_to_wheel_forces(
        robot, best_v, best_w, baseline.backwards_request)


class LidarLineSimulation:
    def __init__(self,
                 rays: int = DEFAULT_RAYS,
                 seed: int = DEFAULT_SEED,
                 max_range_m: float = DEFAULT_MAX_RANGE_M,
                 detector_params: DetectorParams | None = None,
                 base_z_offset_m: float = 0.0,
                 detector_label: str = "sim defaults",
                 line_layer_params: LineLayerParams | None = None,
                 line_layer_label: str = "sim defaults",
                 scenario: str = "competition"):
        self.rays = rays
        self.seed = seed
        self.max_range_m = max_range_m
        self.scenario = scenario
        self.world = make_world(scenario)
        self.line_layer_params = line_layer_params or LineLayerParams()
        self.line_layer_label = line_layer_label
        self.grid_spec = GridSpec(
            res=max(0.02,
                    self.line_layer_params.observation_persistence_resolution_m))
        self.detector_params = (
            detector_params or DetectorParams(range_max_m=max_range_m - 0.5)
        )
        self.base_z_offset_m = base_z_offset_m
        self.detector_label = detector_label
        self.robot = default_robot(scenario)
        self.goal = default_goal(scenario)
        self.nav_goal = self.goal.copy()
        self.sim_time_s = 0.0
        self.scan_period_s = self._period_s(
            self.detector_params.max_processing_rate_hz,
            LINE_DETECTOR_RATE_HZ)
        self.local_update_period_s = self._period_s(
            self.line_layer_params.local_update_frequency_hz, 15.0)
        self.global_update_period_s = self._period_s(
            self.line_layer_params.global_update_frequency_hz, 3.0)
        self.planner_period_s = self._period_s(
            self.line_layer_params.bt_replan_frequency_hz,
            BT_REPLAN_FREQUENCY_HZ)
        self.controller_period_s = 1.0 / CONTROLLER_FREQUENCY_HZ
        self.next_scan_s = 0.0
        self.next_local_update_s = 0.0
        self.next_global_update_s = 0.0
        self.next_planner_s = 0.0
        self.next_controller_s = 0.0
        self.line_last_seen_s = np.full(
            (self.grid_spec.ny, self.grid_spec.nx), -np.inf, dtype=float)
        self.line_memory = np.zeros_like(self.line_last_seen_s, dtype=bool)
        self.line_age_s = np.full_like(self.line_last_seen_s, np.inf)
        self.last_detected_cells = self.line_memory.copy()
        self.latest_detector_cells = self.line_memory.copy()
        self.latest_detector_stamp_s = -np.inf
        self.detector_msg_available = False
        self.local_line_cells = self.line_memory.copy()
        self.local_line_publish_pending = False
        self.pca_obstacle_cells = mark_cones_on_grid(
            self.world.cone_obstacles, self.grid_spec)
        self.pca_inflated = inflate_grid(
            self.pca_obstacle_cells,
            ROBOT_LATERAL_CLEARANCE_M + LOCAL_FOOTPRINT_PADDING_M,
            self.grid_spec.res,
        )
        self.last_pca_points_world = sample_pca_obstacle_points(
            self.world, self.robot)
        self.last_scan: LidarScan | None = None
        self.last_detection: DetectionResult | None = None
        self.last_inflated = self.line_memory.copy()
        self.local_inflated = self.local_line_cells.copy()
        self.line_distance_grid = costmap_distance_cells(
            self.local_inflated | self.pca_inflated)
        self.path: list[tuple[float, float]] = []
        self.trail: list[tuple[float, float]] = [(self.robot.x, self.robot.y)]
        self.last_controller = ControllerOutput(0.0, 0.0, 0.0, 0.0, False)
        self.breadcrumbs: list[tuple[float, float]] = []
        self.last_breadcrumb_drop: tuple[float, float] | None = None
        self.crumbs_consumed_session = 0
        self.recovery_mode = "IDLE"

    @staticmethod
    def _period_s(rate_hz: float, fallback_hz: float) -> float:
        rate = rate_hz if rate_hz > 0.0 else fallback_hz
        return 1.0 / max(1e-6, rate)

    def nav2_planning_hold_ms(self) -> int:
        """Nominal last-detection-to-fresh-plan latency on path_following_two."""
        detector_ms = (
            1000.0 / self.detector_params.max_processing_rate_hz
            if self.detector_params.max_processing_rate_hz > 0.0
            else 0.0
        )
        return int(round(
            detector_ms
            + 1000.0 * self.local_update_period_s
            + 1000.0 * self.global_update_period_s
            + 1000.0 * self.planner_period_s
        ))

    def nav2_rviz_hold_ms(self) -> int:
        publish_ms = LineLayerParams._period_ms(
            self.line_layer_params.global_publish_frequency_hz)
        return int(round(self.nav2_planning_hold_ms() + publish_ms))

    def persistence_summary(self) -> str:
        return (
            f"{self.nav2_planning_hold_ms()} ms planner flow "
            f"(line_layer={self.line_layer_params.observation_persistence_ms} ms, "
            f"rviz~{self.nav2_rviz_hold_ms()} ms)"
        )

    def _reset_memory_arrays(self) -> None:
        self.line_last_seen_s = np.full(
            (self.grid_spec.ny, self.grid_spec.nx), -np.inf, dtype=float)
        self.line_memory = np.zeros_like(self.line_last_seen_s, dtype=bool)
        self.line_age_s = np.full_like(self.line_last_seen_s, np.inf)
        self.last_detected_cells = self.line_memory.copy()
        self.latest_detector_cells = self.line_memory.copy()
        self.latest_detector_stamp_s = -np.inf
        self.detector_msg_available = False
        self.local_line_cells = self.line_memory.copy()
        self.local_line_publish_pending = False
        self.pca_obstacle_cells = mark_cones_on_grid(
            self.world.cone_obstacles, self.grid_spec)
        self.pca_inflated = inflate_grid(
            self.pca_obstacle_cells,
            ROBOT_LATERAL_CLEARANCE_M + LOCAL_FOOTPRINT_PADDING_M,
            self.grid_spec.res,
        )
        self.last_pca_points_world = sample_pca_obstacle_points(
            self.world, self.robot)
        self.last_inflated = self.line_memory.copy()
        self.local_inflated = self.local_line_cells.copy()
        self.line_distance_grid = costmap_distance_cells(
            self.local_inflated | self.pca_inflated)

    def set_line_layer_params(self,
                              params: LineLayerParams,
                              label: str = "sim defaults") -> None:
        old_res = self.grid_spec.res
        self.line_layer_params = params
        self.line_layer_label = label
        self.local_update_period_s = self._period_s(
            params.local_update_frequency_hz, 15.0)
        self.global_update_period_s = self._period_s(
            params.global_update_frequency_hz, 3.0)
        self.planner_period_s = self._period_s(
            params.bt_replan_frequency_hz, BT_REPLAN_FREQUENCY_HZ)
        new_res = max(0.02, params.observation_persistence_resolution_m)
        if abs(old_res - new_res) > 1e-9:
            self.grid_spec = GridSpec(res=new_res)
            self._reset_memory_arrays()

    def set_detector_params(self,
                            params: DetectorParams,
                            base_z_offset_m: float,
                            label: str = "sim defaults") -> None:
        self.detector_params = params
        self.base_z_offset_m = base_z_offset_m
        self.detector_label = label
        self.scan_period_s = self._period_s(
            params.max_processing_rate_hz, LINE_DETECTOR_RATE_HZ)

    def set_scenario(self, scenario: str) -> None:
        self.scenario = scenario
        self.world = make_world(scenario)
        self.reset()

    def reset(self) -> None:
        self.robot = default_robot(self.scenario)
        self.goal = default_goal(self.scenario)
        self.nav_goal = self.goal.copy()
        self.sim_time_s = 0.0
        self.next_scan_s = 0.0
        self.next_local_update_s = 0.0
        self.next_global_update_s = 0.0
        self.next_planner_s = 0.0
        self.next_controller_s = 0.0
        self._reset_memory_arrays()
        self.last_scan = None
        self.last_detection = None
        self.path = []
        self.trail = [(self.robot.x, self.robot.y)]
        self.last_controller = ControllerOutput(0.0, 0.0, 0.0, 0.0, False)
        self.breadcrumbs = []
        self.last_breadcrumb_drop = None
        self.crumbs_consumed_session = 0
        self.recovery_mode = "IDLE"

    def _cells_in_clear_view(self, cell_mask: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(cell_mask)
        view = np.zeros_like(cell_mask, dtype=bool)
        if ys.size == 0:
            return view

        world = np.asarray(
            [self.grid_spec.cell_to_world(int(x), int(y))
             for y, x in zip(ys, xs)],
            dtype=float,
        )
        dx = world[:, 0] - self.robot.x
        dy = world[:, 1] - self.robot.y
        c = math.cos(self.robot.heading)
        s = math.sin(self.robot.heading)
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        ranges = np.hypot(local_x, local_y)
        angles = np.arctan2(local_y, local_x)
        params = self.line_layer_params
        ok = (
            (ranges >= params.line_clear_range_min_m)
            & (ranges <= params.line_clear_range_max_m)
            & (angles >= params.line_clear_angle_min_rad)
            & (angles <= params.line_clear_angle_max_rad)
        )
        view[ys[ok], xs[ok]] = True
        return view

    def _expire_line_memory(self) -> None:
        params = self.line_layer_params
        valid = np.isfinite(self.line_last_seen_s)
        self.line_age_s = np.where(
            valid, self.sim_time_s - self.line_last_seen_s, np.inf)
        if not params.clearing:
            self.line_memory = valid
            return

        stale = valid & (self.line_age_s > params.persistence_s)
        if np.any(stale):
            if params.clear_lines_only_in_view:
                stale &= self._cells_in_clear_view(stale)
            self.line_last_seen_s[stale] = -np.inf

        valid = np.isfinite(self.line_last_seen_s)
        if params.max_persisted_points > 0:
            count = int(np.count_nonzero(valid))
            if count > params.max_persisted_points:
                flat = self.line_last_seen_s.ravel()
                valid_idx = np.flatnonzero(np.isfinite(flat))
                oldest_count = count - params.max_persisted_points
                drop_idx = valid_idx[np.argsort(flat[valid_idx])[:oldest_count]]
                flat[drop_idx] = -np.inf
                self.line_last_seen_s = flat.reshape(self.line_last_seen_s.shape)
                valid = np.isfinite(self.line_last_seen_s)

        self.line_memory = valid
        self.line_age_s = np.where(
            valid, self.sim_time_s - self.line_last_seen_s, np.inf)

    def _local_window_slices(self) -> tuple[int, int, int, int]:
        params = self.line_layer_params
        half_w = max(self.grid_spec.res, params.local_costmap_width_m * 0.5)
        half_h = max(self.grid_spec.res, params.local_costmap_height_m * 0.5)
        x0 = int(math.floor((self.robot.x - half_w - self.grid_spec.xmin)
                            / self.grid_spec.res))
        x1 = int(math.ceil((self.robot.x + half_w - self.grid_spec.xmin)
                           / self.grid_spec.res))
        y0 = int(math.floor((self.robot.y - half_h - self.grid_spec.ymin)
                            / self.grid_spec.res))
        y1 = int(math.ceil((self.robot.y + half_h - self.grid_spec.ymin)
                           / self.grid_spec.res))
        return (
            max(0, min(self.grid_spec.nx, x0)),
            max(0, min(self.grid_spec.nx, x1)),
            max(0, min(self.grid_spec.ny, y0)),
            max(0, min(self.grid_spec.ny, y1)),
        )

    def _run_detector_cycle(self) -> DetectionResult:
        self.last_pca_points_world = sample_pca_obstacle_points(
            self.world, self.robot)
        scan = simulate_scan(
            self.world,
            self.robot,
            rays=self.rays,
            max_range_m=self.max_range_m,
            seed=self.seed,
        )
        detection = detect_lidar_lines(
            scan,
            self.detector_params,
            base_z_offset_m=self.base_z_offset_m,
        )
        cells = mark_detection_on_grid(scan, detection, self.grid_spec)
        self.last_detected_cells = cells
        if self.detector_params.publish_empty_messages or np.any(cells):
            self.latest_detector_cells = cells.copy()
            self.latest_detector_stamp_s = self.sim_time_s
            self.detector_msg_available = True
        self.last_scan = scan
        self.last_detection = detection
        return detection

    def _update_local_line_layer(self, force: bool = False) -> None:
        _ = force
        params = self.line_layer_params
        cells = np.zeros_like(self.line_memory, dtype=bool)
        msg_valid = self.detector_msg_available
        if msg_valid and params.max_message_age_ms >= 0:
            age_ms = (self.sim_time_s - self.latest_detector_stamp_s) * 1000.0
            msg_valid = age_ms <= params.max_message_age_ms
        if msg_valid:
            x0, x1, y0, y1 = self._local_window_slices()
            cells[y0:y1, x0:x1] = self.latest_detector_cells[y0:y1, x0:x1]
        self.local_line_cells = cells
        self.local_inflated = inflate_grid(
            self.local_line_cells, LINE_INFLATION_M, self.grid_spec.res)
        self.line_distance_grid = costmap_distance_cells(
            self.local_inflated | self.pca_inflated)
        self.local_line_publish_pending = True

    def _update_global_mirror_layer(self, force: bool = False) -> None:
        if not force and not self.local_line_publish_pending:
            return
        x0, x1, y0, y1 = self._local_window_slices()
        if x0 >= x1 or y0 >= y1:
            return
        local_window = self.local_line_cells[y0:y1, x0:x1]

        # The robot's lidar_line_memory_mirror_layer mirrors the rolling local
        # /lidar_line_costmap. With allow_decrease=true, FREE cells clear stored
        # global cells inside the current 6x6 m source window, while cells
        # outside that source footprint remain accumulated.
        if self.line_layer_params.lidar_mirror_allow_decrease:
            self.line_memory[y0:y1, x0:x1] = local_window
            self.line_last_seen_s[y0:y1, x0:x1] = np.where(
                local_window, self.sim_time_s, -np.inf)
        else:
            self.line_memory[y0:y1, x0:x1] |= local_window
            self.line_last_seen_s[y0:y1, x0:x1] = np.where(
                local_window,
                self.sim_time_s,
                self.line_last_seen_s[y0:y1, x0:x1],
            )
        valid = np.isfinite(self.line_last_seen_s)
        self.line_age_s = np.where(
            valid, self.sim_time_s - self.line_last_seen_s, np.inf)
        self.last_inflated = inflate_grid(
            self.line_memory, LINE_INFLATION_M, self.grid_spec.res)
        self.last_inflated |= self.pca_inflated
        self.local_line_publish_pending = False

    def _forward_blocked_state(
        self,
        path_xy: list[tuple[float, float]],
        goal_xy: np.ndarray,
    ) -> tuple[bool, bool, bool, float, float]:
        rel_goal = _wrap_angle(
            math.atan2(goal_xy[1] - self.robot.y, goal_xy[0] - self.robot.x)
            - self.robot.heading)
        goal_behind = abs(rel_goal) > FORWARD_BLOCKED_ANGLE_THRESHOLD_RAD
        if len(path_xy) < 2:
            return goal_behind, False, False, rel_goal, 0.0
        rel_path = _lookahead_bearing_error(
            self.robot, path_xy, min_lookahead_m=LOOKAHEAD_M)
        path_behind = abs(rel_path) > FORWARD_BLOCKED_ANGLE_THRESHOLD_RAD
        return goal_behind, path_behind, True, rel_goal, rel_path

    def _bent_goal_for_path(
        self,
        direct_path: list[tuple[float, float]],
    ) -> np.ndarray:
        goal_behind, path_behind, path_valid, rel_goal, _rel_path = (
            self._forward_blocked_state(direct_path, self.goal))
        buffer_empty = not self.breadcrumbs
        bend_now = (
            path_valid
            and path_behind
            and (goal_behind or buffer_empty)
        )
        if not bend_now:
            return self.goal.copy()

        offset = GOAL_BENDER_ANGLE_RAD if rel_goal >= 0.0 else -GOAL_BENDER_ANGLE_RAD
        heading = self.robot.heading + offset
        bent = np.array([
            self.robot.x + GOAL_BENDER_DISTANCE_M * math.cos(heading),
            self.robot.y + GOAL_BENDER_DISTANCE_M * math.sin(heading),
        ], dtype=float)
        bent[0] = min(FIELD_X_MAX - ROBOT_RADIUS_M,
                      max(FIELD_X_MIN + ROBOT_RADIUS_M, bent[0]))
        bent[1] = min(FIELD_Y_MAX - ROBOT_RADIUS_M,
                      max(FIELD_Y_MIN + ROBOT_RADIUS_M, bent[1]))
        return bent

    def _update_planner(self) -> None:
        self.last_inflated = inflate_grid(
            self.line_memory, LINE_INFLATION_M, self.grid_spec.res)
        self.last_inflated |= self.pca_inflated
        direct_path = astar(
            self.last_inflated,
            self.grid_spec,
            np.array([self.robot.x, self.robot.y]),
            self.goal,
        )
        self.nav_goal = self._bent_goal_for_path(direct_path)
        if np.linalg.norm(self.nav_goal - self.goal) <= 1e-6:
            self.path = direct_path
        else:
            self.path = astar(
                self.last_inflated,
                self.grid_spec,
                np.array([self.robot.x, self.robot.y]),
                self.nav_goal,
            )

    def _update_breadcrumbs(self) -> None:
        current = (self.robot.x, self.robot.y)
        if self.robot.u > BREADCRUMB_MIN_FORWARD_VX_MPS:
            if self.last_breadcrumb_drop is None:
                self.breadcrumbs.append(current)
                self.last_breadcrumb_drop = current
            else:
                dx = current[0] - self.last_breadcrumb_drop[0]
                dy = current[1] - self.last_breadcrumb_drop[1]
                if math.hypot(dx, dy) >= BREADCRUMB_STRIDE_M:
                    self.breadcrumbs.append(current)
                    self.last_breadcrumb_drop = current
            if len(self.breadcrumbs) > BREADCRUMB_BUFFER_SIZE:
                self.breadcrumbs = self.breadcrumbs[-BREADCRUMB_BUFFER_SIZE:]
            return

        if self.breadcrumbs:
            tx, ty = self.breadcrumbs[-1]
            if math.hypot(current[0] - tx, current[1] - ty) <= (
                    BREADCRUMB_CONSUME_TOLERANCE_M):
                self.breadcrumbs.pop()
                self.last_breadcrumb_drop = None
                self.crumbs_consumed_session += 1

    def _breadcrumb_reverse_controller(self) -> ControllerOutput:
        if (not self.breadcrumbs
                or self.crumbs_consumed_session >= BREADCRUMB_MAX_CRUMBS_PER_SESSION):
            return ControllerOutput(0.0, 0.0, 0.0, 0.0, True)
        tx, ty = self.breadcrumbs[-1]
        dx = tx - self.robot.x
        dy = ty - self.robot.y
        rel = _wrap_angle(math.atan2(dy, dx) - self.robot.heading)
        err = _wrap_angle(math.pi - rel)
        align_factor = max(0.0, min(1.0, 1.0 - abs(err) / (math.pi / 2.0)))
        v_des = -BREADCRUMB_REVERSE_SPEED_MPS * align_factor
        omega_des = max(
            -BREADCRUMB_MAX_ANGULAR_SPEED_RADPS,
            min(BREADCRUMB_MAX_ANGULAR_SPEED_RADPS, -err * 1.5),
        )
        return velocity_command_to_wheel_forces(
            self.robot, v_des, omega_des, backwards_request=True)

    def _backup_controller(self) -> ControllerOutput:
        return velocity_command_to_wheel_forces(
            self.robot, -BACKUP_SPEED_MPS, 0.0, backwards_request=True)

    def _gradient_escape_controller(self) -> ControllerOutput:
        if self.line_distance_grid is None:
            return velocity_command_to_wheel_forces(
                self.robot, GRADIENT_ESCAPE_SPEED_MPS, 0.0)
        best_score = -1
        best_heading = self.robot.heading
        for k in range(16):
            heading = 2.0 * math.pi * k / 16.0
            sx = self.robot.x + GRADIENT_ESCAPE_SAMPLE_RADIUS_M * math.cos(heading)
            sy = self.robot.y + GRADIENT_ESCAPE_SAMPLE_RADIUS_M * math.sin(heading)
            ix, iy = self.grid_spec.world_to_cell(
                np.asarray([[sx, sy]], dtype=float))
            cx = int(ix[0])
            cy = int(iy[0])
            if not self.grid_spec.in_bounds(cx, cy):
                continue
            score = int(self.line_distance_grid[cy, cx])
            if score > best_score:
                best_score = score
                best_heading = heading
        err = _wrap_angle(best_heading - self.robot.heading)
        return velocity_command_to_wheel_forces(
            self.robot,
            GRADIENT_ESCAPE_SPEED_MPS,
            max(-MAX_BEHAVIOR_THETA_RADPS,
                min(MAX_BEHAVIOR_THETA_RADPS, 1.8 * err)),
        )

    def _select_controller(self) -> ControllerOutput:
        if len(self.path) >= 2:
            goal_behind, path_behind, path_valid, _rel_goal, _rel_path = (
                self._forward_blocked_state(self.path, self.nav_goal))
            forward_blocked = path_valid and path_behind and not goal_behind
            if not forward_blocked:
                self.crumbs_consumed_session = 0
                self.recovery_mode = "FOLLOW_PATH"
                return dwb_with_line_critic(
                    self.robot,
                    self.path,
                    self.line_distance_grid,
                    self.grid_spec,
                    target_speed=DESIRED_SPEED_MPS,
                    lookahead=LOOKAHEAD_M,
                )

            if self.breadcrumbs:
                self.recovery_mode = "BREADCRUMB_REVERSE"
                return self._breadcrumb_reverse_controller()

            self.recovery_mode = "GOAL_BENDER_WAIT"
            return velocity_command_to_wheel_forces(self.robot, 0.0, 0.0)

        if self.breadcrumbs:
            self.recovery_mode = "BREADCRUMB_REVERSE"
            return self._breadcrumb_reverse_controller()

        phase = int(self.sim_time_s) % 2
        if phase == 0:
            self.recovery_mode = "BACKUP"
            return self._backup_controller()
        self.recovery_mode = "GRADIENT_ESCAPE"
        return self._gradient_escape_controller()

    def run_cycle(self, clear_memory: bool = False) -> DetectionResult:
        if clear_memory:
            self._reset_memory_arrays()
        detection = self._run_detector_cycle()
        self._update_local_line_layer(force=True)
        self._update_global_mirror_layer(force=True)
        self._update_planner()
        return detection

    def step_robot(self, distance_m: float = 0.35) -> None:
        start = np.array([self.robot.x, self.robot.y], dtype=float)
        elapsed = 0.0
        while elapsed < 5.0:
            self.advance(1.0 / RENDER_FPS)
            elapsed += 1.0 / RENDER_FPS
            pos = np.array([self.robot.x, self.robot.y], dtype=float)
            if float(np.linalg.norm(pos - start)) >= distance_m:
                break

    def advance(self, dt: float, speed_scale: float = 1.0) -> None:
        target = self.sim_time_s + max(0.0, dt * speed_scale)
        if self.last_scan is None:
            self.run_cycle(clear_memory=True)
            self.next_scan_s = self.sim_time_s + self.scan_period_s
            self.next_local_update_s = (
                self.sim_time_s + self.local_update_period_s)
            self.next_global_update_s = (
                self.sim_time_s + self.global_update_period_s)
            self.next_planner_s = self.sim_time_s + self.planner_period_s

        while self.sim_time_s < target - 1e-9:
            if self.sim_time_s + 1e-9 >= self.next_scan_s:
                self._run_detector_cycle()
                self.next_scan_s += self.scan_period_s
            if self.sim_time_s + 1e-9 >= self.next_local_update_s:
                self._update_local_line_layer()
                self.next_local_update_s += self.local_update_period_s
            if self.sim_time_s + 1e-9 >= self.next_global_update_s:
                self._update_global_mirror_layer()
                self.next_global_update_s += self.global_update_period_s
            if self.sim_time_s + 1e-9 >= self.next_planner_s:
                self._update_planner()
                self.next_planner_s += self.planner_period_s
            if self.sim_time_s + 1e-9 >= self.next_controller_s:
                self.last_controller = self._select_controller()
                self.next_controller_s += self.controller_period_s

            step = min(PHYS_DT, target - self.sim_time_s)
            next_event_s = min(
                self.next_scan_s,
                self.next_local_update_s,
                self.next_global_update_s,
                self.next_planner_s,
                self.next_controller_s,
            )
            if next_event_s > self.sim_time_s:
                step = min(step, next_event_s - self.sim_time_s)
            if step <= 1e-9:
                continue

            self.robot.step_dynamics(
                self.last_controller.F_left,
                self.last_controller.F_right,
                step,
            )
            self.sim_time_s += step
            self._update_breadcrumbs()
            if not self.trail:
                self.trail.append((self.robot.x, self.robot.y))
            else:
                lx, ly = self.trail[-1]
                if math.hypot(self.robot.x - lx, self.robot.y - ly) >= 0.05:
                    self.trail.append((self.robot.x, self.robot.y))
                    if len(self.trail) > 3000:
                        self.trail = self.trail[-3000:]

        if self.sim_time_s + 1e-9 >= self.next_scan_s:
            self._run_detector_cycle()
            self.next_scan_s += self.scan_period_s
        if self.sim_time_s + 1e-9 >= self.next_local_update_s:
            self._update_local_line_layer()
            self.next_local_update_s += self.local_update_period_s
        if self.sim_time_s + 1e-9 >= self.next_global_update_s:
            self._update_global_mirror_layer()
            self.next_global_update_s += self.global_update_period_s
        if self.sim_time_s + 1e-9 >= self.next_planner_s:
            self._update_planner()
            self.next_planner_s += self.planner_period_s


def path_min_tape_distance(path: list[tuple[float, float]],
                           world: World) -> float:
    if not path:
        return 0.0
    pts = np.asarray(path, dtype=float)
    dist, _idx, width = distance_to_tape(pts, world.tape_segments)
    clearance = dist - width * 0.5
    return float(np.min(clearance)) if clearance.size else 0.0


def path_min_robot_clearance(path: list[tuple[float, float]],
                             world: World) -> float:
    return path_min_tape_distance(path, world) - ROBOT_LATERAL_CLEARANCE_M


def path_min_cone_clearance(path: list[tuple[float, float]],
                            world: World) -> float:
    if not path or not world.cone_obstacles:
        return math.inf
    pts = np.asarray(path, dtype=float)
    clearance, _idx, _radius = distance_to_cones(pts, world.cone_obstacles)
    return float(np.min(clearance) - ROBOT_LATERAL_CLEARANCE_M)


def complex_maze_forbidden_passages() -> tuple[tuple[float, float, float, float], ...]:
    return (
        (1.25, 1.85, -4.95, -4.18),
        (1.42, 2.12, 0.38, 0.88),
        (-2.18, -1.50, 3.68, 4.20),
    )


def path_enters_rect(path: list[tuple[float, float]],
                     rect: tuple[float, float, float, float],
                     padding_m: float = 0.0) -> bool:
    if not path:
        return False
    xmin, xmax, ymin, ymax = rect
    xmin -= padding_m
    xmax += padding_m
    ymin -= padding_m
    ymax += padding_m
    for x, y in path:
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True
    return False


def count_forbidden_passage_entries(path: list[tuple[float, float]]) -> int:
    return sum(
        1 for rect in complex_maze_forbidden_passages()
        if path_enters_rect(path, rect)
    )


def straight_line_points(start_xy: np.ndarray,
                         goal_xy: np.ndarray,
                         spacing_m: float = 0.05) -> list[tuple[float, float]]:
    start = np.asarray(start_xy, dtype=float)
    goal = np.asarray(goal_xy, dtype=float)
    dist = float(np.linalg.norm(goal - start))
    samples = max(2, int(math.ceil(dist / spacing_m)))
    return [tuple(p) for p in np.linspace(start, goal, samples)]


def configure_sim_detector(sim: LidarLineSimulation,
                           robot_config_value: str) -> None:
    if not robot_config_value:
        return
    config_path = resolve_robot_config_path(robot_config_value)
    loaded = load_robot_detector_config(config_path)
    sim.set_detector_params(
        loaded.params,
        loaded.base_z_offset_m,
        label=f"robot config: {loaded.path}",
    )


def configure_sim_line_layer(sim: LidarLineSimulation,
                             nav2_config_value: str) -> None:
    if not nav2_config_value:
        return
    config_path = resolve_nav2_config_path(nav2_config_value)
    loaded = load_lidar_line_layer_config(config_path)
    sim.set_line_layer_params(
        loaded.params,
        label=f"nav2 lidar_line_layer: {loaded.path}",
    )


def run_benchmark(args: argparse.Namespace) -> int:
    sim = LidarLineSimulation(
        rays=args.rays,
        seed=args.seed,
        max_range_m=args.max_range,
        scenario=args.scenario,
    )
    configure_sim_detector(sim, args.robot_config)
    configure_sim_line_layer(sim, args.nav2_config)
    if args.scenario == "lidar_line_course":
        scan_poses = (
            RobotPose(0.00, 0.00, 0.0),
            RobotPose(0.35, -0.05, 0.0),
            RobotPose(0.70, -0.10, 0.0),
            RobotPose(0.95, -0.25, 0.0),
            RobotPose(1.10, -0.55, 0.0),
            RobotPose(1.35, -0.85, 0.0),
        )
    else:
        scan_poses = (
            RobotPose(0.0, -5.05, math.pi / 2.0),
            RobotPose(0.0, -3.0, math.pi / 2.0),
            RobotPose(0.25, -1.0, math.pi / 2.0),
            RobotPose(0.2, 0.0, math.pi / 2.0),
            RobotPose(0.6, 1.2, math.pi / 2.0),
            RobotPose(0.8, 3.1, math.pi / 2.0),
        )

    tape_points = 0
    accepted_on_tape = 0
    accepted_near_tape = 0
    accepted_total = 0
    false_positive = 0
    timings: list[float] = []
    cluster_count = 0
    max_line_cells = 0

    for pose in scan_poses:
        sim.robot = RobotPose(pose.x, pose.y, pose.heading)
        detection = sim.run_cycle(clear_memory=False)
        max_line_cells = max(max_line_cells, int(np.count_nonzero(sim.line_memory)))
        scan = sim.last_scan
        if scan is None:
            continue

        near_tape = scan.tape_distance <= (scan.tape_width * 0.5 + 0.07)
        in_gate_tape = detection.ground_mask & scan.on_tape
        accepted = detection.accepted_mask

        tape_points += int(np.count_nonzero(in_gate_tape))
        accepted_on_tape += int(np.count_nonzero(accepted & scan.on_tape))
        accepted_near_tape += int(np.count_nonzero(accepted & near_tape))
        accepted_total += int(np.count_nonzero(accepted))
        false_positive += int(np.count_nonzero(accepted & ~near_tape))
        timings.append(detection.elapsed_ms)
        cluster_count += len(detection.clusters)

    sim.robot = default_robot(args.scenario)
    sim.path = astar(
        sim.last_inflated,
        sim.grid_spec,
        np.array([sim.robot.x, sim.robot.y]),
        sim.goal,
    )
    clearance = path_min_tape_distance(sim.path, sim.world)
    cone_clearance = path_min_cone_clearance(sim.path, sim.world)
    recall = accepted_on_tape / max(1, tape_points)
    precision = accepted_near_tape / max(1, accepted_total)
    max_ms = max(timings) if timings else float("inf")
    avg_ms = sum(timings) / max(1, len(timings))
    line_cells = max_line_cells
    # path_following_two keeps lidar_line_layer observation persistence at
    # 0 ms. The global mirror is spatial: each rolling local window can clear
    # cells it no longer sees, so the detector benchmark scores max observed
    # cells while keeping path validity as the end-to-end smoke test.
    path_ok = bool(sim.path) and clearance > -0.08
    if math.isfinite(cone_clearance):
        path_ok = path_ok and cone_clearance > -0.08

    passed = (
        tape_points >= 80
        and line_cells >= 20
        and cluster_count >= 4
        and recall >= 0.50
        and precision >= 0.80
        and max_ms <= 60.0
        and path_ok
    )

    print("LiDAR line benchmark")
    print(f"  detector:         {sim.detector_label}")
    print(f"  line layer:       {sim.line_layer_label}")
    print(f"  persistence:      {sim.persistence_summary()}")
    print(f"  base z offset:    {sim.base_z_offset_m:.3f} m")
    print(f"  rays:             {args.rays}")
    print(f"  scan poses:       {len(scan_poses)}")
    print(f"  tape points:      {tape_points}")
    print(f"  accepted points:  {accepted_total}")
    print(f"  false positives:  {false_positive}")
    print(f"  recall:           {recall:.3f}")
    print(f"  precision:        {precision:.3f}")
    print(f"  clusters:         {cluster_count}")
    print(f"  line cells:       {line_cells}")
    print(f"  detector avg/max: {avg_ms:.2f} ms / {max_ms:.2f} ms")
    print(f"  path nodes:       {len(sim.path)}")
    print(f"  min clearance:    {clearance:.2f} m")
    if math.isfinite(cone_clearance):
        print(f"  cone clearance:   {cone_clearance:.2f} m")
    print(f"  result:           {'PASS' if passed else 'FAIL'}")

    if args.save:
        render_snapshot(sim, Path(args.save), show=False)
        print(f"  snapshot:         {args.save}")

    return 0 if passed else 1


def run_live_headless(args: argparse.Namespace) -> int:
    sim = LidarLineSimulation(
        rays=args.rays,
        seed=args.seed,
        max_range_m=args.max_range,
        scenario=args.scenario,
    )
    configure_sim_detector(sim, args.robot_config)
    configure_sim_line_layer(sim, args.nav2_config)
    sim.run_cycle(clear_memory=True)

    start_xy = np.array([sim.robot.x, sim.robot.y], dtype=float)
    straight_clearance = path_min_robot_clearance(
        straight_line_points(start_xy, sim.goal), sim.world)
    max_line_cells = int(np.count_nonzero(sim.line_memory))
    max_detected_cells = int(np.count_nonzero(sim.last_detected_cells))
    max_remembered_not_current = int(
        np.count_nonzero(sim.line_memory & ~sim.last_detected_cells))
    max_speed = 0.0
    modes_seen: set[str] = set()
    frames = max(1, int(math.ceil(args.duration * RENDER_FPS)))
    for _ in range(frames):
        sim.advance(1.0 / RENDER_FPS)
        modes_seen.add(sim.recovery_mode)
        max_speed = max(max_speed, abs(sim.robot.u))
        max_line_cells = max(max_line_cells,
                             int(np.count_nonzero(sim.line_memory)))
        max_detected_cells = max(
            max_detected_cells,
            int(np.count_nonzero(sim.last_detected_cells)),
        )
        max_remembered_not_current = max(
            max_remembered_not_current,
            int(np.count_nonzero(sim.line_memory & ~sim.last_detected_cells)),
        )

    trail_clearance = path_min_robot_clearance(sim.trail, sim.world)
    path_clearance = path_min_robot_clearance(sim.path, sim.world)
    cone_trail_clearance = path_min_cone_clearance(sim.trail, sim.world)
    cone_path_clearance = path_min_cone_clearance(sim.path, sim.world)
    goal_progress = float(np.linalg.norm(
        np.array([sim.robot.x, sim.robot.y], dtype=float) - start_xy))
    trail = np.asarray(sim.trail, dtype=float)
    lateral_span = (
        float(np.max(trail[:, 0]) - np.min(trail[:, 0]))
        if trail.size else 0.0
    )
    line_cells = int(np.count_nonzero(sim.line_memory))
    detected_cells = int(np.count_nonzero(sim.last_detected_cells))
    path_ok = bool(sim.path) and path_clearance > 0.05
    if math.isfinite(cone_path_clearance):
        path_ok = path_ok and cone_path_clearance > 0.05
    if sim.nav2_planning_hold_ms() > 0:
        persistence_ok = max_remembered_not_current >= 5
    else:
        persistence_ok = max_remembered_not_current == 0
    motion_ok = goal_progress >= 0.40 and max_speed >= 0.15
    trail_ok = trail_clearance > -0.15
    if math.isfinite(cone_trail_clearance):
        trail_ok = trail_ok and cone_trail_clearance > -0.15
    detection_ok = max_line_cells >= 20 and max_detected_cells >= 1
    narrow_entries = (
        count_forbidden_passage_entries(sim.trail)
        if sim.scenario in ("complex_maze", "dead_end_maze") else 0
    )
    maze_ok = (
        (sim.scenario not in ("line_maze", "complex_maze", "dead_end_maze"))
        or (
            straight_clearance < 0.0
            and lateral_span >= 0.35
            and narrow_entries == 0
        )
    )
    passed = (
        detection_ok and persistence_ok and path_ok and motion_ok
        and trail_ok and maze_ok
    )

    print("LiDAR line live headless validation")
    print(f"  detector:                  {sim.detector_label}")
    print(f"  line layer:                {sim.line_layer_label}")
    print(f"  scenario:                  {sim.scenario}")
    print(f"  duration:                  {args.duration:.1f} s")
    print(f"  persistence:               "
          f"{sim.persistence_summary()}")
    print(f"  max line cells:            {max_line_cells}")
    print(f"  max detected cells:        {max_detected_cells}")
    print(f"  current detected cells:    {detected_cells}")
    print(f"  remembered not current:    {max_remembered_not_current}")
    print(f"  final path nodes:          {len(sim.path)}")
    print(f"  final path clearance:      {path_clearance:.2f} m")
    print(f"  straight path clearance:   {straight_clearance:.2f} m")
    print(f"  driven trail clearance:    {trail_clearance:.2f} m")
    if math.isfinite(cone_path_clearance):
        print(f"  cone path clearance:       {cone_path_clearance:.2f} m")
        print(f"  cone trail clearance:      {cone_trail_clearance:.2f} m")
    print(f"  driven lateral span:       {lateral_span:.2f} m")
    print(f"  narrow passage entries:    {narrow_entries}")
    print(f"  nav modes seen:            {', '.join(sorted(modes_seen))}")
    print(f"  goal progress:             {goal_progress:.2f} m")
    print(f"  max speed:                 {max_speed:.2f} m/s")
    print(f"  result:                    {'PASS' if passed else 'FAIL'}")

    if args.save:
        render_snapshot(sim, Path(args.save), show=False)
        print(f"  snapshot:                  {args.save}")

    return 0 if passed else 1


def _plot_tape(ax, world: World) -> None:
    for seg in world.tape_segments:
        xs = [seg.start[0], seg.end[0]]
        ys = [seg.start[1], seg.end[1]]
        ax.plot(xs, ys, color="#111111", linewidth=7,
                solid_capstyle="round", zorder=2)
        ax.plot(xs, ys, color="white", linewidth=4,
                solid_capstyle="round", zorder=3)


def _plot_cones(ax, world: World) -> None:
    import matplotlib.patches as patches

    for cone in world.cone_obstacles:
        circle = patches.Circle(
            (float(cone.center[0]), float(cone.center[1])),
            cone.radius_m,
            facecolor="#f47b20",
            edgecolor="#5c2e0e",
            linewidth=1.4,
            alpha=0.80,
            zorder=4,
        )
        ax.add_patch(circle)


def _draw_robot(ax, pose: RobotPose) -> None:
    ax.scatter([pose.x], [pose.y], s=80, color="#2f80ed",
               edgecolor="white", zorder=8)
    ax.arrow(
        pose.x,
        pose.y,
        0.45 * math.cos(pose.heading),
        0.45 * math.sin(pose.heading),
        width=0.035,
        head_width=0.18,
        head_length=0.18,
        color="#2f80ed",
        zorder=9,
    )


def _style_world_axis(ax, title: str) -> None:
    ax.set_title(title)
    ax.set_xlim(FIELD_X_MIN, FIELD_X_MAX)
    ax.set_ylim(FIELD_Y_MIN, FIELD_Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.55)


def draw_snapshot(sim: LidarLineSimulation, axes: np.ndarray) -> None:
    if sim.last_scan is None or sim.last_detection is None:
        sim.run_cycle()

    scan = sim.last_scan
    detection = sim.last_detection
    assert scan is not None
    assert detection is not None

    ax_world, ax_rssi, ax_cost = axes
    for ax in axes:
        ax.clear()

    ax_world.set_facecolor("#4b695f")
    _plot_tape(ax_world, sim.world)
    _plot_cones(ax_world, sim.world)
    ax_world.scatter(
        scan.points_world[:, 0],
        scan.points_world[:, 1],
        c=scan.intensity,
        cmap="inferno",
        s=4,
        alpha=0.45,
        linewidths=0,
        zorder=4,
    )
    if detection.line_points_world.size:
        ax_world.scatter(
            detection.line_points_world[:, 0],
            detection.line_points_world[:, 1],
            s=24,
            color="#006d8c",
            edgecolor="#002a30",
            linewidth=0.6,
            zorder=6,
        )
    if sim.last_pca_points_world.size:
        ax_world.scatter(
            sim.last_pca_points_world[:, 0],
            sim.last_pca_points_world[:, 1],
            s=14,
            color="#ff5c00",
            alpha=0.75,
            linewidth=0,
            zorder=6,
        )
    if sim.path:
        path = np.asarray(sim.path)
        ax_world.plot(path[:, 0], path[:, 1], color="#61d394",
                      linewidth=2.5, zorder=7)
    ax_world.scatter([sim.goal[0]], [sim.goal[1]], marker="*",
                     s=160, color="#ffe66d", edgecolor="#202020", zorder=8)
    _draw_robot(ax_world, sim.robot)
    _style_world_axis(ax_world, "World scan, line detections, path")

    ax_rssi.set_facecolor("#f7f7f7")
    ax_rssi.scatter(
        scan.points_local[:, 0],
        scan.points_local[:, 1],
        c=scan.intensity,
        cmap="inferno",
        s=5,
        linewidths=0,
        alpha=0.65,
    )
    if detection.line_points_local.size:
        ax_rssi.scatter(
            detection.line_points_local[:, 0],
            detection.line_points_local[:, 1],
            s=16,
            color="#00a7c8",
            edgecolor="black",
            linewidth=0.25,
        )
    ax_rssi.set_title("Sensor-local RSSI cloud")
    ax_rssi.set_xlabel("forward x (m)")
    ax_rssi.set_ylabel("left y (m)")
    ax_rssi.set_xlim(-0.5, sim.max_range_m)
    ax_rssi.set_ylim(-3.3, 3.3)
    ax_rssi.set_aspect("equal", adjustable="box")
    ax_rssi.grid(True, color="#d0d0d0", linewidth=0.5)

    extent = (sim.grid_spec.xmin, sim.grid_spec.xmax,
              sim.grid_spec.ymin, sim.grid_spec.ymax)
    ax_cost.imshow(sim.last_inflated, origin="lower", extent=extent,
                   cmap="Reds", alpha=0.45, interpolation="nearest")
    ax_cost.imshow(sim.line_memory, origin="lower", extent=extent,
                   cmap="Blues", alpha=0.80, interpolation="nearest")
    _plot_tape(ax_cost, sim.world)
    _plot_cones(ax_cost, sim.world)
    if sim.path:
        path = np.asarray(sim.path)
        ax_cost.plot(path[:, 0], path[:, 1], color="#087f5b",
                     linewidth=2.0, zorder=7)
    ax_cost.scatter([sim.goal[0]], [sim.goal[1]], marker="*",
                    s=140, color="#ffd43b", edgecolor="#202020", zorder=8)
    _draw_robot(ax_cost, sim.robot)
    _style_world_axis(ax_cost, "Detected line costmap")

    summary = (
        f"accepted={np.count_nonzero(detection.accepted_mask)}  "
        f"clusters={len(detection.clusters)}  "
        f"line_cells={np.count_nonzero(sim.line_memory)}  "
        f"detector={detection.elapsed_ms:.2f} ms  "
        f"path_nodes={len(sim.path)}"
    )
    ax_world.text(
        0.01,
        0.99,
        summary,
        transform=ax_world.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        bbox=dict(facecolor="#202020", alpha=0.72, edgecolor="none", pad=4),
    )


def render_snapshot(sim: LidarLineSimulation,
                    path: Path | None = None,
                    show: bool = True) -> None:
    import matplotlib.pyplot as plt

    if sim.last_scan is None:
        sim.run_cycle()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    draw_snapshot(sim, axes)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)


class LidarLineGui:
    def __init__(self, sim: LidarLineSimulation):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button

        self.sim = sim
        self.plt = plt
        self.fig, self.axes = plt.subplots(
            1, 3, figsize=(16, 7), constrained_layout=False)
        self.fig.subplots_adjust(bottom=0.16, wspace=0.25)
        self.status = self.fig.text(0.02, 0.02, "", fontsize=10)

        ax_rescan = self.fig.add_axes([0.57, 0.035, 0.10, 0.045])
        ax_step = self.fig.add_axes([0.68, 0.035, 0.10, 0.045])
        ax_reset = self.fig.add_axes([0.79, 0.035, 0.10, 0.045])
        ax_quit = self.fig.add_axes([0.90, 0.035, 0.07, 0.045])

        self.rescan_button = Button(ax_rescan, "Rescan")
        self.step_button = Button(ax_step, "Step")
        self.reset_button = Button(ax_reset, "Reset")
        self.quit_button = Button(ax_quit, "Quit")

        self.rescan_button.on_clicked(self._on_rescan)
        self.step_button.on_clicked(self._on_step)
        self.reset_button.on_clicked(self._on_reset)
        self.quit_button.on_clicked(lambda _event: plt.close(self.fig))

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def show(self) -> None:
        self.sim.run_cycle(clear_memory=True)
        self._redraw()
        self.plt.show()

    def _redraw(self) -> None:
        draw_snapshot(self.sim, self.axes)
        det = self.sim.last_detection
        msg = (
            "Left-click world panel: set goal. "
            "Right-click or shift-click: move robot. "
            "Keys: r=rescan, g=step, x=reset."
        )
        if det is not None:
            msg += (
                f"  Detector {det.elapsed_ms:.2f} ms, "
                f"{len(det.clusters)} clusters."
            )
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()

    def _on_rescan(self, _event=None) -> None:
        self.sim.run_cycle(clear_memory=False)
        self._redraw()

    def _on_step(self, _event=None) -> None:
        self.sim.step_robot()
        self._redraw()

    def _on_reset(self, _event=None) -> None:
        self.sim.reset()
        self.sim.run_cycle(clear_memory=True)
        self._redraw()

    def _on_click(self, event) -> None:
        if event.inaxes is not self.axes[0]:
            return
        if event.xdata is None or event.ydata is None:
            return

        move_robot = event.button == 3 or event.key == "shift"
        if move_robot:
            old = np.array([self.sim.robot.x, self.sim.robot.y], dtype=float)
            new = np.array([event.xdata, event.ydata], dtype=float)
            delta = new - old
            if np.linalg.norm(delta) > 1e-6:
                self.sim.robot.heading = math.atan2(delta[1], delta[0])
            self.sim.robot.x = float(event.xdata)
            self.sim.robot.y = float(event.ydata)
            self.sim._reset_memory_arrays()
        else:
            self.sim.goal = np.array([event.xdata, event.ydata], dtype=float)
        self.sim.run_cycle(clear_memory=move_robot)
        self._redraw()

    def _on_key(self, event) -> None:
        if event.key == "r":
            self._on_rescan()
        elif event.key == "g":
            self._on_step()
        elif event.key == "x":
            self._on_reset()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone RSSI LiDAR line-detection simulator")
    parser.add_argument("--rays", type=int, default=DEFAULT_RAYS,
                        help="Total simulated rays per scan")
    parser.add_argument("--max-range", type=float,
                        default=DEFAULT_MAX_RANGE_M,
                        help="Maximum LiDAR range in meters")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Deterministic scenario seed")
    parser.add_argument("--scenario", type=str, default="lidar_line_course",
                        choices=("competition", "diagonal_strip",
                                 "line_maze", "complex_maze",
                                 "lidar_line_course"),
                        help="Tape scenario to simulate")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run the detector/path benchmark and exit")
    parser.add_argument("--live-headless", action="store_true",
                        help="Run the live dynamics validation without GUI")
    parser.add_argument("--duration", type=float, default=100.0,
                        help="Live/headless simulation duration in seconds")
    parser.add_argument("--robot-config", nargs="?", const="auto", default="",
                        metavar="PATH",
                        help=("Load robot lidar_line_detector.yaml. Use "
                              "'auto' or omit PATH to search ~/code/git."))
    parser.add_argument("--nav2-config", nargs="?", const="auto", default="",
                        metavar="PATH",
                        help=("Load robot nav2_paramsv2.yaml lidar_line_layer. "
                              "Use 'auto' or omit PATH to search ~/code/git."))
    parser.add_argument("--robot-benchmark", action="store_true",
                        help=("Run benchmark with --robot-config auto and "
                              "--nav2-config auto"))
    parser.add_argument("--save", type=str, default="",
                        help="Save a PNG snapshot to this path")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run one scan and print metrics without showing UI")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.robot_benchmark:
        args.benchmark = True
        if not args.robot_config:
            args.robot_config = "auto"
        if not args.nav2_config:
            args.nav2_config = "auto"

    if args.benchmark:
        return run_benchmark(args)
    if args.live_headless:
        if not args.robot_config:
            args.robot_config = "auto"
        if not args.nav2_config:
            args.nav2_config = "auto"
        return run_live_headless(args)

    sim = LidarLineSimulation(
        rays=args.rays,
        seed=args.seed,
        max_range_m=args.max_range,
        scenario=args.scenario,
    )
    configure_sim_detector(sim, args.robot_config)
    configure_sim_line_layer(sim, args.nav2_config)
    sim.run_cycle(clear_memory=True)

    if args.no_gui:
        det = sim.last_detection
        assert det is not None
        print("LiDAR line single scan")
        print(f"  detector:         {sim.detector_label}")
        print(f"  line layer:       {sim.line_layer_label}")
        print(f"  base z offset:    {sim.base_z_offset_m:.3f} m")
        print(f"  rays:             {args.rays}")
        print(f"  accepted points:  {np.count_nonzero(det.accepted_mask)}")
        print(f"  clusters:         {len(det.clusters)}")
        print(f"  line cells:       {np.count_nonzero(sim.line_memory)}")
        print(f"  detector:         {det.elapsed_ms:.2f} ms")
        print(f"  path nodes:       {len(sim.path)}")
        if args.save:
            render_snapshot(sim, Path(args.save), show=False)
            print(f"  snapshot:         {args.save}")
        return 0

    if args.save:
        render_snapshot(sim, Path(args.save), show=False)

    gui = LidarLineGui(sim)
    gui.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
