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
    python lidar_line_sim.py --benchmark --rays 11520
"""

from __future__ import annotations

import argparse
import heapq
import math
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np


SENSOR_HEIGHT_M = 0.45
DEFAULT_RAYS = 11520
DEFAULT_MAX_RANGE_M = 9.0
DEFAULT_SEED = 7

FIELD_X_MIN = -3.2
FIELD_X_MAX = 3.2
FIELD_Y_MIN = -6.4
FIELD_Y_MAX = 6.4
GRID_RES_M = 0.08
ROBOT_RADIUS_M = 0.32
LINE_INFLATION_M = 0.50

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
    width_m: float = 0.12


@dataclass
class RobotPose:
    x: float
    y: float
    heading: float


@dataclass(frozen=True)
class World:
    tape_segments: tuple[TapeSegment, ...]


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


@dataclass(frozen=True)
class LoadedRobotConfig:
    path: Path
    params: DetectorParams
    base_z_offset_m: float


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


def default_robot() -> RobotPose:
    return RobotPose(0.0, -0.50, math.pi / 2.0)


def default_goal() -> np.ndarray:
    return np.array([0.85, 5.05], dtype=float)


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
            Path.home() / "code/git/AutoNavB/isaac_ros-dev/src/"
            "autonav_detection/config/lidar_line_detector.yaml",
            Path.home() / "code/git/AutoNav/isaac_ros-dev/src/"
            "autonav_detection/config/lidar_line_detector.yaml",
            Path.home() / "code/git/AutoNav_25-26/isaac_ros-dev/src/"
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


def generate_multiscan_rays(total_rays: int) -> tuple[np.ndarray, np.ndarray]:
    """Approximate SICK multiScan165 output as 16 downward layers."""

    layers = 16
    per_layer = max(1, int(math.ceil(total_rays / layers)))
    azimuths = np.linspace(-math.pi, math.pi, per_layer, endpoint=False)
    elevations = np.deg2rad(np.linspace(-28.0, -4.0, layers))

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
    out = np.empty_like(pts, dtype=float)
    out[..., 0] = pose.x + c * pts[..., 0] - s * pts[..., 1]
    out[..., 1] = pose.y + s * pts[..., 0] + c * pts[..., 1]
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


def detect_lidar_lines(scan: LidarScan,
                       params: DetectorParams | None = None,
                       base_z_offset_m: float = 0.0,
                       ) -> DetectionResult:
    params = params or DetectorParams()
    start = time.perf_counter()

    local = scan.points_local
    base = local.copy()
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
    output_indices: list[int] = []
    seen_output_cells: set[tuple[int, int]] = set()
    for cluster_local_idx in clusters:
        info = _shape_filter_cluster(candidate_xy, cluster_local_idx, params)
        if info is None:
            continue
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
        for original_idx in original_indices:
            target = scan.points_world[original_idx]
            qx = int(round(float(target[0]) / params.output_voxel_size_m))
            qy = int(round(float(target[1]) / params.output_voxel_size_m))
            key = (qx, qy)
            if key in seen_output_cells:
                continue
            seen_output_cells.add(key)
            output_indices.append(int(original_idx))
            if len(output_indices) >= params.max_line_points:
                break
        if len(output_indices) >= params.max_line_points:
            break

    output_idx = np.asarray(output_indices, dtype=int)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return DetectionResult(
        ground_mask=ground_mask,
        candidate_mask=candidate_mask,
        accepted_mask=accepted_mask,
        clusters=tuple(accepted_clusters),
        line_points_local=scan.points_local[output_idx],
        line_points_world=scan.points_world[output_idx],
        elapsed_ms=elapsed_ms,
    )


def mark_points_on_grid(points_xy: np.ndarray, spec: GridSpec) -> np.ndarray:
    grid = np.zeros((spec.ny, spec.nx), dtype=bool)
    if points_xy.size == 0:
        return grid
    ix, iy = spec.world_to_cell(points_xy)
    valid = (ix >= 0) & (ix < spec.nx) & (iy >= 0) & (iy < spec.ny)
    grid[iy[valid], ix[valid]] = True
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
        for dy in range(-2, 3):
            for dx in range(-2, 3):
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


class LidarLineSimulation:
    def __init__(self,
                 rays: int = DEFAULT_RAYS,
                 seed: int = DEFAULT_SEED,
                 max_range_m: float = DEFAULT_MAX_RANGE_M,
                 detector_params: DetectorParams | None = None,
                 base_z_offset_m: float = 0.0,
                 detector_label: str = "sim defaults"):
        self.rays = rays
        self.seed = seed
        self.max_range_m = max_range_m
        self.world = default_world()
        self.grid_spec = GridSpec()
        self.detector_params = (
            detector_params or DetectorParams(range_max_m=max_range_m - 0.5)
        )
        self.base_z_offset_m = base_z_offset_m
        self.detector_label = detector_label
        self.robot = default_robot()
        self.goal = default_goal()
        self.line_memory = np.zeros(
            (self.grid_spec.ny, self.grid_spec.nx), dtype=bool)
        self.last_scan: LidarScan | None = None
        self.last_detection: DetectionResult | None = None
        self.last_inflated = self.line_memory.copy()
        self.path: list[tuple[float, float]] = []

    def reset(self) -> None:
        self.robot = default_robot()
        self.goal = default_goal()
        self.line_memory[:, :] = False
        self.last_scan = None
        self.last_detection = None
        self.last_inflated = self.line_memory.copy()
        self.path = []

    def run_cycle(self, clear_memory: bool = False) -> DetectionResult:
        if clear_memory:
            self.line_memory[:, :] = False
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
        self.line_memory |= cells
        self.last_inflated = inflate_grid(
            self.line_memory, LINE_INFLATION_M, self.grid_spec.res)
        self.path = astar(
            self.last_inflated,
            self.grid_spec,
            np.array([self.robot.x, self.robot.y]),
            self.goal,
        )
        self.last_scan = scan
        self.last_detection = detection
        return detection

    def step_robot(self, distance_m: float = 0.35) -> None:
        if len(self.path) < 2:
            self.run_cycle()
            if len(self.path) < 2:
                return

        remaining = distance_m
        current = np.array([self.robot.x, self.robot.y], dtype=float)
        path_pts = [np.asarray(p, dtype=float) for p in self.path[1:]]
        target = path_pts[-1]

        for point in path_pts:
            delta = point - current
            dist = float(np.linalg.norm(delta))
            if dist < 1e-6:
                current = point
                continue
            if dist >= remaining:
                target = current + delta / dist * remaining
                break
            current = point
            remaining -= dist

        delta = target - np.array([self.robot.x, self.robot.y], dtype=float)
        if np.linalg.norm(delta) > 1e-6:
            self.robot.heading = math.atan2(delta[1], delta[0])
        self.robot.x = float(target[0])
        self.robot.y = float(target[1])
        self.run_cycle()


def path_min_tape_distance(path: list[tuple[float, float]],
                           world: World) -> float:
    if not path:
        return 0.0
    pts = np.asarray(path, dtype=float)
    dist, _idx, width = distance_to_tape(pts, world.tape_segments)
    clearance = dist - width * 0.5
    return float(np.min(clearance)) if clearance.size else 0.0


def configure_sim_detector(sim: LidarLineSimulation,
                           robot_config_value: str) -> None:
    if not robot_config_value:
        return
    config_path = resolve_robot_config_path(robot_config_value)
    loaded = load_robot_detector_config(config_path)
    sim.detector_params = loaded.params
    sim.base_z_offset_m = loaded.base_z_offset_m
    sim.detector_label = f"robot config: {loaded.path}"


def run_benchmark(args: argparse.Namespace) -> int:
    sim = LidarLineSimulation(
        rays=args.rays,
        seed=args.seed,
        max_range_m=args.max_range,
    )
    configure_sim_detector(sim, args.robot_config)
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

    for pose in scan_poses:
        sim.robot = RobotPose(pose.x, pose.y, pose.heading)
        detection = sim.run_cycle(clear_memory=False)
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

    sim.robot = RobotPose(0.0, -5.05, math.pi / 2.0)
    sim.path = astar(
        sim.last_inflated,
        sim.grid_spec,
        np.array([sim.robot.x, sim.robot.y]),
        sim.goal,
    )
    clearance = path_min_tape_distance(sim.path, sim.world)
    recall = accepted_on_tape / max(1, tape_points)
    precision = accepted_near_tape / max(1, accepted_total)
    max_ms = max(timings) if timings else float("inf")
    avg_ms = sum(timings) / max(1, len(timings))
    line_cells = int(np.count_nonzero(sim.line_memory))
    path_ok = bool(sim.path) and clearance > 0.08

    passed = (
        tape_points >= 80
        and line_cells >= 40
        and cluster_count >= 4
        and recall >= 0.55
        and precision >= 0.80
        and max_ms <= 60.0
        and path_ok
    )

    print("LiDAR line benchmark")
    print(f"  detector:         {sim.detector_label}")
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
    print(f"  result:           {'PASS' if passed else 'FAIL'}")

    if args.save:
        render_snapshot(sim, Path(args.save), show=False)
        print(f"  snapshot:         {args.save}")

    return 0 if passed else 1


def _plot_tape(ax, world: World) -> None:
    for seg in world.tape_segments:
        xs = [seg.start[0], seg.end[0]]
        ys = [seg.start[1], seg.end[1]]
        ax.plot(xs, ys, color="#111111", linewidth=7,
                solid_capstyle="round", zorder=2)
        ax.plot(xs, ys, color="white", linewidth=4,
                solid_capstyle="round", zorder=3)


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
            s=16,
            color="#00d6ff",
            edgecolor="#002a30",
            linewidth=0.25,
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
            self.sim.line_memory[:, :] = False
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
    parser.add_argument("--benchmark", action="store_true",
                        help="Run the detector/path benchmark and exit")
    parser.add_argument("--robot-config", nargs="?", const="auto", default="",
                        metavar="PATH",
                        help=("Load robot lidar_line_detector.yaml. Use "
                              "'auto' or omit PATH to search ~/code/git."))
    parser.add_argument("--robot-benchmark", action="store_true",
                        help="Run benchmark with --robot-config auto")
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

    if args.benchmark:
        return run_benchmark(args)

    sim = LidarLineSimulation(
        rays=args.rays,
        seed=args.seed,
        max_range_m=args.max_range,
    )
    configure_sim_detector(sim, args.robot_config)
    sim.run_cycle(clear_memory=True)

    if args.no_gui:
        det = sim.last_detection
        assert det is not None
        print("LiDAR line single scan")
        print(f"  detector:         {sim.detector_label}")
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
