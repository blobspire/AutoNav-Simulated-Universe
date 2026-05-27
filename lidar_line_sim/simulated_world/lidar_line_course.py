#!/usr/bin/env python3
"""Shared geometry for the physical lidar-line avoidance test course."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "lidar_line_course.yaml"
)


@dataclass(frozen=True)
class CourseTape:
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    width_m: float


@dataclass(frozen=True)
class CourseCone:
    name: str
    center: tuple[float, float]
    radius_m: float
    height_m: float
    left_boundary_y_m: float


@dataclass(frozen=True)
class LidarLineCourse:
    tapes: tuple[CourseTape, ...]
    cones: tuple[CourseCone, ...]
    goal: tuple[float, float]
    perp_x_m: float
    tape_right_y_m: float
    tape_left_y_m: float
    required_tape_side_y_m: float
    required_cone_side_y_m: float
    nominal_centerline_y_m: float
    lidar_x_from_nav_center_m: float


def _parse_scalar(raw: str) -> object:
    value = raw.split("#", 1)[0].strip()
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


def load_flat_course_config(path: Path | None = None) -> dict[str, object]:
    """Load the flat YAML course file without requiring PyYAML."""

    config_path = path or DEFAULT_CONFIG_PATH
    values: dict[str, object] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        values[key.strip()] = _parse_scalar(raw_value)
    return values


def _float(values: dict[str, object], key: str) -> float:
    return float(values[key])


def load_lidar_line_course(path: Path | None = None) -> LidarLineCourse:
    values = load_flat_course_config(path)
    lidar_x_from_nav = _float(values, "lidar_x_from_nav_center_m")
    tape_width = _float(values, "tape_width_m")

    def lidar_to_nav_x(x_lidar: float) -> float:
        return x_lidar + lidar_x_from_nav

    left_y = _float(values, "left_tape_y_m")
    left_start_lidar_x = _float(values, "left_tape_lidar_start_x_m")
    left_end_lidar_x = left_start_lidar_x + _float(values, "left_tape_length_m")
    perp_lidar_x = _float(values, "perpendicular_tape_lidar_x_m")
    perp_x = lidar_to_nav_x(perp_lidar_x)
    perp_left_y = _float(values, "perpendicular_tape_left_y_m")
    perp_right_y = _float(values, "perpendicular_tape_right_y_m")

    cone_radius = _float(values, "cone_radius_m")
    cone_left_boundary = _float(values, "cone_left_boundary_y_m")
    cone_center = (perp_x, cone_left_boundary - cone_radius)

    return LidarLineCourse(
        tapes=(
            CourseTape(
                name="left_tape",
                start=(lidar_to_nav_x(left_start_lidar_x), left_y),
                end=(lidar_to_nav_x(left_end_lidar_x), left_y),
                width_m=tape_width,
            ),
            CourseTape(
                name="perpendicular_tape",
                start=(perp_x, perp_left_y),
                end=(perp_x, perp_right_y),
                width_m=tape_width,
            ),
        ),
        cones=(
            CourseCone(
                name="dot_cone",
                center=cone_center,
                radius_m=cone_radius,
                height_m=_float(values, "cone_height_m"),
                left_boundary_y_m=cone_left_boundary,
            ),
        ),
        goal=(_float(values, "goal_forward_m"), 0.0),
        perp_x_m=perp_x,
        tape_right_y_m=perp_right_y,
        tape_left_y_m=perp_left_y,
        required_tape_side_y_m=_float(
            values, "required_centerline_tape_side_y_m"),
        required_cone_side_y_m=_float(
            values, "required_centerline_cone_side_y_m"),
        nominal_centerline_y_m=_float(values, "nominal_centerline_y_m"),
        lidar_x_from_nav_center_m=lidar_x_from_nav,
    )
