#!/usr/bin/env python3
"""Shared geometry for ROS lidar-line course scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SIM_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = SIM_ROOT / "config" / "lidar_line_course.yaml"
SCENARIO_CONFIG_DIR = SIM_ROOT / "config" / "scenarios"
DEFAULT_SCENARIO_ID = "canonical_5ft_gap"


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
class CourseWall:
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    height_m: float
    thickness_m: float


@dataclass(frozen=True)
class AnalysisStation:
    label: str
    x_m: float
    y_min_m: float
    y_max_m: float


@dataclass(frozen=True)
class LidarLineCourse:
    scenario_id: str
    description: str
    tapes: tuple[CourseTape, ...]
    cones: tuple[CourseCone, ...]
    walls: tuple[CourseWall, ...]
    start: tuple[float, float, float]
    goal: tuple[float, float]
    goal_yaw_rad: float
    analysis_stations: tuple[AnalysisStation, ...]
    perp_x_m: float | None
    tape_right_y_m: float | None
    tape_left_y_m: float | None
    required_tape_side_y_m: float | None
    required_cone_side_y_m: float | None
    nominal_centerline_y_m: float | None
    lidar_x_from_nav_center_m: float
    static_walls_in_map: bool
    config_path: Path


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


def _scenario_path(scenario_id: str) -> Path:
    normalized = str(scenario_id).strip().lower().replace("-", "_")
    return SCENARIO_CONFIG_DIR / f"{normalized}.yaml"


def resolve_course_config(path: Path | str | None = None,
                          scenario_id: str | None = None) -> Path:
    if path is not None and str(path).strip():
        return Path(path).expanduser()

    scenario = scenario_id or DEFAULT_SCENARIO_ID
    candidate = _scenario_path(scenario)
    if candidate.is_file():
        return candidate
    if scenario in ("", DEFAULT_SCENARIO_ID, "lidar_line_course"):
        return DEFAULT_CONFIG_PATH
    raise FileNotFoundError(
        f"Unknown lidar-line scenario '{scenario}'. Expected {candidate}"
    )


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


def _float(values: dict[str, object], key: str,
           default: float | None = None) -> float:
    if key not in values:
        if default is None:
            raise KeyError(key)
        return float(default)
    return float(values[key])


def _int(values: dict[str, object], key: str, default: int = 0) -> int:
    return int(values.get(key, default))


def _str(values: dict[str, object], key: str, default: str = "") -> str:
    return str(values.get(key, default))


def _bool(values: dict[str, object], key: str, default: bool = False) -> bool:
    value = values.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _optional_float(values: dict[str, object], key: str) -> float | None:
    if key not in values or values[key] == "":
        return None
    return float(values[key])


def _load_indexed_tapes(values: dict[str, object]) -> tuple[CourseTape, ...]:
    count = _int(values, "tape_count")
    default_width = _float(values, "tape_width_m", 0.12)
    tapes: list[CourseTape] = []
    for idx in range(count):
        prefix = f"tape_{idx}_"
        tapes.append(CourseTape(
            name=_str(values, f"{prefix}name", f"tape_{idx}"),
            start=(
                _float(values, f"{prefix}start_x_m"),
                _float(values, f"{prefix}start_y_m"),
            ),
            end=(
                _float(values, f"{prefix}end_x_m"),
                _float(values, f"{prefix}end_y_m"),
            ),
            width_m=_float(values, f"{prefix}width_m", default_width),
        ))
    return tuple(tapes)


def _load_indexed_cones(values: dict[str, object]) -> tuple[CourseCone, ...]:
    count = _int(values, "cone_count")
    default_radius = _float(values, "cone_radius_m", 0.23)
    default_height = _float(values, "cone_height_m", 0.70)
    cones: list[CourseCone] = []
    for idx in range(count):
        prefix = f"cone_{idx}_"
        radius = _float(values, f"{prefix}radius_m", default_radius)
        center = (
            _float(values, f"{prefix}center_x_m"),
            _float(values, f"{prefix}center_y_m"),
        )
        cones.append(CourseCone(
            name=_str(values, f"{prefix}name", f"cone_{idx}"),
            center=center,
            radius_m=radius,
            height_m=_float(values, f"{prefix}height_m", default_height),
            left_boundary_y_m=_float(
                values, f"{prefix}left_boundary_y_m", center[1] + radius),
        ))
    return tuple(cones)


def _load_indexed_walls(values: dict[str, object]) -> tuple[CourseWall, ...]:
    count = _int(values, "wall_count")
    default_height = _float(values, "wall_height_m", 1.0)
    default_thickness = _float(values, "wall_thickness_m", 0.08)
    walls: list[CourseWall] = []
    for idx in range(count):
        prefix = f"wall_{idx}_"
        walls.append(CourseWall(
            name=_str(values, f"{prefix}name", f"wall_{idx}"),
            start=(
                _float(values, f"{prefix}start_x_m"),
                _float(values, f"{prefix}start_y_m"),
            ),
            end=(
                _float(values, f"{prefix}end_x_m"),
                _float(values, f"{prefix}end_y_m"),
            ),
            height_m=_float(values, f"{prefix}height_m", default_height),
            thickness_m=_float(
                values, f"{prefix}thickness_m", default_thickness),
        ))
    return tuple(walls)


def _load_analysis_stations(
        values: dict[str, object]) -> tuple[AnalysisStation, ...]:
    count = _int(values, "analysis_station_count")
    stations: list[AnalysisStation] = []
    for idx in range(count):
        prefix = f"analysis_station_{idx}_"
        stations.append(AnalysisStation(
            label=_str(values, f"{prefix}label", f"station_{idx}"),
            x_m=_float(values, f"{prefix}x_m"),
            y_min_m=_float(values, f"{prefix}y_min_m"),
            y_max_m=_float(values, f"{prefix}y_max_m"),
        ))
    return tuple(stations)


def _load_legacy_course(values: dict[str, object],
                        config_path: Path) -> LidarLineCourse:
    lidar_x_from_nav = _float(values, "lidar_x_from_nav_center_m")
    tape_width = _float(values, "tape_width_m")

    def lidar_to_nav_x(x_lidar: float) -> float:
        return x_lidar + lidar_x_from_nav

    left_y = _float(values, "left_tape_y_m")
    left_start_lidar_x = _float(values, "left_tape_lidar_start_x_m")
    left_end_lidar_x = left_start_lidar_x + _float(
        values, "left_tape_length_m")
    perp_lidar_x = _float(values, "perpendicular_tape_lidar_x_m")
    perp_x = lidar_to_nav_x(perp_lidar_x)
    perp_left_y = _float(values, "perpendicular_tape_left_y_m")
    perp_right_y = _float(values, "perpendicular_tape_right_y_m")

    cone_radius = _float(values, "cone_radius_m")
    cone_left_boundary = _float(values, "cone_left_boundary_y_m")
    cone_center = (perp_x, cone_left_boundary - cone_radius)
    goal_forward = float(values.get(
        "through_gap_goal_forward_m", values["goal_forward_m"]))
    nominal_y = _float(values, "nominal_centerline_y_m")

    return LidarLineCourse(
        scenario_id=DEFAULT_SCENARIO_ID,
        description="Canonical 10 ft lane / 5 ft tape-to-cone gap",
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
        walls=(),
        start=(0.0, 0.0, 0.0),
        goal=(goal_forward, nominal_y),
        goal_yaw_rad=0.0,
        analysis_stations=(
            AnalysisStation(
                label="through_gap",
                x_m=perp_x,
                y_min_m=_float(values, "required_centerline_cone_side_y_m"),
                y_max_m=_float(values, "required_centerline_tape_side_y_m"),
            ),
        ),
        perp_x_m=perp_x,
        tape_right_y_m=perp_right_y,
        tape_left_y_m=perp_left_y,
        required_tape_side_y_m=_float(
            values, "required_centerline_tape_side_y_m"),
        required_cone_side_y_m=_float(
            values, "required_centerline_cone_side_y_m"),
        nominal_centerline_y_m=nominal_y,
        lidar_x_from_nav_center_m=lidar_x_from_nav,
        static_walls_in_map=True,
        config_path=config_path,
    )


def load_lidar_line_course(path: Path | str | None = None,
                           scenario_id: str | None = None) -> LidarLineCourse:
    config_path = resolve_course_config(path, scenario_id)
    values = load_flat_course_config(config_path)

    if "tape_count" not in values:
        return _load_legacy_course(values, config_path)

    scenario = _str(values, "scenario_id", scenario_id or config_path.stem)
    goal_y = _float(values, "goal_y_m")
    nominal_centerline_y = _optional_float(
        values, "analysis_nominal_centerline_y_m")
    return LidarLineCourse(
        scenario_id=scenario,
        description=_str(values, "description", scenario),
        tapes=_load_indexed_tapes(values),
        cones=_load_indexed_cones(values),
        walls=_load_indexed_walls(values),
        start=(
            _float(values, "start_x_m", 0.0),
            _float(values, "start_y_m", 0.0),
            _float(values, "start_yaw_rad", 0.0),
        ),
        goal=(
            _float(values, "goal_x_m"),
            goal_y,
        ),
        goal_yaw_rad=_float(values, "goal_yaw_rad", 0.0),
        analysis_stations=_load_analysis_stations(values),
        perp_x_m=_optional_float(values, "analysis_perp_x_m"),
        tape_right_y_m=_optional_float(values, "analysis_tape_right_y_m"),
        tape_left_y_m=_optional_float(values, "analysis_tape_left_y_m"),
        required_tape_side_y_m=_optional_float(
            values, "analysis_required_tape_side_y_m"),
        required_cone_side_y_m=_optional_float(
            values, "analysis_required_cone_side_y_m"),
        nominal_centerline_y_m=(
            nominal_centerline_y
            if nominal_centerline_y is not None
            else goal_y
        ),
        lidar_x_from_nav_center_m=_float(
            values, "lidar_x_from_nav_center_m", 0.4348),
        static_walls_in_map=_bool(values, "static_walls_in_map", True),
        config_path=config_path,
    )
