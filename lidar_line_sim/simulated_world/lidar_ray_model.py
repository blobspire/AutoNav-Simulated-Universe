#!/usr/bin/env python3
"""Shared first-return LiDAR geometry helpers for the lidar-line sims."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Protocol


class CylindricalCone(Protocol):
    center: tuple[float, float]
    radius_m: float
    height_m: float


@dataclass(frozen=True)
class ConeRayHit:
    x: float
    y: float
    z: float
    range_m: float
    layer: int
    azimuth_rad: float
    elevation_rad: float


def _cone_center_xy(cone: CylindricalCone) -> tuple[float, float]:
    cx, cy = cone.center
    return float(cx), float(cy)


def _ray_circle_near_distance(
    origin_xy: tuple[float, float],
    direction_xy: tuple[float, float],
    center_xy: tuple[float, float],
    radius_m: float,
) -> float | None:
    ox, oy = origin_xy
    dx, dy = direction_xy
    cx, cy = center_xy
    fx = ox - cx
    fy = oy - cy
    half_b = fx * dx + fy * dy
    c = fx * fx + fy * fy - radius_m * radius_m
    discriminant = half_b * half_b - c
    if discriminant < 0.0:
        return None

    root = math.sqrt(discriminant)
    near = -half_b - root
    far = -half_b + root
    if near >= 0.0:
        return near
    if far >= 0.0:
        return far
    return None


def cone_occludes_point(
    cones: Iterable[CylindricalCone],
    origin_xy: tuple[float, float],
    sensor_height_m: float,
    point_xyz: tuple[float, float, float],
    margin_m: float = 0.02,
) -> bool:
    """Return true if a finite cone cylinder blocks the sightline to a point."""

    px, py, pz = point_xyz
    ox, oy = origin_xy
    vx = px - ox
    vy = py - oy
    horizontal_distance = math.hypot(vx, vy)
    if horizontal_distance <= margin_m:
        return False

    direction_xy = (vx / horizontal_distance, vy / horizontal_distance)
    slope_z = (pz - sensor_height_m) / horizontal_distance
    for cone in cones:
        hit_distance = _ray_circle_near_distance(
            origin_xy,
            direction_xy,
            _cone_center_xy(cone),
            float(cone.radius_m),
        )
        if hit_distance is None:
            continue
        if hit_distance >= horizontal_distance - margin_m:
            continue
        hit_z = sensor_height_m + hit_distance * slope_z
        if 0.0 <= hit_z <= float(cone.height_m):
            return True
    return False


def raycast_cylindrical_cones(
    cones: Iterable[CylindricalCone],
    origin_xy: tuple[float, float],
    heading_rad: float,
    sensor_height_m: float,
    azimuth_min_rad: float,
    azimuth_max_rad: float,
    horizontal_resolution_rad: float,
    elevation_min_rad: float,
    elevation_max_rad: float,
    layers: int,
    range_min_m: float,
    range_max_m: float,
) -> list[ConeRayHit]:
    """Raycast finite vertical cone cylinders and return first hits per beam.

    This intentionally models first-return geometry only. It prevents the
    synthetic cloud from seeing the back side of a cone through its front side,
    which is the physically impossible artifact that confused the PCA path.
    """

    cone_list = tuple(cones)
    if not cone_list or layers <= 0 or horizontal_resolution_rad <= 0.0:
        return []

    azimuth_count = (
        int(math.floor((azimuth_max_rad - azimuth_min_rad)
                       / horizontal_resolution_rad))
        + 1
    )
    hits: list[ConeRayHit] = []
    for layer in range(layers):
        if layers == 1:
            elevation = 0.5 * (elevation_min_rad + elevation_max_rad)
        else:
            t = layer / float(layers - 1)
            elevation = elevation_min_rad + (
                elevation_max_rad - elevation_min_rad) * t
        cos_elevation = math.cos(elevation)
        if cos_elevation <= 1e-6:
            continue
        tan_elevation = math.tan(elevation)

        for azimuth_idx in range(azimuth_count):
            azimuth = azimuth_min_rad + (
                horizontal_resolution_rad * azimuth_idx)
            if azimuth > azimuth_max_rad + 1e-9:
                continue

            world_angle = heading_rad + azimuth
            direction_xy = (math.cos(world_angle), math.sin(world_angle))
            best_hit: ConeRayHit | None = None
            best_range = math.inf

            for cone in cone_list:
                horizontal_distance = _ray_circle_near_distance(
                    origin_xy,
                    direction_xy,
                    _cone_center_xy(cone),
                    float(cone.radius_m),
                )
                if horizontal_distance is None:
                    continue

                range_m = horizontal_distance / cos_elevation
                if range_m < range_min_m or range_m > range_max_m:
                    continue

                z = sensor_height_m + horizontal_distance * tan_elevation
                if z < 0.0 or z > float(cone.height_m):
                    continue

                if range_m < best_range:
                    best_range = range_m
                    best_hit = ConeRayHit(
                        x=origin_xy[0] + horizontal_distance
                        * direction_xy[0],
                        y=origin_xy[1] + horizontal_distance
                        * direction_xy[1],
                        z=z,
                        range_m=range_m,
                        layer=layer,
                        azimuth_rad=azimuth,
                        elevation_rad=elevation,
                    )

            if best_hit is not None:
                hits.append(best_hit)

    return hits
