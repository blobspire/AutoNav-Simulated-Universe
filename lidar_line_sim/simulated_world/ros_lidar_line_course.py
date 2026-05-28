#!/usr/bin/env python3
"""ROS sensor harness for the physical lidar-line avoidance course.

This node is intentionally small and explicit: it publishes the robot-facing
topics that the real AutoNav detection/Nav2 stack expects, then integrates the
robot pose from the final /cmd_vel command. It is not a replacement planner or
controller. The point is to feed conservative, beam-faithful synthetic
LiDAR/PCA data into the actual stack.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from lidar_line_course import DEFAULT_SCENARIO_ID, load_lidar_line_course
from lidar_ray_model import raycast_cylindrical_cones

try:
    import rclpy
    from rclpy._rclpy_pybind11 import RCLError
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import TransformStamped, Twist
    from nav_msgs.msg import OccupancyGrid, Odometry
    from sensor_msgs.msg import JointState, LaserScan, PointCloud2, PointField
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Bool, Header
    from tf2_ros import TransformBroadcaster
except ImportError as exc:  # pragma: no cover - only used outside ROS shells.
    raise SystemExit(
        "ros_lidar_line_course.py must run in a sourced ROS 2 environment "
        "with rclpy, tf2_ros, sensor_msgs_py, and Nav2 message packages."
    ) from exc


BASE_LINK_TO_NAV_CENTER_M = 0.225
LIDAR_X_FROM_BASE_LINK_M = 0.6598
LIDAR_Z_FROM_BASE_LINK_M = 0.20568
BASE_LINK_HEIGHT_ABOVE_GROUND_M = 0.11303
GROUND_Z_BASE_M = -BASE_LINK_HEIGHT_ABOVE_GROUND_M
SENSOR_HEIGHT_M = BASE_LINK_HEIGHT_ABOVE_GROUND_M + LIDAR_Z_FROM_BASE_LINK_M
WHEEL_TRACK_M = 0.72326
WHEEL_RADIUS_M = 0.12946
MULTISCAN_LAYERS = 16
LIDAR_AZIMUTH_MIN_RAD = -math.pi / 2.0
LIDAR_AZIMUTH_MAX_RAD = math.pi / 2.0
LIDAR_HARDWARE_ELEVATION_MIN_RAD = math.radians(-35.0)
LIDAR_HARDWARE_ELEVATION_MAX_RAD = math.radians(7.5)
LIDAR_HORIZONTAL_RES_RAD = math.radians(0.5)

MAX_LINEAR_SPEED_MPS = 0.25
MAX_ANGULAR_SPEED_RADPS = 1.0
CMD_TIMEOUT_S = 0.40
RANGE_MIN_M = 0.20
RANGE_MAX_M = 8.5
FLOOR_RSSI = 30000.0
TAPE_RSSI = 52000.0
CONE_RSSI = 33000.0
CONE_REFLECTOR_RSSI = 50000.0


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


def _stamp_to_float(stamp: Time) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class LidarLineCourseHarness(Node):
    def __init__(self) -> None:
        super().__init__("lidar_line_course_harness")
        self.declare_parameter(
            "course_config",
            "",
        )
        self.declare_parameter("scenario", DEFAULT_SCENARIO_ID)
        self.declare_parameter("publish_ground_truth_pca", False)
        self.declare_parameter("cloud_rate_hz", 10.0)
        self.declare_parameter("odom_rate_hz", 50.0)
        self.declare_parameter("map_rate_hz", 1.0)
        # Deprecated compatibility knobs. The current harness uses
        # beam-first returns instead of sampled floor/tape/cone point grids.
        self.declare_parameter("floor_spacing_m", 0.10)
        self.declare_parameter("tape_spacing_m", 0.025)
        self.declare_parameter("cone_spacing_m", 0.06)
        self.declare_parameter("cmd_latency_s", 0.08)
        self.declare_parameter("linear_time_constant_s", 0.20)
        self.declare_parameter("angular_time_constant_s", 0.18)
        self.declare_parameter("linear_deadband_mps", 0.02)
        self.declare_parameter("angular_deadband_radps", 0.04)

        course_path = str(self.get_parameter("course_config").value).strip()
        if course_path == "__auto__":
            course_path = ""
        scenario = str(self.get_parameter("scenario").value).strip()
        self.course = load_lidar_line_course(
            Path(course_path).expanduser() if course_path else None,
            scenario_id=scenario,
        )
        self.publish_ground_truth_pca = bool(
            self.get_parameter("publish_ground_truth_pca").value)
        self.cmd_latency_s = max(
            0.0, float(self.get_parameter("cmd_latency_s").value))
        self.linear_time_constant_s = max(
            0.0, float(self.get_parameter("linear_time_constant_s").value))
        self.angular_time_constant_s = max(
            0.0, float(self.get_parameter("angular_time_constant_s").value))
        self.linear_deadband_mps = max(
            0.0, float(self.get_parameter("linear_deadband_mps").value))
        self.angular_deadband_radps = max(
            0.0, float(self.get_parameter("angular_deadband_radps").value))

        sensor_qos = QoSProfile(depth=5)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        map_qos = QoSProfile(depth=1)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.cloud_pub = self.create_publisher(
            PointCloud2, "/cloud_all_fields_fullframe", sensor_qos)
        self.pca_gt_pub = (
            self.create_publisher(
                PointCloud2, "/scan_pca_filtered_points", sensor_qos)
            if self.publish_ground_truth_pca else None
        )
        self.scan_pub = self.create_publisher(
            LaserScan, "/scan_fullframe", sensor_qos)
        self.map_pub = self.create_publisher(
            OccupancyGrid, "/map_padded", map_qos)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.local_odom_pub = self.create_publisher(
            Odometry, "/local_ekf/odom", 10)
        self.joint_state_pub = self.create_publisher(
            JointState, "/joint_states", 10)
        self.autonomous_pub = self.create_publisher(
            Bool, "/autonomous_mode", 1)

        self.cmd_sub = self.create_subscription(
            Twist, "/cmd_vel", self._cmd_vel_callback, 10)

        self.tf_pub = TransformBroadcaster(self)

        self.nav_x = self.course.start[0]
        self.nav_y = self.course.start[1]
        self.heading = self.course.start[2]
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.applied_v = 0.0
        self.applied_w = 0.0
        self.left_wheel_position = 0.0
        self.right_wheel_position = 0.0
        self.pending_commands: list[tuple[float, float, float]] = []
        self.last_cmd_s = -math.inf
        self.last_step_s: float | None = None

        odom_period = 1.0 / max(
            1.0, float(self.get_parameter("odom_rate_hz").value))
        cloud_period = 1.0 / max(
            1.0, float(self.get_parameter("cloud_rate_hz").value))
        map_period = 1.0 / max(
            0.1, float(self.get_parameter("map_rate_hz").value))
        self.create_timer(odom_period, self._step_and_publish_odom)
        self.create_timer(cloud_period, self._publish_sensor_frame)
        self.create_timer(map_period, self._publish_map)
        self._publish_map()

        self.get_logger().info(
            "Loaded lidar-line scenario '%s': start=(%.2f, %.2f, %.1fdeg) "
            "goal=(%.2f, %.2f) tapes=%d cones=%d config=%s "
            "publish_ground_truth_pca=%s"
            % (
                self.course.scenario_id,
                self.course.start[0],
                self.course.start[1],
                math.degrees(self.course.start[2]),
                self.course.goal[0],
                self.course.goal[1],
                len(self.course.tapes),
                len(self.course.cones),
                self.course.config_path,
                self.publish_ground_truth_pca,
            )
        )

    def _cmd_vel_callback(self, msg: Twist) -> None:
        now_s = _stamp_to_float(self.get_clock().now().to_msg())
        cmd_v = _clamp(float(msg.linear.x),
                       -MAX_LINEAR_SPEED_MPS, MAX_LINEAR_SPEED_MPS)
        cmd_w = _clamp(float(msg.angular.z),
                       -MAX_ANGULAR_SPEED_RADPS, MAX_ANGULAR_SPEED_RADPS)
        self.pending_commands.append((now_s + self.cmd_latency_s, cmd_v, cmd_w))
        self.last_cmd_s = now_s

    @staticmethod
    def _make_static_transform(stamp: Time,
                               parent: str,
                               child: str,
                               x: float,
                               y: float,
                               z: float,
                               qx: float,
                               qy: float,
                               qz: float,
                               qw: float) -> TransformStamped:
        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = parent
        msg.child_frame_id = child
        msg.transform.translation.x = float(x)
        msg.transform.translation.y = float(y)
        msg.transform.translation.z = float(z)
        msg.transform.rotation.x = float(qx)
        msg.transform.rotation.y = float(qy)
        msg.transform.rotation.z = float(qz)
        msg.transform.rotation.w = float(qw)
        return msg

    def _base_pose_world(self) -> tuple[float, float]:
        c = math.cos(self.heading)
        s = math.sin(self.heading)
        return (
            self.nav_x - BASE_LINK_TO_NAV_CENTER_M * c,
            self.nav_y - BASE_LINK_TO_NAV_CENTER_M * s,
        )

    def _lidar_origin_world(self) -> tuple[float, float]:
        bx, by = self._base_pose_world()
        c = math.cos(self.heading)
        s = math.sin(self.heading)
        return (
            bx + LIDAR_X_FROM_BASE_LINK_M * c,
            by + LIDAR_X_FROM_BASE_LINK_M * s,
        )

    def _world_to_base(self,
                       x: float,
                       y: float,
                       z_above_ground: float = 0.0,
                       ) -> tuple[float, float, float]:
        bx, by = self._base_pose_world()
        dx = x - bx
        dy = y - by
        c = math.cos(self.heading)
        s = math.sin(self.heading)
        return (
            c * dx + s * dy,
            -s * dx + c * dy,
            GROUND_Z_BASE_M + z_above_ground,
        )

    @staticmethod
    def _base_to_lidar(base: tuple[float, float, float]
                       ) -> tuple[float, float, float]:
        bx, by, bz = base
        return (
            bx - LIDAR_X_FROM_BASE_LINK_M,
            -by,
            -(bz - LIDAR_Z_FROM_BASE_LINK_M),
        )

    def _append_cloud_point(
        self,
        out: list[tuple[float, float, float, float,
                        float, float, float, float]],
        world_x: float,
        world_y: float,
        z_above_ground: float,
        intensity: float,
        reflector: bool,
        layer: int = 0,
    ) -> None:
        lidar = self._base_to_lidar(
            self._world_to_base(world_x, world_y, z_above_ground))
        rng = math.sqrt(lidar[0] ** 2 + lidar[1] ** 2 + lidar[2] ** 2)
        if rng < RANGE_MIN_M or rng > RANGE_MAX_M:
            return
        out.append(self._cloud_tuple(lidar, rng, intensity, reflector, layer))

    @staticmethod
    def _cloud_tuple(lidar: tuple[float, float, float],
                     rng: float,
                     intensity: float,
                     reflector: bool,
                     layer: int) -> tuple[float, float, float, float,
                                          float, float, float, float]:
        return (
            float(lidar[0]),
            float(lidar[1]),
            float(lidar[2]),
            float(intensity),
            float(rng),
            float(layer),
            0.0,
            1.0 if reflector else 0.0,
        )

    def _tape_distance(self, x: float, y: float) -> float:
        best = math.inf
        for tape in self.course.tapes:
            ax, ay = tape.start
            bx, by = tape.end
            abx = bx - ax
            aby = by - ay
            denom = abx * abx + aby * aby
            if denom <= 1e-12:
                dist = math.hypot(x - ax, y - ay)
            else:
                t = max(0.0, min(1.0, ((x - ax) * abx + (y - ay) * aby) / denom))
                qx = ax + t * abx
                qy = ay + t * aby
                dist = math.hypot(x - qx, y - qy)
            best = min(best, dist)
        return best

    def _raycast_cone_hits(self):
        return raycast_cylindrical_cones(
            self.course.cones,
            self._lidar_origin_world(),
            self.heading,
            SENSOR_HEIGHT_M,
            LIDAR_AZIMUTH_MIN_RAD,
            LIDAR_AZIMUTH_MAX_RAD,
            LIDAR_HORIZONTAL_RES_RAD,
            LIDAR_HARDWARE_ELEVATION_MIN_RAD,
            LIDAR_HARDWARE_ELEVATION_MAX_RAD,
            MULTISCAN_LAYERS,
            RANGE_MIN_M,
            RANGE_MAX_M,
        )

    def _build_cloud_points(self) -> list[tuple[float, float, float, float,
                                               float, float, float, float]]:
        """Generate first-return synthetic LiDAR points.

        Older versions sampled floor grids and tape centerlines directly,
        which made retroreflective tape visible even when no beam intersected
        it. This path uses one ordered return per physical beam: cone surface
        beats floor/tape when closer, otherwise the downward beam returns the
        floor point with the reflector bit set only if the finite tape geometry
        is actually under that beam.
        """

        points: list[tuple[float, float, float, float,
                           float, float, float, float]] = []
        cone_hits = {
            (hit.layer, round(hit.azimuth_rad, 9)): hit
            for hit in self._raycast_cone_hits()
        }
        azimuth_count = (
            int(math.floor((LIDAR_AZIMUTH_MAX_RAD - LIDAR_AZIMUTH_MIN_RAD)
                           / LIDAR_HORIZONTAL_RES_RAD))
            + 1
        )
        origin_x, origin_y = self._lidar_origin_world()
        for layer in range(MULTISCAN_LAYERS):
            if MULTISCAN_LAYERS == 1:
                elevation = 0.5 * (
                    LIDAR_HARDWARE_ELEVATION_MIN_RAD
                    + LIDAR_HARDWARE_ELEVATION_MAX_RAD)
            else:
                frac = layer / float(MULTISCAN_LAYERS - 1)
                elevation = LIDAR_HARDWARE_ELEVATION_MIN_RAD + (
                    LIDAR_HARDWARE_ELEVATION_MAX_RAD
                    - LIDAR_HARDWARE_ELEVATION_MIN_RAD) * frac
            cos_elevation = math.cos(elevation)
            tan_elevation = math.tan(elevation)
            if cos_elevation <= 1e-6:
                continue

            for azimuth_idx in range(azimuth_count):
                azimuth = LIDAR_AZIMUTH_MIN_RAD + (
                    LIDAR_HORIZONTAL_RES_RAD * azimuth_idx)
                if azimuth > LIDAR_AZIMUTH_MAX_RAD + 1e-9:
                    continue

                best_range = math.inf
                best: tuple[float, float, float, float, bool] | None = None
                cone_hit = cone_hits.get((layer, round(azimuth, 9)))
                if cone_hit is not None:
                    reflector = cone_hit.z > 0.28
                    best_range = cone_hit.range_m
                    best = (
                        cone_hit.x,
                        cone_hit.y,
                        cone_hit.z,
                        CONE_REFLECTOR_RSSI if reflector else CONE_RSSI,
                        reflector,
                    )

                if tan_elevation < -1e-6:
                    horizontal_distance = -SENSOR_HEIGHT_M / tan_elevation
                    floor_range = horizontal_distance / cos_elevation
                    if RANGE_MIN_M <= floor_range <= RANGE_MAX_M:
                        world_angle = self.heading + azimuth
                        floor_x = (
                            origin_x + horizontal_distance
                            * math.cos(world_angle))
                        floor_y = (
                            origin_y + horizontal_distance
                            * math.sin(world_angle))
                        if floor_range < best_range:
                            on_tape = self._point_on_tape(floor_x, floor_y)
                            best_range = floor_range
                            best = (
                                floor_x,
                                floor_y,
                                0.0,
                                TAPE_RSSI if on_tape else FLOOR_RSSI,
                                on_tape,
                            )

                if best is None:
                    continue
                world_x, world_y, z_above_ground, intensity, reflector = best
                self._append_cloud_point(
                    points,
                    world_x,
                    world_y,
                    z_above_ground,
                    intensity,
                    reflector,
                    layer,
                )
        return points

    def _point_on_tape(self, x: float, y: float) -> bool:
        for tape in self.course.tapes:
            if self._point_tape_distance(x, y, tape.start, tape.end) <= (
                    0.5 * tape.width_m):
                return True
        return False

    @staticmethod
    def _point_tape_distance(x: float,
                             y: float,
                             start: tuple[float, float],
                             end: tuple[float, float]) -> float:
        ax, ay = start
        bx, by = end
        abx = bx - ax
        aby = by - ay
        denom = abx * abx + aby * aby
        if denom <= 1e-12:
            return math.hypot(x - ax, y - ay)
        t = max(0.0, min(1.0, ((x - ax) * abx + (y - ay) * aby) / denom))
        qx = ax + t * abx
        qy = ay + t * aby
        return math.hypot(x - qx, y - qy)

    def _build_cone_cloud_points(self) -> list[tuple[float, float, float, float,
                                                     float, float, float, float]]:
        points: list[tuple[float, float, float, float,
                           float, float, float, float]] = []
        for hit in self._raycast_cone_hits():
            reflector = hit.z > 0.28
            self._append_cloud_point(
                points, hit.x, hit.y, hit.z,
                CONE_REFLECTOR_RSSI if reflector else CONE_RSSI,
                reflector,
                layer=hit.layer,
            )
        return points

    def _make_cloud(self,
                    stamp: Time,
                    points: list[tuple[float, float, float, float,
                                       float, float, float, float]],
                    topic_frame: str = "lidar_footprint") -> PointCloud2:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="i", offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name="range", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="layer", offset=20, datatype=PointField.FLOAT32, count=1),
            PointField(name="echo", offset=24, datatype=PointField.FLOAT32, count=1),
            PointField(name="reflector", offset=28, datatype=PointField.FLOAT32, count=1),
        ]
        header = Header()
        header.stamp = stamp
        header.frame_id = topic_frame
        cloud = point_cloud2.create_cloud(
            header=header,
            fields=fields,
            points=points,
        )
        return cloud

    def _pca_ground_truth_points(self) -> list[tuple[float, float, float,
                                                     float, float, float,
                                                     float, float]]:
        points: list[tuple[float, float, float, float,
                           float, float, float, float]] = []
        for hit in self._raycast_cone_hits():
            # Ground-truth PCA isolates physical obstacles from reflective
            # tape; reflector state is deliberately false on this debug path.
            self._append_cloud_point(points, hit.x, hit.y, hit.z, CONE_RSSI,
                                     False, layer=hit.layer)
        return points

    def _publish_scan_fullframe(self,
                                stamp: Time,
                                points: list[tuple[float, float, float, float,
                                                   float, float, float, float]]
                                ) -> None:
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = "lidar_footprint"
        scan.angle_min = -math.pi
        scan.angle_max = math.pi
        scan.angle_increment = math.radians(0.5)
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = 0.20
        scan.range_max = 25.0
        count = int(round((scan.angle_max - scan.angle_min)
                          / scan.angle_increment)) + 1
        ranges = [math.inf] * count
        for x, y, _z, _i, rng, _layer, _echo, _reflector in points:
            angle = math.atan2(y, x)
            idx = int(round((angle - scan.angle_min) / scan.angle_increment))
            if 0 <= idx < count and rng < ranges[idx]:
                ranges[idx] = float(rng)
        scan.ranges = ranges
        self.scan_pub.publish(scan)

    def _publish_sensor_frame(self) -> None:
        stamp = self.get_clock().now().to_msg()
        cloud_points = self._build_cloud_points()
        self.cloud_pub.publish(self._make_cloud(stamp, cloud_points))
        if self.pca_gt_pub is not None:
            self.pca_gt_pub.publish(
                self._make_cloud(stamp, self._pca_ground_truth_points()))
        self._publish_scan_fullframe(stamp, cloud_points)

    def _step_and_publish_odom(self) -> None:
        now = self.get_clock().now().to_msg()
        now_s = _stamp_to_float(now)
        if self.last_step_s is None:
            self.last_step_s = now_s
        dt = max(0.0, min(0.10, now_s - self.last_step_s))
        self.last_step_s = now_s

        while self.pending_commands and self.pending_commands[0][0] <= now_s:
            _apply_s, self.cmd_v, self.cmd_w = self.pending_commands.pop(0)

        if now_s - self.last_cmd_s > CMD_TIMEOUT_S:
            target_v = 0.0
            target_w = 0.0
        else:
            target_v = self.cmd_v
            target_w = self.cmd_w

        if abs(target_v) < self.linear_deadband_mps:
            target_v = 0.0
        if abs(target_w) < self.angular_deadband_radps:
            target_w = 0.0

        self.applied_v = self._first_order_response(
            self.applied_v, target_v, dt, self.linear_time_constant_s)
        self.applied_w = self._first_order_response(
            self.applied_w, target_w, dt, self.angular_time_constant_s)

        self.nav_x += self.applied_v * math.cos(self.heading) * dt
        self.nav_y += self.applied_v * math.sin(self.heading) * dt
        self.heading = math.atan2(
            math.sin(self.heading + self.applied_w * dt),
            math.cos(self.heading + self.applied_w * dt),
        )
        self._integrate_wheel_joints(dt, self.applied_v, self.applied_w)
        self._publish_dynamic_transforms(now)
        self._publish_odom(now, self.applied_v, self.applied_w)
        self._publish_joint_states(now)
        self.autonomous_pub.publish(Bool(data=True))

    def _integrate_wheel_joints(self, dt: float, v: float, w: float) -> None:
        left_linear = v - w * WHEEL_TRACK_M * 0.5
        right_linear = v + w * WHEEL_TRACK_M * 0.5
        self.left_wheel_position += left_linear / WHEEL_RADIUS_M * dt
        self.right_wheel_position += right_linear / WHEEL_RADIUS_M * dt

    def _publish_joint_states(self, stamp: Time) -> None:
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = ["Left_Wheel", "Right_Wheel"]
        msg.position = [self.left_wheel_position, self.right_wheel_position]
        msg.velocity = [
            (self.applied_v - self.applied_w * WHEEL_TRACK_M * 0.5)
            / WHEEL_RADIUS_M,
            (self.applied_v + self.applied_w * WHEEL_TRACK_M * 0.5)
            / WHEEL_RADIUS_M,
        ]
        self.joint_state_pub.publish(msg)

    @staticmethod
    def _first_order_response(current: float,
                              target: float,
                              dt: float,
                              tau: float) -> float:
        if tau <= 1e-6:
            return target
        alpha = 1.0 - math.exp(-max(0.0, dt) / tau)
        return current + alpha * (target - current)

    def _publish_dynamic_transforms(self, stamp: Time) -> None:
        transforms: list[TransformStamped] = []
        transforms.append(self._make_static_transform(
            stamp, "map", "odom", 0.0, 0.0, 0.0, *_yaw_quaternion(0.0)))
        bx, by = self._base_pose_world()
        transforms.append(self._make_static_transform(
            stamp, "odom", "base_link", bx, by, 0.0,
            *_yaw_quaternion(self.heading)))
        self.tf_pub.sendTransform(transforms)

    def _publish_odom(self, stamp: Time, v: float, w: float) -> None:
        bx, by = self._base_pose_world()
        qx, qy, qz, qw = _yaw_quaternion(self.heading)
        for publisher, topic in (
                (self.odom_pub, "odom"), (self.local_odom_pub, "local_ekf/odom")):
            _ = topic
            msg = Odometry()
            msg.header.stamp = stamp
            msg.header.frame_id = "odom"
            msg.child_frame_id = "base_link"
            msg.pose.pose.position.x = bx
            msg.pose.pose.position.y = by
            msg.pose.pose.position.z = 0.0
            msg.pose.pose.orientation.x = qx
            msg.pose.pose.orientation.y = qy
            msg.pose.pose.orientation.z = qz
            msg.pose.pose.orientation.w = qw
            msg.twist.twist.linear.x = v
            msg.twist.twist.angular.z = w
            publisher.publish(msg)

    def _publish_map(self) -> None:
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.info.resolution = 0.05
        min_x, min_y, max_x, max_y = self._map_bounds(msg.info.resolution)
        msg.info.width = int(math.ceil((max_x - min_x) / msg.info.resolution))
        msg.info.height = int(math.ceil((max_y - min_y) / msg.info.resolution))
        msg.info.origin.position.x = min_x
        msg.info.origin.position.y = min_y
        msg.info.origin.orientation.w = 1.0
        msg.data = [0] * (msg.info.width * msg.info.height)
        self.map_pub.publish(msg)

    def _map_bounds(self, resolution: float) -> tuple[float, float, float, float]:
        xs = [self.course.start[0], self.course.goal[0], -3.0, 5.0]
        ys = [self.course.start[1], self.course.goal[1], -4.0, 4.0]
        for tape in self.course.tapes:
            xs.extend((tape.start[0], tape.end[0]))
            ys.extend((tape.start[1], tape.end[1]))
        for cone in self.course.cones:
            xs.extend((cone.center[0] - cone.radius_m,
                       cone.center[0] + cone.radius_m))
            ys.extend((cone.center[1] - cone.radius_m,
                       cone.center[1] + cone.radius_m))
        margin = 2.0
        min_x = math.floor((min(xs) - margin) / resolution) * resolution
        min_y = math.floor((min(ys) - margin) / resolution) * resolution
        max_x = math.ceil((max(xs) + margin) / resolution) * resolution
        max_y = math.ceil((max(ys) + margin) / resolution) * resolution
        return min_x, min_y, max_x, max_y


def main(argv: list[str] | None = None) -> int:
    rclpy.init(args=sys.argv if argv is None else argv)
    node = LidarLineCourseHarness()
    try:
        rclpy.spin(node)
    except RCLError:
        if rclpy.ok():
            raise
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
