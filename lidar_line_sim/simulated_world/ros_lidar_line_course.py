#!/usr/bin/env python3
"""ROS sensor harness for the physical lidar-line avoidance course.

This node is intentionally small and explicit: it publishes the robot-facing
topics that the real AutoNav detection/Nav2 stack expects, then integrates the
robot pose from the final /cmd_vel command. It is not a replacement planner or
controller. The point is to feed canonical synthetic lidar/PCA data into the
actual stack.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from lidar_line_course import load_lidar_line_course
from lidar_ray_model import cone_occludes_point, raycast_cylindrical_cones

try:
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import TransformStamped, Twist
    from nav_msgs.msg import OccupancyGrid, Odometry
    from sensor_msgs.msg import LaserScan, PointCloud2, PointField
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Bool, Header
    from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
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
MULTISCAN_LAYERS = 16
LIDAR_AZIMUTH_MIN_RAD = -math.pi / 2.0
LIDAR_AZIMUTH_MAX_RAD = math.pi / 2.0
LIDAR_HARDWARE_ELEVATION_MIN_RAD = math.radians(-35.0)
LIDAR_HARDWARE_ELEVATION_MAX_RAD = math.radians(7.5)
LIDAR_HORIZONTAL_RES_RAD = math.radians(0.5)

MAX_LINEAR_SPEED_MPS = 0.25
MAX_ANGULAR_SPEED_RADPS = 0.65
CMD_TIMEOUT_S = 0.40


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


def _rpy_quaternion(roll: float,
                    pitch: float,
                    yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _stamp_to_float(stamp: Time) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class LidarLineCourseHarness(Node):
    def __init__(self) -> None:
        super().__init__("lidar_line_course_harness")
        self.declare_parameter(
            "course_config",
            str(Path(__file__).resolve().parents[1]
                / "config" / "lidar_line_course.yaml"),
        )
        self.declare_parameter("publish_ground_truth_pca", False)
        self.declare_parameter("cloud_rate_hz", 10.0)
        self.declare_parameter("odom_rate_hz", 50.0)
        self.declare_parameter("map_rate_hz", 1.0)
        self.declare_parameter("floor_spacing_m", 0.10)
        self.declare_parameter("tape_spacing_m", 0.025)
        self.declare_parameter("cone_spacing_m", 0.06)

        course_path = Path(
            str(self.get_parameter("course_config").value)).expanduser()
        self.course = load_lidar_line_course(course_path)
        self.publish_ground_truth_pca = bool(
            self.get_parameter("publish_ground_truth_pca").value)
        self.floor_spacing_m = float(
            self.get_parameter("floor_spacing_m").value)
        self.tape_spacing_m = float(
            self.get_parameter("tape_spacing_m").value)
        self.cone_spacing_m = float(
            self.get_parameter("cone_spacing_m").value)

        sensor_qos = QoSProfile(depth=5)
        sensor_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        map_qos = QoSProfile(depth=1)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.cloud_pub = self.create_publisher(
            PointCloud2, "/cloud_all_fields_fullframe", sensor_qos)
        self.pca_gt_pub = self.create_publisher(
            PointCloud2, "/scan_pca_filtered_points", sensor_qos)
        self.scan_pub = self.create_publisher(
            LaserScan, "/scan_fullframe", sensor_qos)
        self.map_pub = self.create_publisher(
            OccupancyGrid, "/map_padded", map_qos)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.local_odom_pub = self.create_publisher(
            Odometry, "/local_ekf/odom", 10)
        self.autonomous_pub = self.create_publisher(
            Bool, "/autonomous_mode", 1)

        self.cmd_sub = self.create_subscription(
            Twist, "/cmd_vel", self._cmd_vel_callback, 10)

        self.tf_pub = TransformBroadcaster(self)
        self.static_tf_pub = StaticTransformBroadcaster(self)
        self._publish_static_transforms()

        self.nav_x = 0.0
        self.nav_y = 0.0
        self.heading = 0.0
        self.cmd_v = 0.0
        self.cmd_w = 0.0
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
            "Loaded lidar-line course: perp_x=%.3f tape_y=[%.2f, %.2f] "
            "goal=(%.2f, %.2f) publish_ground_truth_pca=%s"
            % (
                self.course.perp_x_m,
                self.course.tape_right_y_m,
                self.course.tape_left_y_m,
                self.course.goal[0],
                self.course.goal[1],
                self.publish_ground_truth_pca,
            )
        )

    def _cmd_vel_callback(self, msg: Twist) -> None:
        self.cmd_v = max(-MAX_LINEAR_SPEED_MPS,
                         min(MAX_LINEAR_SPEED_MPS, float(msg.linear.x)))
        self.cmd_w = max(-MAX_ANGULAR_SPEED_RADPS,
                         min(MAX_ANGULAR_SPEED_RADPS, float(msg.angular.z)))
        self.last_cmd_s = _stamp_to_float(self.get_clock().now().to_msg())

    def _publish_static_transforms(self) -> None:
        stamp = self.get_clock().now().to_msg()
        transforms = [
            self._make_static_transform(
                stamp, "base_link", "nav_center",
                BASE_LINK_TO_NAV_CENTER_M, 0.0, 0.0,
                *_yaw_quaternion(0.0)),
            self._make_static_transform(
                stamp, "base_link", "base_footprint",
                0.0, 0.0, -BASE_LINK_HEIGHT_ABOVE_GROUND_M,
                *_yaw_quaternion(0.0)),
            self._make_static_transform(
                stamp, "base_link", "lidar_footprint",
                LIDAR_X_FROM_BASE_LINK_M, 0.000105,
                LIDAR_Z_FROM_BASE_LINK_M,
                *_rpy_quaternion(-math.pi, 0.0, 0.0)),
        ]
        self.static_tf_pub.sendTransform(transforms)

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

    def _append_cloud_point(self,
                            out: list[tuple[float, float, float, float,
                                            float, float, float, float]],
                            world_x: float,
                            world_y: float,
                            z_above_ground: float,
                            intensity: float,
                            reflector: bool,
                            layer: int = 0) -> None:
        lidar = self._base_to_lidar(
            self._world_to_base(world_x, world_y, z_above_ground))
        rng = math.sqrt(lidar[0] ** 2 + lidar[1] ** 2 + lidar[2] ** 2)
        if rng < 0.20 or rng > 8.5:
            return
        out.append((
            float(lidar[0]),
            float(lidar[1]),
            float(lidar[2]),
            float(intensity),
            float(rng),
            float(layer),
            0.0,
            1.0 if reflector else 0.0,
        ))

    def _world_in_front_arc(self, x: float, y: float) -> bool:
        bx, by, _bz = self._world_to_base(x, y, 0.0)
        if bx < -0.2 or bx > 6.0:
            return False
        return abs(math.atan2(by, max(0.001, bx))) <= math.pi / 2.0

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

    def _cone_occludes_point(self,
                             x: float,
                             y: float,
                             z_above_ground: float) -> bool:
        return cone_occludes_point(
            self.course.cones,
            self._lidar_origin_world(),
            SENSOR_HEIGHT_M,
            (x, y, z_above_ground),
        )

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
            0.20,
            8.5,
        )

    def _build_cloud_points(self) -> list[tuple[float, float, float, float,
                                               float, float, float, float]]:
        points: list[tuple[float, float, float, float,
                           float, float, float, float]] = []
        floor_x = np.arange(-0.5, 6.1, self.floor_spacing_m)
        floor_y = np.arange(-3.2, 3.21, self.floor_spacing_m)
        for x in floor_x:
            for y in floor_y:
                if not self._world_in_front_arc(float(x), float(y)):
                    continue
                if self._cone_occludes_point(float(x), float(y), 0.0):
                    continue
                on_tape = self._tape_distance(float(x), float(y)) <= 0.06
                self._append_cloud_point(
                    points, float(x), float(y), 0.0,
                    50000.0 if on_tape else 30000.0,
                    on_tape,
                    layer=0,
                )

        for tape in self.course.tapes:
            ax, ay = tape.start
            bx, by = tape.end
            length = math.hypot(bx - ax, by - ay)
            samples = max(2, int(math.ceil(length / self.tape_spacing_m)) + 1)
            for t in np.linspace(0.0, 1.0, samples):
                x = ax + (bx - ax) * float(t)
                y = ay + (by - ay) * float(t)
                if self._world_in_front_arc(x, y):
                    if self._cone_occludes_point(x, y, 0.0):
                        continue
                    self._append_cloud_point(
                        points, x, y, 0.0, 52000.0, True, layer=0)

        for hit in self._raycast_cone_hits():
            reflector = hit.z > 0.28
            self._append_cloud_point(
                points, hit.x, hit.y, hit.z,
                50000.0 if reflector else 33000.0,
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
            self._append_cloud_point(
                points, hit.x, hit.y, hit.z, 33000.0, False,
                layer=hit.layer)
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
        if self.publish_ground_truth_pca:
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

        if now_s - self.last_cmd_s > CMD_TIMEOUT_S:
            v = 0.0
            w = 0.0
        else:
            v = self.cmd_v
            w = self.cmd_w
        self.nav_x += v * math.cos(self.heading) * dt
        self.nav_y += v * math.sin(self.heading) * dt
        self.heading = math.atan2(
            math.sin(self.heading + w * dt),
            math.cos(self.heading + w * dt),
        )
        self._publish_dynamic_transforms(now)
        self._publish_odom(now, v, w)
        self.autonomous_pub.publish(Bool(data=True))

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
        msg.info.width = 160
        msg.info.height = 160
        msg.info.origin.position.x = -3.0
        msg.info.origin.position.y = -4.0
        msg.info.origin.orientation.w = 1.0
        msg.data = [0] * (msg.info.width * msg.info.height)
        self.map_pub.publish(msg)


def main(argv: list[str] | None = None) -> int:
    rclpy.init(args=sys.argv if argv is None else argv)
    node = LidarLineCourseHarness()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
