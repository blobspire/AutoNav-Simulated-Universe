# LiDAR-Line Simulation Canonicality

This document defines what can and cannot be trusted when using the
lidar-line simulator for robot planning work.

## Authority Levels

- Canonical pass/fail: `Run_LIDAR_LINE_ROS_COURSE_TEST.command`.
- Visualization and fast debugging: `lidar_line_sim.py` and
  `lidar_line_live_gui.py`.
- Real-world source of truth: physical robot bags from the measured course.

The canonical target is safety conservative. A simulated pass should imply
high confidence on the robot; a simulated fail may be stricter than reality.

## Canonical ROS Harness

The ROS harness starts the real AutoNav detection and Nav2 stack, including
the current Smac Lattice global planner, MPPI local controller, costmaps,
behavior tree, and recovery behaviors. The harness owns only the synthetic
world and the robot motion response.

The harness now generates LiDAR points as first returns per SICK-style beam.
Each beam chooses the nearest valid return among:

- finite cone-cylinder surface hits,
- floor intersections,
- finite-width retroreflective tape where the beam hits the floor.

This replaces the older optimistic model that injected floor-grid and tape
centerline points independent of beam geometry.

The harness also applies a conservative drivetrain response between `/cmd_vel`
and odometry:

- command latency,
- first-order linear and angular response,
- low-command deadbands,
- command timeout.

These parameters are intentionally conservative until calibrated from robot
bags.

## Known Non-Canonical Pieces

- The cone is still modeled as a finite cylinder, not a measured DOT cone mesh.
- Reflective cone tape is approximated by hit height.
- Beam divergence, multi-echo behavior, rolling scan timing, real RSSI
  distributions, self-reflections, sensor mount tolerance, wheel slip, battery
  voltage sag, and encoder quantization are not yet calibrated.
- The harness publishes a flat `/map_padded`; it does not exercise
  `slam_toolbox` plus `map_padder` exactly as the robot does.
- The standalone Python simulator uses A* and DWB-like logic for speed. It is
  not authoritative for current MPPI behavior.

## Calibration Gates

Before treating a simulator result as competition evidence, compare the
canonical sim bag against a recent physical-course robot bag:

- first perpendicular-tape detection time,
- `/lidar_line_points` density and extent,
- `/lidar_line_costmap` hard/soft geometry,
- PCA cone point location and scan projection,
- global plan clearance through the 5 ft gap,
- local controller command timing and stop/turn behavior,
- final odom path and measured-course footprint clearance.

Accept the sim as canonical only when it is equal or more conservative than
the robot on these metrics.

## Running the Canonical Test

```bash
cd lidar_line_sim
./Run_LIDAR_LINE_ROS_COURSE_TEST.command
```

The runner records the same core topics as the robot test plus MPPI trajectory
debug topics when available, then runs the AutoNav bag analysis suite.
