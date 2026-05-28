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

The harness publishes dynamic `map -> odom -> base_link` TF and simulated
`/joint_states` for the drive wheels. Child frames under `base_link`, including
the lidar, nav center, caster, GPS, camera, and wheel links, come from the real
`shogi.urdf` through `robot_state_publisher`. This keeps RViz frame geometry
matched to the robot stack while still letting the harness control odometry.

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
debug topics when available, sends the configured through-gap goal, then runs
the AutoNav bag analysis suite. The default goal is beyond the perpendicular
tape on the 5 ft gap centerline so a pass proves the executed footprint clears
the gap. Use `GOAL_X=2.0 GOAL_Y=0.0` only as an intentionally bad straight-goal
safety diagnostic.

For live RViz work, `Run_LIDAR_LINE_ROS_COURSE.command` defaults to a clean
start and stops stale course, detector, PCA converter, and Nav2 processes from
prior interrupted runs. Duplicate ROS stacks publish the same odom/PCA topics
and can make the cone and costmaps appear to jump or smear. Set
`AUTONAV_SIM_CLEAN_START=0` only when intentionally running multiple stacks.
