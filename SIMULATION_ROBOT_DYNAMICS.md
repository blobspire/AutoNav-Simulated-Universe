# Simulation Robot Dynamics

This repository contains lightweight simulators for validating AutoNav robot
behavior before testing on hardware. Future agents should use this file as the
starting point for robot dimensions, motion assumptions, and expected simulator
features.

## Robot Description

The robot dimensions and navigation limits are sourced from the active
`~/code/git/AutoNav_25-26` checkout. Nav2 uses `nav_center`, not `base_link`,
as the planning/controller frame. The standalone simulator state is the
`nav_center` pose `(x, y, theta)`, forward body speed `u`, and yaw rate
`omega`. The canonical ROS harness publishes odometry from the same frame
relationship but runs the real Nav2 stack.

Core dimensions and physical values:

- Mass: `35.0 kg`
- `nav_center` offset from `base_link`: `+0.225 m`
- LiDAR offset from `base_link`: `+0.6598 m x`, `+0.20568 m z`
- LiDAR height above ground: `0.31871 m`
- Rear wheel track width used by control: `0.6858 m`
- Rear wheel radius: `0.12946 m`
- Front caster radius: `0.09 m`
- Nav2 footprint in `nav_center`: `+/-0.545 m x`, `+/-0.410 m y`
- Local costmap footprint padding: `0.03 m`
- MPPI max forward speed: `0.25 m/s`
- MPPI max yaw speed: `1.0 rad/s`
- FollowPath is forward-only: `vx_min: 0.0`
- Per-wheel force limit: `-120 N` to `200 N`
- Linear damping: `6.0 N per m/s`
- Angular damping: `2.0 N*m per rad/s`
- Physics timestep: `1 / 240 s`
- Render/update target: `30 FPS`

## Dynamics

The kinematic limits and behavior source of truth is:

- `AutoNav_25-26/isaac_ros-dev/src/slam/config/nav2_params_lidar.yaml`
- `AutoNav_25-26/isaac_ros-dev/src/slam/behavior_trees/bt_nav.xml`
- `AutoNav_25-26/isaac_ros-dev/src/bringup/description/shogi.urdf`

The standalone low-level physics integrator remains a lightweight nonholonomic
differential-drive approximation for fast visual testing. It integrates:

- Forward acceleration from total wheel force, drivetrain damping, and COM
  offset centripetal coupling.
- Angular acceleration from differential wheel torque, angular damping, and
  COM offset coupling.
- `nav_center` kinematics from forward speed and heading.

The standalone controller is not canonical for current MPPI behavior. It
mirrors broad behavior only: GoalBender-style forward targets, forward-only
FollowPath, DWB-like rollout scoring against the line costmap and path, and
breadcrumb reverse before BackUp/GradientEscape-style recovery.

The ROS lidar-line harness is the canonical path. It accepts `/cmd_vel` from
the real controller and applies a conservative drivetrain response before
publishing odometry: command latency, low-command deadbands, first-order
linear/angular response, and command timeout.

## LiDAR Line Simulation Features

The LiDAR Line Sim models retroreflective boundary tape on grey rubber floor.
The SICK driver exposes retroreflective hits through the PointCloud2
`reflector` field, which is the primary detector input for the robot. RSSI is
still simulated for visualization and fallback experiments.

Expected behavior:

- Generate layered SICK multiScan-style first returns using the robot's
  forward processing cone: 16 layers, 0.5 degree native horizontal spacing,
  front 180 degrees around robot +x, 8.5 m navigation range, and
  upside-down robot-frame vertical FOV of -35 to +7.5 degrees.
- In the canonical ROS harness, each beam chooses the nearest valid return
  among cone surface, finite-width tape, and floor. Tape points are not
  injected unless a beam actually intersects tape.
- Set `reflector=True` for retroreflective tape hits.
- Detect tape using point fields only: local xyz, range, layer, echo,
  reflector, and intensity.
- Publish accepted tape-like clusters into a line costmap.
- Complete accepted sparse reflector clusters into conservative dense local
  line segments, matching the robot `lidar_line_detector`.
- Simulate the physical lidar-line course, including the 5 ft gap and DOT cone
  PCA obstacle.
- Apply detected LiDAR line cells according to the loaded robot Nav2
  `lidar_line_layer`; on the current robot checkout, the layer's own
  `observation_persistence_ms` is `-1` for manual-clear tape-test memory,
  while the sim models the detector, local line layer, global mirror, planner,
  and controller as separate stages.
- Provide a `complex_maze` live scenario with longitudinal line walls, side
  dead ends, and too-narrow false passages that should be rejected by the
  footprint-aware planner/controller model.

The canonical lidar-line course simulation is the ROS harness in
`lidar_line_sim/Run_LIDAR_LINE_ROS_COURSE.command`. It publishes synthetic
SICK/PCA sensor topics into the real AutoNav detection, costmap, Smac Lattice,
MPPI, BT, and recovery stack. The standalone simulator remains a fast
approximation for visualization and quick debugging.

Current robot LiDAR line-layer defaults:

- `observation_persistence_ms: -1`
- `max_message_age_ms: 750`
- `observation_persistence_resolution_m: 0.10`
- `clear_lines_only_in_view: false`
- `line_clear_angle_min_rad: -0.95`
- `line_clear_angle_max_rad: 0.95`
- `line_clear_range_min_m: 0.2`
- `line_clear_range_max_m: 6.0`
- `max_persisted_points: 2000`
- `inflation_radius: 0.80`
- `inscribed_radius: 0.10`
- `cost_scaling_factor: 5.0`
- Nominal planner-flow cadence: `/lidar_line_points` at 10 Hz, local line
  layer at 15 Hz, global mirror at 3 Hz, and BT replanning at 3 Hz.
- Nominal last-detection-to-fresh-plan latency is roughly
  `1/10 s + 1/15 s + 1/3 s + 1/3 s ~= 0.83 s`,
  matching detector cadence, local line-layer update, global mirror update, and
  the behavior tree's 3 Hz `RateController`.
- RViz/global costmap publication can lag by another `1/2 s`, so the displayed
  worst case is about `1.33 s`.

## Entry Points

Snapshot/benchmark simulator:

```bash
cd lidar_line_sim/simulated_world
python lidar_line_sim.py
python lidar_line_sim.py --robot-benchmark
python lidar_line_sim.py --live-headless --scenario lidar_line_course
```

Live GUI simulator:

```bash
cd lidar_line_sim/simulated_world
python lidar_line_live_gui.py
```

Headless validation:

```bash
cd lidar_line_sim/simulated_world
python lidar_line_live_gui.py --headless
```

Canonical ROS lidar-line course:

```bash
cd lidar_line_sim
./Run_LIDAR_LINE_ROS_COURSE.command
./Run_LIDAR_LINE_ROS_COURSE_TEST.command
```

The live GUI and headless mode share the same simulation core. Any future
simulation that validates line detection, costmap persistence, planning, or
trajectory execution should reuse that core behavior instead of adding another
point-mass or teleporting robot model.
