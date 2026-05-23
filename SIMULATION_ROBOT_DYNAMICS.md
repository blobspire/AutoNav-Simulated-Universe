# Simulation Robot Dynamics

This repository contains lightweight simulators for validating AutoNav robot
behavior before testing on hardware. Future agents should use this file as the
starting point for robot dimensions, motion assumptions, and expected simulator
features.

## Robot Description

The robot is modeled from the Behavior Tree Sim as a rear-axle differential
drive chassis with two driven rear knife-edge wheels and a passive front
caster. The simulated state is the rear-axle midpoint `(x, y)`, heading
`theta`, forward body speed `u`, and yaw rate `omega`.

Core dimensions and physical values:

- Mass: `35.0 kg`
- Center of mass offset from rear axle: `0.25 m`
- Wheelbase, rear axle to front caster: `0.39 m`
- Rear wheel track width: `0.54 m`
- Rear wheel radius: `0.20 m`
- Front caster radius: `0.09 m`
- Footprint half width: `0.21 m`
- Footprint rear extension: `0.10 m`
- Footprint forward extension: `0.44 m`
- Per-wheel force limit: `-120 N` to `200 N`
- Linear damping: `6.0 N per m/s`
- Angular damping: `2.0 N*m per rad/s`
- Physics timestep: `1 / 240 s`
- Render/update target: `30 FPS`

## Dynamics

The dynamics source of truth is:

`BEHAVIOR TREE Sim/simulated_world/bt_sim_gui.py`

The model is a Chaplygin-sleigh approximation. The rear axle is constrained to
move along the robot body x-axis, while yaw comes from differential rear wheel
force. The simulator integrates:

- Forward acceleration from total wheel force, drivetrain damping, and COM
  offset centripetal coupling.
- Angular acceleration from differential wheel torque, angular damping, and
  COM offset coupling.
- Rear-axle kinematics from forward speed and heading.

The controller should use the same force allocation style as the Behavior Tree
Sim: pure pursuit produces desired linear/angular motion, DWB-style rollout
scores nearby candidate trajectories against obstacle/costmap distance, and
the selected command is converted into left/right wheel forces.

## LiDAR Line Simulation Features

The LiDAR Line Sim models retroreflective boundary tape on grey rubber floor.
The SICK driver exposes retroreflective hits through the PointCloud2
`reflector` field, which is the primary detector input for the robot. RSSI is
still simulated for visualization and fallback experiments.

Expected behavior:

- Generate layered SICK multiScan-style ground returns.
- Set `reflector=True` for retroreflective tape hits.
- Detect tape using point fields only: local xyz, range, layer, echo,
  reflector, and intensity.
- Publish accepted tape-like clusters into a line costmap.
- Persist detected LiDAR line cells according to the robot Nav2
  `lidar_line_layer`.
- Plan and drive around remembered line obstacles even when the LiDAR briefly
  loses direct visibility of tape on the ground.

Current robot LiDAR line-layer defaults:

- `observation_persistence_ms: 10000`
- `observation_persistence_resolution_m: 0.10`
- `clear_lines_only_in_view: true`
- `line_clear_angle_min_rad: -0.95`
- `line_clear_angle_max_rad: 0.95`
- `line_clear_range_min_m: 0.2`
- `line_clear_range_max_m: 6.0`
- `max_persisted_points: 12000`

## Entry Points

Snapshot/benchmark simulator:

```bash
cd "LiDAR Line Sim/simulated_world"
python lidar_line_sim.py
python lidar_line_sim.py --robot-benchmark --rays 11520
```

Live GUI simulator:

```bash
cd "LiDAR Line Sim/simulated_world"
python lidar_line_live_gui.py
```

Headless validation:

```bash
cd "LiDAR Line Sim/simulated_world"
python lidar_line_live_gui.py --headless --duration 12
```

The live GUI and headless mode share the same simulation core. Any future
simulation that validates line detection, costmap persistence, planning, or
trajectory execution should reuse that core behavior instead of adding another
point-mass or teleporting robot model.
