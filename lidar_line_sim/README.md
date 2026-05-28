# LiDAR Line Sim

Reflector/RSSI line-detection simulator for the tape boundary system. The
canonical mode is now the ROS lidar-line course harness, which publishes
synthetic SICK/PCA sensor data into the real `AutoNav_25-26` detection, Nav2,
Smac Lattice, MPPI, costmap, and recovery stack. The standalone Python GUI is
still useful as a fast approximation and visual debugger.

The standalone sim models SICK multiScan-like layered ground returns,
SICK-style reflector hits from retroreflective tape, optional adaptive RSSI
fallback extraction, a line avoidance costmap, A* path planning around detected
line cells, DWB-style local trajectory scoring, PCA cone points, and live robot
motion constrained by the current `AutoNav_25-26` Nav2 settings.

The canonical ROS ray model follows the robot's forward LiDAR processing cone:
SICK multiScan165, 16 layers, 0.5 degree native horizontal spacing, 8.5 m
navigation range, front 180 degrees around robot +x, and the upside-down
robot-frame vertical FOV of -35 to +7.5 degrees. The physical sensor can
publish a full 360 degree cloud, but the robot's local planning/detection
pipeline clamps to the forward half-space for this behavior. The ROS harness
uses one first return per beam: cone surfaces can occlude floor/tape, and tape
is visible only where a downward beam actually intersects its finite-width
ground geometry.

## Run

Create the local venv once:

```bash
cd lidar_line_sim/simulated_world
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

Then launch:

```bash
python lidar_line_sim.py
python lidar_line_sim.py --benchmark
python lidar_line_sim.py --robot-benchmark
python lidar_line_sim.py --live-headless --scenario lidar_line_course
python lidar_line_sim.py --live-headless --scenario complex_maze
python lidar_line_live_gui.py
python lidar_line_live_gui.py --headless
python lidar_line_sim.py --benchmark --robot-config ~/code/git/AutoNav_25-26/isaac_ros-dev/src/autonav_detection/config/lidar_line_detector.yaml
python lidar_line_sim.py --no-gui --save /tmp/lidar_line_snapshot.png
```

The macOS and Windows launchers at this folder's root use the same
`simulated_world/.venv` layout as the other sims.

See `CANONICALITY.md` before using any simulation result as robot evidence.

## Canonical ROS Course Harness

Run this from a ROS Humble shell where `AutoNav_25-26` has been built:

```bash
cd lidar_line_sim
./Run_LIDAR_LINE_ROS_COURSE.command
```

For an automated run that records the same topics as the real robot test,
sends the selected scenario's NavigateToPose goal, and then runs the robot bag
analysis suite:

```bash
cd lidar_line_sim
./Run_LIDAR_LINE_ROS_COURSE_TEST.command
```

The default scenario is `canonical_5ft_gap`, whose target is beyond the
perpendicular tape at the configured 5 ft gap centerline, currently
`x=2.50 m, y=-0.89 m`. Run another deterministic regression scenario with
`SCENARIO`:

```bash
SCENARIO=open_10ft_lane_centering ./Run_LIDAR_LINE_ROS_COURSE_TEST.command
```

Set `GROUND_TRUTH_PCA=true` to run the same bag-analysis workflow with the
simulator's ground-truth cone scan instead of the real PCA detector path.

Run the deterministic multi-scenario suite with:

```bash
./Run_LIDAR_LINE_ROS_COURSE_SUITE.command
```

The suite currently includes:

- `canonical_5ft_gap`
- `open_10ft_lane_centering`
- `center_obstacle_dual_passage`
- `edge_obstacle_single_5ft_route`
- `narrow_decoy_gap_plus_legal_gap`
- `internal_line_no_cross`
- `minimum_turn_radius_curve`
- `canonical_5ft_gap_pose_offset`

The suite intentionally does not include a no-route/bad-goal scenario or a
dashed-line scenario. Use `STRICT_SCENARIO_GEOMETRY=1` to promote padded
clearance and scenario-specific station diagnostics to failures once baselines
are accepted. Set `GOAL_TIMEOUT`, `STARTUP_WAIT_SEC`, `PRE_GOAL_WAIT_SEC`, or
`SCENARIO_SETTLE_SEC` when a slower VM needs different timing.

On this Mac/Lima VM setup, launch live RViz through the VM-backed VNC session:

```bash
cd lidar_line_sim
./Run_LIDAR_LINE_ROS_COURSE_RVIZ.command
```

That script starts RViz inside the `autonav-ros22` VM on a virtual Linux
display, opens a local VNC tunnel at `vnc://localhost:5902`, and syncs the
course RViz config before launch. The default VNC password is `autonav`. Run
the ROS course stack in a second terminal while RViz stays open. The RViz
config shows global/local costmaps, the lidar-line costmap, `/plan`,
`/local_plan`, PCA scan/points, and `/lidar_line_detection/debug/points`. The
custom `/lidar_line_points` message is not displayed directly; use the debug
PointCloud2 topic for detected line points.

Set `AUTONAV_REPO=/path/to/AutoNav_25-26` if the checkout is not at
`~/code/git/AutoNav_25-26`. The launcher defaults to a clean start and stops
stale course, detector, PCA converter, and Nav2 processes from prior
interrupted runs. This prevents duplicate publishers from making odom, PCA
detections, and costmaps appear to jump or smear in RViz. Set
`AUTONAV_SIM_CLEAN_START=0` only when intentionally running multiple stacks.

The launch file starts:

- `ros_lidar_line_course.py`, which publishes `/cloud_all_fields_fullframe`,
  `/scan_fullframe`, `/map_padded`, `/odom`, `/local_ekf/odom`, TF, and
  `/joint_states`, then integrates motion from `/cmd_vel` through a
  conservative latency/deadband/first-order drivetrain model. The harness owns
  only `map -> odom -> base_link`; robot child frames come from the URDF.
- `robot_state_publisher` with the robot `shogi.urdf`, so RViz sees the same
  base, lidar, caster, camera, GPS, nav-center, and wheel link frames as the
  robot stack.
- The real `autonav_detection` grade detector and lidar-line detector.
- The same PCA PointCloud2-to-LaserScan converters used by the robot.
- Nav2 using the current robot `nav2_paramsv2.yaml` and BT XML.

The harness defaults to the real grade/PCA detector path. Launch with
`ground_truth_pca:=true` only when isolating Nav2/costmap behavior from PCA
perception.

Scenario geometry lives in `config/scenarios/*.yaml` using Nav2 `map` /
`nav_center` coordinates. The legacy `config/lidar_line_course.yaml` remains
for compatibility with the physical course document's lidar-start convention.
The canonical scenario's perpendicular tape is at `x=1.34 m`, the left lane
tape is at `y=+1.524 m` to represent the nominal 10 ft IGVC lane, the tape end
is at `y=-0.13 m`, the cone's left boundary is at `y=-1.654 m`, and the
desired gap centerline is near `y=-0.89 m`. The canonical automated target is
farther forward than the tape so success requires driving through the gap, not
just reaching its entrance.

Simulation failures should be diagnosed from the recorded bag, not just the
final pose. The test runner records `/lidar_line_points`,
`/scan_pca_filtered_points`, line/local/global costmaps, `/plan`,
`/local_plan`, MPPI trajectory debug topics when available, `/evaluation`,
Nav2 action statuses, odom/TF, and `/cmd_vel`, then runs the same lidar-line
analysis suite used after physical robot tests.

## Live Simulation

`lidar_line_live_gui.py` is the preferred tool for watching the robot in real
time. It loads the robot detector config and Nav2 `lidar_line_layer` config by
default, starts in the `complex_maze` scenario, then shows raw LiDAR returns,
reflector candidates, accepted line points, global mirrored line cells,
inflated costmap cells, the planned path, the driven trail, breadcrumbs,
recovery mode, and the robot footprint.

Use the GUI controls to play/pause, reset, change scenario, change simulation
speed, and toggle overlays. Left-click sets the goal. Right-click moves the
robot and clears remembered line cells.

For automated validation, run:

```bash
python lidar_line_live_gui.py --headless
```

The headless check passes only when the robot detects retroreflective tape,
uses the loaded robot Nav2 line-layer persistence policy, proves the straight
path is blocked in the maze, moves laterally under the force-based dynamics,
avoids the too-narrow false passages, and does not cut materially through tape.

The GUI remains a standalone approximation. It does not launch a ROS 2 graph
or Nav2 bringup. It loads the robot detector/Nav2 YAML values and mirrors the
Nav2 behavioral pieces that matter for quick visualization: the loaded
line-layer costmap policy, A*-style global planning, DWB-style local trajectory
scoring, forward-only FollowPath, breadcrumb reverse recovery, and force-based
controller execution. That DWB-like controller is intentionally not a model of
the current MPPI stack. Use the ROS harness for canonical pass/fail decisions.

## Snapshot Interaction

- Left-click the world panel to move the goal.
- Right-click or shift-click the world panel to move the robot and clear the
  line-memory costmap.
- Use `Rescan`, `Step`, and `Reset`, or keyboard shortcuts `r`, `g`, and `x`.

## Detector Contract

The detector consumes only simulated point cloud fields:

- local xyz
- range
- layer
- echo
- reflector
- RSSI intensity

The benchmark uses ground-truth tape only after detection to score recall,
precision, timing, and whether the planned path stays clear of tape.

## Robot Config Mode

`--robot-config` loads the real ROS 2 `lidar_line_detector.yaml` and applies
the candidate mode, reflector gate, fallback adaptive RSSI thresholds, cluster
filters, voxel output size, and max output point count in the sim.
`--nav2-config` loads the real ROS 2 `nav2_paramsv2.yaml` and applies the
`lidar_line_layer` persistence values. `--robot-benchmark` is shorthand for
`--benchmark --robot-config auto --nav2-config auto`, where `auto` searches
the common local AutoNav checkout paths under `~/code/git` and prefers
`AutoNav_25-26`.

The robot node gates ground points in `base_link`, while this sim generates
points in the LiDAR sensor frame. Robot config mode applies the equivalent
z-offset before the ground gate so the sim exercises the same configured
`ground_z_m` and `ground_z_tolerance_m` values.

## Lab Reflector Model

The built-in return profile assumes grey rubber floor RSSI near the low 30s
at short range, attenuating with range and layer angle. Retroreflective tape
sets `reflector=True` and adds a high RSSI return for visualization and
fallback experiments. The default detector uses `candidate_mode: reflector`;
the adaptive RSSI values remain available for non-reflective tape testing.

The line memory mirrors the robot's `lidar_line_layer` behavior. On the current
`AutoNav_25-26` checkout, the lidar line layer uses manual-clear persistence
for the tape test (`observation_persistence_ms: -1`),
updates locally at 15 Hz, mirrors to global at 3 Hz, and the behavior tree
replans at 3 Hz. The simulator also models the C++ detector's completed
segment output so sparse reflector hits become dense `/lidar_line_points`
without bridging separated tape/cone obstacles.

If measured SICK RSSI values differ in the lab, tune the constants near the
top of `lidar_line_sim.py` and rerun:

```bash
python lidar_line_sim.py --robot-benchmark
```
