# LiDAR Line Sim

Standalone reflector/RSSI line-detection simulator for the tape boundary
system. This is separate from the PCA terrain/grade LiDAR sim. The default
return model matches the current robot setup: retroreflective tape strips on
grey rubber flooring.

The sim models a flat competition-style field, SICK multiScan-like layered
ground returns, SICK-style reflector hits from retroreflective tape, optional
adaptive RSSI fallback extraction, a persistent line avoidance costmap, A*
path planning around detected line cells, and live robot motion using the
Behavior Tree Sim force-based dynamics.

## Run

Create the local venv once:

```bash
cd "LiDAR Line Sim/simulated_world"
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

Then launch:

```bash
python lidar_line_sim.py
python lidar_line_sim.py --benchmark --rays 11520
python lidar_line_sim.py --robot-benchmark --rays 11520
python lidar_line_sim.py --live-headless --scenario diagonal_strip
python lidar_line_live_gui.py
python lidar_line_live_gui.py --headless --duration 12
python lidar_line_sim.py --benchmark --robot-config ~/code/git/AutoNavB/isaac_ros-dev/src/autonav_detection/config/lidar_line_detector.yaml
python lidar_line_sim.py --no-gui --save /tmp/lidar_line_snapshot.png
```

The macOS and Windows launchers at this folder's root use the same
`simulated_world/.venv` layout as the other sims.

## Live Simulation

`lidar_line_live_gui.py` is the preferred tool for watching the robot in real
time. It loads the robot detector config and Nav2 `lidar_line_layer` config by
default, then shows raw LiDAR returns, reflector candidates, accepted line
points, remembered line cells, inflated costmap cells, the planned path, the
driven trail, and the robot footprint.

Use the GUI controls to play/pause, reset, change scenario, change simulation
speed, and toggle overlays. Left-click sets the goal. Right-click moves the
robot and clears remembered line cells.

For automated validation, run:

```bash
python lidar_line_live_gui.py --headless --duration 12
```

The headless check passes only when the robot detects retroreflective tape,
keeps remembered line cells after direct detections disappear, plans around
the remembered costmap, moves under the force-based dynamics, and avoids
driving through the tape.

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
the common local AutoNav checkout paths under `~/code/git`.

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

The line memory mirrors the robot's `lidar_line_layer` behavior: detections
refresh timestamped cells, stale cells persist for the configured observation
persistence window, and view-gated clearing only forgets stale cells that are
inside the configured LiDAR clearing cone/range. The current robot default is
`observation_persistence_ms: 10000`.

If measured SICK RSSI values differ in the lab, tune the constants near the
top of `lidar_line_sim.py` and rerun:

```bash
python lidar_line_sim.py --robot-benchmark --rays 11520
```
