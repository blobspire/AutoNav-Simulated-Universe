# LiDAR Line Sim

Standalone RSSI line-detection simulator for the white-tape boundary system.
This is separate from the PCA terrain/grade LiDAR sim. The default RSSI model
matches the current lab setup: white duct tape strips on grey rubber flooring.

The sim models a flat competition-style field, SICK multiScan-like layered
ground returns, moderate RSSI lift from white duct tape, adaptive line
extraction, a line avoidance costmap, and A* path planning around detected
line cells.

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
python lidar_line_sim.py --benchmark --robot-config ~/code/git/AutoNavB/isaac_ros-dev/src/autonav_detection/config/lidar_line_detector.yaml
python lidar_line_sim.py --no-gui --save /tmp/lidar_line_snapshot.png
```

The macOS and Windows launchers at this folder's root use the same
`simulated_world/.venv` layout as the other sims.

## Interaction

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
those detector gates, adaptive RSSI thresholds, cluster filters, voxel output
size, and max output point count in the sim. `--robot-benchmark` is shorthand
for `--benchmark --robot-config auto`, where `auto` searches the common local
AutoNav checkout paths under `~/code/git`.

The robot node gates ground points in `base_link`, while this sim generates
points in the LiDAR sensor frame. Robot config mode applies the equivalent
z-offset before the ground gate so the sim exercises the same configured
`ground_z_m` and `ground_z_tolerance_m` values.

## Lab RSSI Model

The built-in RSSI profile assumes grey rubber floor returns near the low 30s
at short range, attenuating with range and layer angle. White duct tape adds a
moderate RSSI boost into roughly the 70-90 range for the usual lab distances.
The sim intentionally does not mark duct tape as a reflector by default; the
robot config must detect it from intensity contrast.

If measured SICK RSSI values differ in the lab, tune the constants near the
top of `lidar_line_sim.py` and rerun:

```bash
python lidar_line_sim.py --robot-benchmark --rays 11520
```
