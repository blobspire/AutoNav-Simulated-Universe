#!/usr/bin/env python3
"""Live PyQt LiDAR line simulation.

This is a real-time front end around lidar_line_sim.py. The same simulator
core drives the GUI and the headless validation mode, so agents can test the
line detector without a display and humans can watch the robot detect, plan,
and drive around remembered retroreflective tape.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from lidar_line_sim import (
    DEFAULT_MAX_RANGE_M,
    DEFAULT_RAYS,
    DEFAULT_SEED,
    FIELD_X_MAX,
    FIELD_X_MIN,
    FIELD_Y_MAX,
    FIELD_Y_MIN,
    RENDER_FPS,
    LidarLineSimulation,
    configure_sim_detector,
    configure_sim_line_layer,
    run_live_headless,
)


def _configure_from_args(sim: LidarLineSimulation,
                         args: argparse.Namespace) -> None:
    if args.robot_config:
        configure_sim_detector(sim, args.robot_config)
    if args.nav2_config:
        configure_sim_line_layer(sim, args.nav2_config)


def build_sim(args: argparse.Namespace) -> LidarLineSimulation:
    sim = LidarLineSimulation(
        rays=args.rays,
        seed=args.seed,
        max_range_m=args.max_range,
        scenario=args.scenario,
    )
    _configure_from_args(sim, args)
    sim.run_cycle(clear_memory=True)
    return sim


class LiveWorldWidget:
    def __init__(self, qt_widgets, qt_gui, qt_core,
                 sim: LidarLineSimulation):
        self.QtWidgets = qt_widgets
        self.QtGui = qt_gui
        self.QtCore = qt_core
        self.sim = sim
        self.widget = qt_widgets.QWidget()
        self.widget.setMinimumSize(980, 720)
        self.widget.paintEvent = self.paint_event
        self.widget.mousePressEvent = self.mouse_press_event
        self.show_scan = True
        self.show_candidates = True
        self.show_detected = True
        self.show_pca = True
        self.show_memory = True
        self.show_inflation = True
        self.show_path = True
        self.show_trail = True

    def update(self) -> None:
        self.widget.update()

    def _view_transform(self) -> tuple[float, float, float]:
        margin = 28.0
        world_w = FIELD_X_MAX - FIELD_X_MIN
        world_h = FIELD_Y_MAX - FIELD_Y_MIN
        avail_w = max(1.0, self.widget.width() - 2.0 * margin)
        avail_h = max(1.0, self.widget.height() - 2.0 * margin)
        scale = min(avail_w / world_w, avail_h / world_h)
        draw_w = world_w * scale
        draw_h = world_h * scale
        origin_x = (self.widget.width() - draw_w) * 0.5
        origin_y = (self.widget.height() - draw_h) * 0.5
        return origin_x, origin_y, scale

    def _map_rect(self):
        origin_x, origin_y, scale = self._view_transform()
        return self.QtCore.QRectF(
            origin_x,
            origin_y,
            (FIELD_X_MAX - FIELD_X_MIN) * scale,
            (FIELD_Y_MAX - FIELD_Y_MIN) * scale,
        )

    def _world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        origin_x, origin_y, scale = self._view_transform()
        sx = origin_x + (x - FIELD_X_MIN) * scale
        sy = origin_y + (FIELD_Y_MAX - y) * scale
        return sx, sy

    def _screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        origin_x, origin_y, scale = self._view_transform()
        x = FIELD_X_MIN + (sx - origin_x) / scale
        y = FIELD_Y_MAX - (sy - origin_y) / scale
        x = min(FIELD_X_MAX, max(FIELD_X_MIN, x))
        y = min(FIELD_Y_MAX, max(FIELD_Y_MIN, y))
        return x, y

    def _draw_polyline(self, painter, pts, color, width=2.0) -> None:
        if len(pts) < 2:
            return
        pen = self.QtGui.QPen(self.QtGui.QColor(color))
        pen.setWidthF(width)
        painter.setPen(pen)
        path = self.QtGui.QPainterPath()
        x0, y0 = self._world_to_screen(float(pts[0][0]), float(pts[0][1]))
        path.moveTo(x0, y0)
        for x, y in pts[1:]:
            sx, sy = self._world_to_screen(float(x), float(y))
            path.lineTo(sx, sy)
        painter.drawPath(path)

    def _draw_grid_cells(self, painter, grid: np.ndarray,
                         color: str, alpha: int) -> None:
        ys, xs = np.nonzero(grid)
        if ys.size == 0:
            return
        brush = self.QtGui.QColor(color)
        brush.setAlpha(alpha)
        painter.setPen(self.QtCore.Qt.NoPen)
        painter.setBrush(brush)
        res = self.sim.grid_spec.res
        for y, x in zip(ys, xs):
            wx, wy = self.sim.grid_spec.cell_to_world(int(x), int(y))
            sx0, sy0 = self._world_to_screen(wx - res * 0.5, wy - res * 0.5)
            sx1, sy1 = self._world_to_screen(wx + res * 0.5, wy + res * 0.5)
            left = min(sx0, sx1)
            top = min(sy0, sy1)
            painter.drawRect(
                self.QtCore.QRectF(left, top, abs(sx1 - sx0), abs(sy1 - sy0)))

    def _draw_tape(self, painter) -> None:
        for seg in self.sim.world.tape_segments:
            p0 = self._world_to_screen(float(seg.start[0]), float(seg.start[1]))
            p1 = self._world_to_screen(float(seg.end[0]), float(seg.end[1]))
            pen = self.QtGui.QPen(self.QtGui.QColor("#111111"))
            pen.setWidthF(9.0)
            pen.setCapStyle(self.QtCore.Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(self.QtCore.QPointF(*p0), self.QtCore.QPointF(*p1))
            pen = self.QtGui.QPen(self.QtGui.QColor("#ffffff"))
            pen.setWidthF(5.0)
            pen.setCapStyle(self.QtCore.Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(self.QtCore.QPointF(*p0), self.QtCore.QPointF(*p1))

    def _draw_cones(self, painter) -> None:
        for cone in self.sim.world.cone_obstacles:
            cx, cy = self._world_to_screen(
                float(cone.center[0]), float(cone.center[1]))
            edge_x, _edge_y = self._world_to_screen(
                float(cone.center[0] + cone.radius_m),
                float(cone.center[1]))
            radius_px = abs(edge_x - cx)
            painter.setBrush(self.QtGui.QColor(244, 123, 32, 185))
            painter.setPen(self.QtGui.QPen(self.QtGui.QColor("#5c2e0e"), 1.5))
            painter.drawEllipse(self.QtCore.QPointF(cx, cy),
                                radius_px, radius_px)

    def _draw_points(self, painter) -> None:
        scan = self.sim.last_scan
        detection = self.sim.last_detection
        if scan is None or detection is None:
            return
        if self.show_pca and self.sim.last_pca_points_world.size:
            pen = self.QtGui.QPen(self.QtGui.QColor(255, 92, 0, 205))
            pen.setWidthF(2.4)
            painter.setPen(pen)
            pts = self.sim.last_pca_points_world
            step = max(1, pts.shape[0] // 1000)
            for point in pts[::step]:
                sx, sy = self._world_to_screen(float(point[0]), float(point[1]))
                painter.drawPoint(self.QtCore.QPointF(sx, sy))
        if self.show_scan:
            pen = self.QtGui.QPen(self.QtGui.QColor(255, 222, 95, 185))
            pen.setWidthF(1.8)
            painter.setPen(pen)
            step = max(1, scan.points_world.shape[0] // 7000)
            for point in scan.points_world[::step]:
                sx, sy = self._world_to_screen(float(point[0]), float(point[1]))
                painter.drawPoint(self.QtCore.QPointF(sx, sy))
        if self.show_candidates:
            pts = scan.points_world[detection.candidate_mask]
            pen = self.QtGui.QPen(self.QtGui.QColor(255, 183, 3, 175))
            pen.setWidthF(2.2)
            painter.setPen(pen)
            step = max(1, pts.shape[0] // 1000)
            for point in pts[::step]:
                sx, sy = self._world_to_screen(float(point[0]), float(point[1]))
                painter.drawPoint(self.QtCore.QPointF(sx, sy))
        if self.show_detected and detection.line_points_world.size:
            painter.setBrush(self.QtGui.QColor(0, 109, 140, 90))
            pen = self.QtGui.QPen(self.QtGui.QColor(0, 42, 54, 230))
            pen.setWidthF(1.5)
            painter.setPen(pen)
            for point in detection.line_points_world:
                sx, sy = self._world_to_screen(float(point[0]), float(point[1]))
                painter.drawEllipse(self.QtCore.QPointF(sx, sy), 3.4, 3.4)

    def _draw_robot(self, painter) -> None:
        poly = self.sim.robot.footprint_polygon()
        qpoly = self.QtGui.QPolygonF([
            self.QtCore.QPointF(*self._world_to_screen(float(p[0]), float(p[1])))
            for p in poly
        ])
        painter.setBrush(self.QtGui.QColor("#2f80ed"))
        painter.setPen(self.QtGui.QPen(self.QtGui.QColor("#ffffff"), 1.4))
        painter.drawPolygon(qpoly)
        x0, y0 = self._world_to_screen(self.sim.robot.x, self.sim.robot.y)
        fx, fy = self.sim.robot.front_caster()
        x1, y1 = self._world_to_screen(fx, fy)
        painter.setPen(self.QtGui.QPen(self.QtGui.QColor("#003366"), 3.0))
        painter.drawLine(self.QtCore.QPointF(x0, y0),
                         self.QtCore.QPointF(x1, y1))

    def paint_event(self, _event) -> None:
        painter = self.QtGui.QPainter(self.widget)
        painter.setRenderHint(self.QtGui.QPainter.Antialiasing)
        painter.fillRect(self.widget.rect(), self.QtGui.QColor("#263b36"))
        painter.fillRect(self._map_rect(), self.QtGui.QColor("#47655d"))
        painter.setPen(self.QtGui.QPen(self.QtGui.QColor(255, 255, 255, 60), 1.0))
        painter.setBrush(self.QtCore.Qt.NoBrush)
        painter.drawRect(self._map_rect())

        if self.show_inflation:
            self._draw_grid_cells(painter, self.sim.last_inflated,
                                  "#ff595e", 48)
        if self.show_memory:
            self._draw_grid_cells(painter, self.sim.line_memory,
                                  "#168aad", 60)
        self._draw_tape(painter)
        self._draw_cones(painter)
        self._draw_points(painter)
        if self.show_trail:
            self._draw_polyline(painter, self.sim.trail, "#0b3954", 2.0)
        if self.show_path:
            self._draw_polyline(painter, self.sim.path, "#80ed99", 3.0)

        gx, gy = self._world_to_screen(float(self.sim.goal[0]),
                                       float(self.sim.goal[1]))
        painter.setBrush(self.QtGui.QColor("#ffd43b"))
        painter.setPen(self.QtGui.QPen(self.QtGui.QColor("#222222"), 1.0))
        painter.drawEllipse(self.QtCore.QPointF(gx, gy), 7.0, 7.0)
        self._draw_robot(painter)

        det = self.sim.last_detection
        text = (
            f"t={self.sim.sim_time_s:5.1f}s  "
            f"v={self.sim.robot.u:4.2f}m/s  "
            f"w={self.sim.robot.omega:4.2f}rad/s  "
            f"cells={int(np.count_nonzero(self.sim.line_memory))}  "
            f"path={len(self.sim.path)}  "
            f"mode={self.sim.recovery_mode}  "
            f"crumbs={len(self.sim.breadcrumbs)}"
        )
        if det is not None:
            text += (
                f"  refl={det.reflector_candidate_count}  "
                f"cand={det.selected_candidate_count}  "
                f"clusters={len(det.clusters)}/{det.raw_cluster_count}  "
                f"rej={det.rejected_cluster_count}  "
                f"out={det.line_points_world.shape[0]}  "
                f"det={det.elapsed_ms:.1f}ms"
            )
        painter.setPen(self.QtGui.QPen(self.QtGui.QColor("#ffffff")))
        painter.drawText(18, 24, text)
        painter.end()

    def mouse_press_event(self, event) -> None:
        x, y = self._screen_to_world(float(event.x()), float(event.y()))
        if event.button() == self.QtCore.Qt.RightButton:
            self.sim.robot.x = x
            self.sim.robot.y = y
            self.sim._reset_memory_arrays()
            self.sim.trail = [(self.sim.robot.x, self.sim.robot.y)]
            self.sim.breadcrumbs = []
            self.sim.last_breadcrumb_drop = None
            self.sim.crumbs_consumed_session = 0
            self.sim.recovery_mode = "IDLE"
            self.sim.run_cycle(clear_memory=True)
        else:
            self.sim.goal = np.array([x, y], dtype=float)
            self.sim.run_cycle(clear_memory=False)
        self.update()


class MainWindow:
    def __init__(self, qt_widgets, qt_gui, qt_core,
                 sim: LidarLineSimulation, args: argparse.Namespace):
        self.QtWidgets = qt_widgets
        self.QtCore = qt_core
        self.sim = sim
        self.args = args
        self.playing = True
        self.speed_scale = 1.0

        self.window = qt_widgets.QMainWindow()
        self.window.setWindowTitle("LiDAR Line Live Simulation")
        root = qt_widgets.QWidget()
        layout = qt_widgets.QVBoxLayout(root)
        toolbar = qt_widgets.QHBoxLayout()
        layout.addLayout(toolbar)

        self.play_button = qt_widgets.QPushButton("Pause")
        self.play_button.clicked.connect(self.toggle_play)
        toolbar.addWidget(self.play_button)

        reset_button = qt_widgets.QPushButton("Reset")
        reset_button.clicked.connect(self.reset)
        toolbar.addWidget(reset_button)

        toolbar.addWidget(qt_widgets.QLabel("Scenario"))
        self.scenario_box = qt_widgets.QComboBox()
        self.scenario_box.addItems([
            "lidar_line_course", "complex_maze", "line_maze",
            "diagonal_strip", "competition"])
        self.scenario_box.setCurrentText(args.scenario)
        self.scenario_box.currentTextChanged.connect(self.change_scenario)
        toolbar.addWidget(self.scenario_box)

        toolbar.addWidget(qt_widgets.QLabel("Speed"))
        self.speed_slider = qt_widgets.QSlider(qt_core.Qt.Horizontal)
        self.speed_slider.setMinimum(25)
        self.speed_slider.setMaximum(300)
        self.speed_slider.setValue(100)
        self.speed_slider.valueChanged.connect(self.change_speed)
        toolbar.addWidget(self.speed_slider)
        self.speed_label = qt_widgets.QLabel("1.00x")
        toolbar.addWidget(self.speed_label)

        self.canvas = LiveWorldWidget(qt_widgets, qt_gui, qt_core, sim)
        layout.addWidget(self.canvas.widget, 1)

        toggles = qt_widgets.QHBoxLayout()
        layout.addLayout(toggles)
        for label, attr in (
            ("Scan", "show_scan"),
            ("Candidates", "show_candidates"),
            ("Detected", "show_detected"),
            ("PCA", "show_pca"),
            ("Memory", "show_memory"),
            ("Inflation", "show_inflation"),
            ("Path", "show_path"),
            ("Trail", "show_trail"),
        ):
            cb = qt_widgets.QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(lambda checked, name=attr:
                               self.set_overlay(name, checked))
            toggles.addWidget(cb)
        toggles.addStretch(1)

        self.status = qt_widgets.QLabel("")
        layout.addWidget(self.status)
        self.window.setCentralWidget(root)

        self.timer = qt_core.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(int(1000 / RENDER_FPS))
        self.update_status()

    def set_overlay(self, name: str, checked: bool) -> None:
        setattr(self.canvas, name, checked)
        self.canvas.update()

    def toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_button.setText("Pause" if self.playing else "Play")

    def change_speed(self, value: int) -> None:
        self.speed_scale = value / 100.0
        self.speed_label.setText(f"{self.speed_scale:.2f}x")

    def change_scenario(self, scenario: str) -> None:
        self.sim.set_scenario(scenario)
        _configure_from_args(self.sim, self.args)
        self.sim.run_cycle(clear_memory=True)
        self.canvas.update()

    def reset(self) -> None:
        self.sim.reset()
        self.sim.run_cycle(clear_memory=True)
        self.canvas.update()

    def tick(self) -> None:
        if self.playing:
            self.sim.advance(1.0 / RENDER_FPS, speed_scale=self.speed_scale)
        self.update_status()
        self.canvas.update()

    def update_status(self) -> None:
        det = self.sim.last_detection
        clusters = len(det.clusters) if det is not None else 0
        rejected = det.rejected_cluster_count if det is not None else 0
        output_points = (
            det.line_points_world.shape[0] if det is not None else 0)
        self.status.setText(
            f"Detector: {self.sim.detector_label} | "
            f"Line layer: {self.sim.line_layer_label} | "
            f"Persistence: "
            f"{self.sim.persistence_summary()} | "
            f"Accepted clusters: {clusters} | "
            f"Rejected clusters: {rejected} | "
            f"Output points: {output_points} | "
            f"Mode: {self.sim.recovery_mode} | "
            f"Breadcrumbs: {len(self.sim.breadcrumbs)} | "
            "Left-click sets goal, right-click moves robot"
        )

    def show(self) -> None:
        self.window.resize(1220, 860)
        self.window.show()


def run_gui(args: argparse.Namespace) -> int:
    from PyQt5 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication(sys.argv[:1])
    sim = build_sim(args)
    window = MainWindow(QtWidgets, QtGui, QtCore, sim, args)
    window.show()
    return int(app.exec_())


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live PyQt LiDAR line-detection simulation")
    parser.add_argument("--rays", type=int, default=DEFAULT_RAYS)
    parser.add_argument("--max-range", type=float, default=DEFAULT_MAX_RANGE_M)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scenario", type=str, default="lidar_line_course",
                        choices=("competition", "diagonal_strip",
                                 "line_maze", "complex_maze",
                                 "lidar_line_course"))
    parser.add_argument("--robot-config", nargs="?", const="auto",
                        default="auto", metavar="PATH")
    parser.add_argument("--nav2-config", nargs="?", const="auto",
                        default="auto", metavar="PATH")
    parser.add_argument("--headless", action="store_true",
                        help="Run deterministic live validation without GUI")
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--save", type=str, default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.headless:
        return run_live_headless(args)
    return run_gui(args)


if __name__ == "__main__":
    raise SystemExit(main())
