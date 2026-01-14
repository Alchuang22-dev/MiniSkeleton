"""3D viewport and interaction layer built on PyVista + Qt."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget
import pyvista as pv
from pyvistaqt import QtInteractor
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


class RigViewport(QWidget):
    """Handles rendering, picking, gizmo, and mouse dragging for joints."""

    def __init__(self, controller, on_status, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.on_status = on_status

        self.plotter = QtInteractor(self)
        self.plotter.set_background("white")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plotter.interactor)

        self.plotter.interactor.installEventFilter(self)
        self.picker = vtk.vtkPropPicker()

        # Cache of actors
        self.mesh_actor = None
        self.bone_actors = []
        self.joint_actors = []
        self.joint_sphere_actors = {}
        self.gizmo_actors = []
        self.axis_arrows = {}
        self.label_actor = None
        self._mesh_points = None
        self._mesh_points_np = None
        self._orientation_widget = None

        # Interaction state
        self.dragging_axis = None
        self.is_dragging = False
        self.is_dragging_model = False
        self.last_mouse_pos = None
        # Deferred mesh update during drag
        self._camera_set = False
        self._init_orientation_widget()

    # ---------------------------------------------------------------- Events

    def eventFilter(self, obj, event):
        if obj == self.plotter.interactor:
            if event.type() == QEvent.MouseButtonPress:
                handled = self.handle_mouse_press(event)
                self._notify(
                    f"MousePress button={event.button()} handled={handled} "
                    f"dragging_model={self.is_dragging_model}"
                )
                return bool(handled)
            if event.type() == QEvent.MouseMove:
                self.handle_mouse_move(event)
                return self.is_dragging or self.is_dragging_model
            if event.type() == QEvent.MouseButtonRelease:
                self.handle_mouse_release(event)
                return False
        return super().eventFilter(obj, event)

    def handle_mouse_press(self, event) -> bool:
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return False

        mouse_x, mouse_y, mouse_x_scaled, mouse_y_vtk = self._map_mouse(event)
        dpr = self.plotter.interactor.devicePixelRatio()
        self.picker.Pick(mouse_x_scaled, mouse_y_vtk, 0, self.plotter.renderer)
        picked_actor = self.picker.GetActor()
        if picked_actor is None:
            picked = "None"
        elif picked_actor == self.mesh_actor:
            picked = "mesh"
        elif picked_actor in self.axis_arrows:
            picked = "gizmo"
        elif picked_actor in self.joint_sphere_actors:
            picked = "joint"
        else:
            picked = "other"
        self._notify(f"Pick actor={picked} button={event.button()} dpr={dpr:.2f}")

        if event.button() == Qt.RightButton:
            if picked_actor is not None and (
                picked_actor == self.mesh_actor or picked_actor in self.joint_sphere_actors
            ):
                self.is_dragging_model = True
                self.last_mouse_pos = (mouse_x_scaled, mouse_y_vtk)
                self.plotter.disable()
                # self._notify("Dragging model position (RMB)")
                return True
            return False

        # 1) drag gizmo axis
        if picked_actor in self.axis_arrows:
            axis_name, axis_vector = self.axis_arrows[picked_actor]
            self.is_dragging = True
            self.dragging_axis = (axis_name, axis_vector)
            self.last_mouse_pos = (mouse_x, mouse_y)
            self.plotter.disable()
            # self._notify(f"Start dragging {axis_name.upper()} axis")
            return True

        # 2) pick joint by screen-space proximity (supports overlap cycling)
        radius_px = 12.0 * dpr
        candidates = self._pick_joint_candidates(mouse_x_scaled, mouse_y_vtk, radius_px)
        if candidates:
            if (
                self.controller.selected_joint in candidates
                and not (event.modifiers() & Qt.ShiftModifier)
            ):
                self.is_dragging = True
                self.last_mouse_pos = (mouse_x, mouse_y)
                self.plotter.disable()
                self._notify(f"Dragging joint [{self.controller.selected_joint}]")
                return True

            if (
                event.modifiers() & Qt.ShiftModifier
                and self.controller.selected_joint in candidates
                and len(candidates) > 1
            ):
                current_idx = candidates.index(self.controller.selected_joint)
                joint_idx = candidates[(current_idx + 1) % len(candidates)]
            else:
                joint_idx = candidates[0]

            self.controller.selected_joint = joint_idx
            self.update_gizmo_only()
            joint_name = self.controller.skeleton.joints[joint_idx].name
            if len(candidates) > 1:
                self._notify(
                    f"Selected joint [{joint_idx}] {joint_name} "
                    f"(Shift-click to cycle {len(candidates)} overlaps)."
                )
            else:
                self._notify(f"Selected joint [{joint_idx}] {joint_name}")
            return True

        # 3) click other actor -> deselect
        self.controller.selected_joint = None
        self.update_gizmo_only()
        self._notify("Click red sphere to select a joint.")
        return False

    def handle_mouse_move(self, event):
        if self.is_dragging_model and event.buttons() & Qt.RightButton:
            self._handle_model_drag(event)
            return
        if not (self.is_dragging and event.buttons() & Qt.LeftButton):
            return
        if self.controller.selected_joint is None:
            return

        x, y = event.x(), event.y()
        if self.last_mouse_pos is None:
            self.last_mouse_pos = (x, y)
            return

        dx = x - self.last_mouse_pos[0]
        dy = y - self.last_mouse_pos[1]
        if abs(dx) < 1 and abs(dy) < 1:
            return

        camera = self.plotter.camera
        camera_pos = np.array(camera.GetPosition())

        G_current = self.controller.compute_current_global_mats()
        joint_pos = G_current[self.controller.selected_joint, :3, 3] + self._get_model_offset()

        distance = np.linalg.norm(camera_pos - joint_pos)
        scale = distance * 0.001

        view_up = np.array(camera.GetViewUp())
        view_dir = camera_pos - joint_pos
        view_dir = view_dir / np.linalg.norm(view_dir)
        right = np.cross(view_up, view_dir)
        right = right / np.linalg.norm(right)
        up = np.cross(view_dir, right)
        up = up / np.linalg.norm(up)

        if self.dragging_axis is not None:
            _, axis_vector = self.dragging_axis
            screen_delta = right * dx * scale + up * dy * scale
            delta = np.dot(screen_delta, axis_vector) * axis_vector
        else:
            delta = right * dx * scale + up * dy * scale

        self.last_mouse_pos = (x, y)
        self.controller.queue_joint_delta(self.controller.selected_joint, delta)

    def handle_mouse_release(self, event):
        if event.button() == Qt.RightButton and self.is_dragging_model:
            self.is_dragging_model = False
            self.last_mouse_pos = None
            self.plotter.enable()
            self.controller.request_deform_update()
            self._notify("Model move finished")
            return
        if event.button() != Qt.LeftButton or not self.is_dragging:
            return
        self.is_dragging = False
        self.dragging_axis = None
        self.last_mouse_pos = None
        self.plotter.enable()

        self.controller.request_deform_update()

        if self.controller.selected_joint is not None:
            joint_name = self.controller.skeleton.joints[self.controller.selected_joint].name
            self._notify(f"Joint [{self.controller.selected_joint}] {joint_name} move finished")

    # -------------------------------------------------------------- Rendering

    def render_scene_full(self):
        if self.controller.mesh is None:
            return

        self.plotter.clear()
        self.joint_sphere_actors = {}
        self.axis_arrows = {}
        self.bone_actors = []
        self.joint_actors = []

        G_current = self.controller.compute_current_global_mats()
        offset = self._get_model_offset()
        current_joint_positions = G_current[:, :3, 3] + offset
        deformed_vertices = self.controller.compute_deformed_vertices() + offset

        mesh_size = np.linalg.norm(deformed_vertices.max(axis=0) - deformed_vertices.min(axis=0))
        sphere_radius = mesh_size * 0.015

        faces = self.controller.mesh.faces.astype(np.int64)
        faces_with_count = np.hstack([np.full((len(faces), 1), 3), faces])
        mesh_pv = pv.PolyData(deformed_vertices, faces_with_count)
        self.mesh_actor = self.plotter.add_mesh(
            mesh_pv,
            color="lightblue",
            opacity=0.6,
            show_edges=True,
            edge_color="navy",
            line_width=0.3,
            smooth_shading=True,
            pickable=True,
        )
        vtk_points = self.mesh_actor.GetMapper().GetInput().GetPoints()
        vtk_array = vtk_points.GetData()
        self._mesh_points = vtk_points
        self._mesh_points_np = vtk_to_numpy(vtk_array)

        for jp, jc in self.controller.bones:
            p1 = current_joint_positions[jp]
            p2 = current_joint_positions[jc]
            line = pv.Line(p1, p2)
            actor = self.plotter.add_mesh(
                line,
                color="darkred",
                line_width=8,
                opacity=0.8,
                pickable=False,
            )
            self.bone_actors.append((actor, jp, jc))

        for i, pos in enumerate(current_joint_positions):
            sphere = pv.Sphere(
                radius=sphere_radius,
                center=pos.tolist(),
                theta_resolution=16,
                phi_resolution=16,
            )
            color = "yellow" if i == self.controller.selected_joint else "red"
            actor = self.plotter.add_mesh(
                sphere,
                color=color,
                opacity=0.9,
                pickable=True,
                lighting=True,
            )
            self.joint_sphere_actors[actor] = i
            self.joint_actors.append((actor, i, sphere_radius))

        self.update_gizmo_only()

        if not self._camera_set:
            self.plotter.reset_camera()
            self.plotter.camera.elevation = 15
            self.plotter.camera.azimuth = -60
            self.plotter.camera.zoom(1.4)
            self._camera_set = True

        self.plotter.update()

    def update_deformed_mesh_only(self, vertices: np.ndarray | None = None):
        if self.mesh_actor is None:
            return

        base_vertices = vertices if vertices is not None else self.controller.compute_deformed_vertices()
        offset = self._get_model_offset()
        deformed_vertices = base_vertices + offset
        if (
            self._mesh_points_np is None
            or self._mesh_points_np.shape != deformed_vertices.shape
            or self._mesh_points is None
        ):
            vtk_points = self.mesh_actor.GetMapper().GetInput().GetPoints()
            vtk_array = numpy_to_vtk(deformed_vertices, deep=True)
            vtk_points.SetData(vtk_array)
            vtk_points.Modified()
            self._mesh_points = vtk_points
            self._mesh_points_np = vtk_to_numpy(vtk_array)
        else:
            self._mesh_points_np[:] = deformed_vertices
            self._mesh_points.Modified()

        G_current = self.controller.compute_current_global_mats()
        current_joint_positions = G_current[:, :3, 3] + offset

        for actor, jp, jc in self.bone_actors:
            p1 = current_joint_positions[jp]
            p2 = current_joint_positions[jc]
            line = pv.Line(p1, p2)
            actor.GetMapper().SetInputData(line)

        for actor, joint_idx, radius in self.joint_actors:
            pos = current_joint_positions[joint_idx]
            sphere = pv.Sphere(
                radius=radius,
                center=pos.tolist(),
                theta_resolution=16,
                phi_resolution=16,
            )
            actor.GetMapper().SetInputData(sphere)

        self.update_gizmo_only()
        self.plotter.update()

    def update_gizmo_only(self):
        for actor in self.gizmo_actors:
            self.plotter.remove_actor(actor)
        self.gizmo_actors = []
        self.axis_arrows = {}

        if self.label_actor is not None:
            self.plotter.remove_actor(self.label_actor)
            self.label_actor = None

        if self.controller.selected_joint is None:
            self.plotter.update()
            return

        G_current = self.controller.compute_current_global_mats()
        current_joint_positions = G_current[:, :3, 3] + self._get_model_offset()
        pos = current_joint_positions[self.controller.selected_joint]

        mesh_size = np.linalg.norm(
            self.controller.mesh.vertices.max(axis=0) - self.controller.mesh.vertices.min(axis=0)
        )
        arrow_length = mesh_size * 0.1

        axes = [
            ("x", np.array([1.0, 0.0, 0.0]), "red"),
            ("y", np.array([0.0, 1.0, 0.0]), "green"),
            ("z", np.array([0.0, 0.0, 1.0]), "blue"),
        ]

        for axis_name, direction, color in axes:
            arrow = pv.Arrow(
                start=pos.tolist(),
                direction=direction.tolist(),
                tip_length=0.25,
                tip_radius=0.1,
                shaft_radius=0.03,
                scale=float(arrow_length),
            )
            actor = self.plotter.add_mesh(
                arrow,
                color=color,
                opacity=0.8,
                pickable=True,
                lighting=True,
            )
            self.axis_arrows[actor] = (axis_name, direction)
            self.gizmo_actors.append(actor)

        joint_name = self.controller.skeleton.joints[self.controller.selected_joint].name
        sphere_radius = mesh_size * 0.015
        label_pos = pos + np.array([0, sphere_radius * 3, 0])

        self.label_actor = self.plotter.add_point_labels(
            [label_pos],
            [f"[{self.controller.selected_joint}] {joint_name}"],
            font_size=14,
            bold=True,
            text_color="black",
            point_color="yellow",
            point_size=20,
            shape_opacity=0.8,
        )

        self.plotter.update()

    # ---------------------------------------------------------------- Helpers

    def _notify(self, msg: str):
        if self.on_status:
            self.on_status(msg)

    def _map_mouse(self, event):
        mouse_x, mouse_y = event.x(), event.y()
        window_size = self.plotter.window_size
        dpr = self.plotter.interactor.devicePixelRatio()

        mouse_x_scaled = mouse_x * dpr
        mouse_y_scaled = mouse_y * dpr
        window_height = window_size[1]
        mouse_y_vtk = window_height - mouse_y_scaled
        return mouse_x, mouse_y, mouse_x_scaled, mouse_y_vtk

    def _handle_model_drag(self, event):
        _, _, mouse_x_scaled, mouse_y_vtk = self._map_mouse(event)
        if self.last_mouse_pos is None:
            self.last_mouse_pos = (mouse_x_scaled, mouse_y_vtk)
            return
        prev_x, prev_y = self.last_mouse_pos
        dx = mouse_x_scaled - prev_x
        dy = mouse_y_vtk - prev_y
        if abs(dx) < 1 and abs(dy) < 1:
            return

        target = self._get_model_center()
        renderer = self.plotter.renderer
        renderer.SetWorldPoint(float(target[0]), float(target[1]), float(target[2]), 1.0)
        renderer.WorldToDisplay()
        _, _, depth = renderer.GetDisplayPoint()

        def display_to_world(x, y, z):
            renderer.SetDisplayPoint(float(x), float(y), float(z))
            renderer.DisplayToWorld()
            wx, wy, wz, w = renderer.GetWorldPoint()
            if w == 0:
                return np.array([wx, wy, wz], dtype=np.float32)
            return np.array([wx / w, wy / w, wz / w], dtype=np.float32)

        world_prev = display_to_world(prev_x, prev_y, depth)
        world_curr = display_to_world(mouse_x_scaled, mouse_y_vtk, depth)
        delta = world_curr - world_prev
        self.last_mouse_pos = (mouse_x_scaled, mouse_y_vtk)
        offset = self._get_model_offset() + delta
        setter = getattr(self.controller, "set_model_offset", None)
        if callable(setter):
            setter(float(offset[0]), float(offset[1]), float(offset[2]))

    def _get_model_center(self) -> np.ndarray:
        if self.controller.mesh is None:
            return np.zeros(3, dtype=np.float32)
        verts = np.asarray(self.controller.mesh.vertices, dtype=np.float32)
        center = (verts.min(axis=0) + verts.max(axis=0)) * 0.5
        return center + self._get_model_offset()

    def _get_model_offset(self) -> np.ndarray:
        getter = getattr(self.controller, "get_model_offset", None)
        if callable(getter):
            return np.asarray(getter(), dtype=np.float32)
        offset = getattr(self.controller, "model_offset", None)
        if offset is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(offset, dtype=np.float32)

    def _init_orientation_widget(self) -> None:
        try:
            if hasattr(vtk, "vtkCameraOrientationWidget"):
                widget = vtk.vtkCameraOrientationWidget()
                widget.SetInteractor(self.plotter.interactor)
                widget.SetParentRenderer(self.plotter.renderer)
                if hasattr(widget, "SetViewport"):
                    widget.SetViewport(0.82, 0.82, 0.98, 0.98)
                widget.On()
                self._orientation_widget = widget
                return
        except Exception:
            self._orientation_widget = None

        try:
            axes = vtk.vtkAxesActor()
            widget = vtk.vtkOrientationMarkerWidget()
            widget.SetOrientationMarker(axes)
            widget.SetInteractor(self.plotter.interactor)
            widget.SetViewport(0.82, 0.82, 0.98, 0.98)
            widget.SetEnabled(1)
            widget.InteractiveOff()
            self._orientation_widget = widget
        except Exception:
            self._orientation_widget = None

    def _pick_joint_candidates(
        self,
        mouse_x_vtk: float,
        mouse_y_vtk: float,
        radius_px: float,
    ) -> list[int]:
        if self.controller.skeleton is None:
            return []
        renderer = self.plotter.renderer
        G_current = self.controller.compute_current_global_mats()
        joint_positions = G_current[:, :3, 3] + self._get_model_offset()
        matches: list[tuple[int, float]] = []
        for idx, pos in enumerate(joint_positions):
            renderer.SetWorldPoint(float(pos[0]), float(pos[1]), float(pos[2]), 1.0)
            renderer.WorldToDisplay()
            sx, sy, _ = renderer.GetDisplayPoint()
            dx = sx - mouse_x_vtk
            dy = sy - mouse_y_vtk
            dist = (dx * dx + dy * dy) ** 0.5
            if dist <= radius_px:
                matches.append((idx, dist))
        matches.sort(key=lambda it: it[1])
        return [idx for idx, _ in matches]
