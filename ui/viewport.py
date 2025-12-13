"""3D viewport and interaction layer built on PyVista + Qt."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QEvent, QTimer, Qt
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import pyvista as pv
from pyvistaqt import QtInteractor
import vtk
from vtk.util.numpy_support import numpy_to_vtk


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

        # Interaction state
        self.dragging_axis = None
        self.is_dragging = False
        self.last_mouse_pos = None

        # Deferred mesh update during drag
        self.pending_update = False
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(16)
        self.update_timer.timeout.connect(self._deferred_update)

        self._camera_set = False

    # ---------------------------------------------------------------- Events

    def eventFilter(self, obj, event):
        if obj == self.plotter.interactor:
            if event.type() == QEvent.MouseButtonPress:
                self.handle_mouse_press(event)
                return False
            if event.type() == QEvent.MouseMove:
                self.handle_mouse_move(event)
                return self.is_dragging
            if event.type() == QEvent.MouseButtonRelease:
                self.handle_mouse_release(event)
                return False
        return super().eventFilter(obj, event)

    def handle_mouse_press(self, event):
        if event.button() != Qt.LeftButton:
            return

        mouse_x, mouse_y = event.x(), event.y()
        window_size = self.plotter.window_size
        dpr = self.plotter.interactor.devicePixelRatio()

        mouse_x_scaled = mouse_x * dpr
        mouse_y_scaled = mouse_y * dpr
        window_height = window_size[1]

        self.picker.Pick(mouse_x_scaled, window_height - mouse_y_scaled, 0, self.plotter.renderer)
        picked_actor = self.picker.GetActor()

        if picked_actor is None:
            self.controller.selected_joint = None
            self.update_gizmo_only()
            self._notify("Click red sphere to select a joint.")
            return

        # 1) drag gizmo axis
        if picked_actor in self.axis_arrows:
            axis_name, axis_vector = self.axis_arrows[picked_actor]
            self.is_dragging = True
            self.dragging_axis = (axis_name, axis_vector)
            self.last_mouse_pos = (mouse_x, mouse_y)
            self.plotter.disable()
            self._notify(f"Start dragging {axis_name.upper()} axis")
            return

        # 2) pick joint sphere
        for sphere_actor, joint_idx in self.joint_sphere_actors.items():
            if sphere_actor == picked_actor:
                if self.controller.selected_joint == joint_idx:
                    self.is_dragging = True
                    self.last_mouse_pos = (mouse_x, mouse_y)
                    self.plotter.disable()
                    self._notify(f"Dragging joint [{joint_idx}]")
                else:
                    self.controller.selected_joint = joint_idx
                    self.update_gizmo_only()
                    joint_name = self.controller.skeleton.joints[joint_idx].name
                    self._notify(f"Selected joint [{joint_idx}] {joint_name}")
                return

        # 3) click other actor -> deselect
        self.controller.selected_joint = None
        self.update_gizmo_only()
        self._notify("Click red sphere to select a joint.")

    def handle_mouse_move(self, event):
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
        joint_pos = G_current[self.controller.selected_joint, :3, 3]

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

        self.controller.joint_transforms[self.controller.selected_joint][:3, 3] += delta
        self.controller.update_children_cascade(self.controller.selected_joint, delta)

        self.last_mouse_pos = (x, y)
        self.pending_update = True
        if not self.update_timer.isActive():
            self.update_timer.start()

    def handle_mouse_release(self, event):
        if event.button() != Qt.LeftButton or not self.is_dragging:
            return
        self.is_dragging = False
        self.dragging_axis = None
        self.last_mouse_pos = None
        self.plotter.enable()

        self.update_timer.stop()
        self.update_deformed_mesh_only()

        if self.controller.selected_joint is not None:
            joint_name = self.controller.skeleton.joints[self.controller.selected_joint].name
            self._notify(f"Joint [{self.controller.selected_joint}] {joint_name} move finished")

    def _deferred_update(self):
        if self.pending_update:
            self.pending_update = False
            self.update_deformed_mesh_only()

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
        current_joint_positions = G_current[:, :3, 3]
        deformed_vertices = self.controller.compute_deformed_vertices()

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
            pickable=False,
        )

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

    def update_deformed_mesh_only(self):
        if self.mesh_actor is None:
            return

        deformed_vertices = self.controller.compute_deformed_vertices()
        vtk_points = self.mesh_actor.GetMapper().GetInput().GetPoints()
        vtk_array = numpy_to_vtk(deformed_vertices, deep=True)
        vtk_points.SetData(vtk_array)
        vtk_points.Modified()

        G_current = self.controller.compute_current_global_mats()
        current_joint_positions = G_current[:, :3, 3]

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
        current_joint_positions = G_current[:, :3, 3]
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
