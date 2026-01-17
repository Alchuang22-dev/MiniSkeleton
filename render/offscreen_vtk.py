# -*- coding: utf-8 -*-
"""Offscreen rendering with PyVista/VTK for frame capture."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pyvista as pv
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


class OffscreenVtkRenderer:
    """Minimal offscreen renderer based on PyVista/VTK."""

    def __init__(
        self,
        *,
        width: int = 1024,
        height: int = 1024,
        background: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.plotter = pv.Plotter(off_screen=True, window_size=(self.width, self.height))
        self._set_background(background)
        self._lights: list[pv.Light] = []
        self._ground_actor = None
        self._last_vertices = None
        self._shadow_actor = None
        self._ground_center = None
        self._ground_normal = None
        self._ground_extent = None

        self.mesh_actor = None
        self._mesh_points = None
        self._mesh_points_np = None
        self._rotation = None

    @staticmethod
    def _apply_rotation(vertices: np.ndarray, rotation: Optional[np.ndarray]) -> np.ndarray:
        if rotation is None:
            return np.asarray(vertices, dtype=np.float32)
        rot = np.asarray(rotation, dtype=np.float32)
        if rot.shape == (4, 4):
            rot = rot[:3, :3]
        if rot.shape != (3, 3):
            raise ValueError(f"rotation must be 3x3 or 4x4, got {rot.shape}")
        verts = np.asarray(vertices, dtype=np.float32)
        return (verts @ rot.T).astype(np.float32)

    def apply_ui_camera(self) -> None:
        """Match the viewport camera preset (reset + tilt + zoom)."""
        self.plotter.reset_camera()
        self.plotter.camera.elevation = 15
        self.plotter.camera.azimuth = -60
        self.plotter.camera.zoom(1.4)

    def set_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        rotation: Optional[np.ndarray] = None,
        apply_ui_camera: bool = True,
    ) -> None:
        self._rotation = rotation
        faces = np.asarray(faces, dtype=np.int64)
        faces_with_count = np.hstack([np.full((len(faces), 1), 3), faces])
        verts = self._apply_rotation(vertices, self._rotation)
        self._last_vertices = np.array(verts, copy=True)
        mesh = pv.PolyData(verts, faces_with_count)
        self.mesh_actor = self.plotter.add_mesh(
            mesh,
            color="lightblue",
            opacity=0.9,
            show_edges=False,
            smooth_shading=True,
            lighting=True,
        )
        self._configure_lights(verts)
        if self.mesh_actor is not None:
            try:
                self.mesh_actor.SetCastShadows(True)
            except Exception:
                pass
        if apply_ui_camera:
            self.apply_ui_camera()
        self._update_ground(verts)
        self._update_shadow(verts)

        vtk_points = self.mesh_actor.GetMapper().GetInput().GetPoints()
        vtk_array = vtk_points.GetData()
        self._mesh_points = vtk_points
        self._mesh_points_np = vtk_to_numpy(vtk_array)

    def update_vertices(self, vertices: np.ndarray) -> None:
        if self.mesh_actor is None:
            return
        verts = self._apply_rotation(vertices, self._rotation)
        self._last_vertices = np.array(verts, copy=True)
        if (
            self._mesh_points_np is None
            or self._mesh_points is None
            or self._mesh_points_np.shape != verts.shape
        ):
            vtk_points = self.mesh_actor.GetMapper().GetInput().GetPoints()
            vtk_array = numpy_to_vtk(verts, deep=True)
            vtk_points.SetData(vtk_array)
            vtk_points.Modified()
            self._mesh_points = vtk_points
            self._mesh_points_np = vtk_to_numpy(vtk_array)
        else:
            self._mesh_points_np[:] = verts
            self._mesh_points.Modified()
        self._update_ground(verts)
        self._update_shadow(verts)

    def _set_background(self, base_color: Tuple[float, float, float]) -> None:
        top_color = (0.85, 0.90, 0.96)
        try:
            self.plotter.set_background(base_color, top=top_color)
        except Exception:
            self.plotter.set_background(base_color)

    def _enable_shadows(self) -> None:
        """Deprecated: kept for compatibility but disabled to avoid shader issues."""
        return

    def _configure_lights(self, vertices: np.ndarray) -> None:
        if vertices.size == 0:
            return
        bounds = pv.PolyData(vertices).bounds
        cx = 0.5 * (bounds[0] + bounds[1])
        cy = 0.5 * (bounds[2] + bounds[3])
        cz = 0.5 * (bounds[4] + bounds[5])
        extent = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        extent = max(extent, 1e-3)

        if hasattr(self.plotter, "clear_lights"):
            self.plotter.clear_lights()
        self._lights = []

        def _set_light_color(light: pv.Light, color: tuple[float, float, float]) -> None:
            if hasattr(light, "SetColor"):
                light.SetColor(float(color[0]), float(color[1]), float(color[2]))
                return
            if hasattr(light, "set_color"):
                try:
                    light.set_color(color)
                except TypeError:
                    light.set_color(float(color[0]), float(color[1]), float(color[2]))

        key = pv.Light()
        key.position = (cx + extent, cy + extent * 0.8, cz + extent)
        key.focal_point = (cx, cy, cz)
        key.intensity = 0.9
        _set_light_color(key, (1.0, 1.0, 1.0))
        self.plotter.add_light(key)
        self._lights.append(key)

        fill = pv.Light()
        fill.position = (cx - extent * 0.8, cy + extent * 0.4, cz + extent * 0.2)
        fill.focal_point = (cx, cy, cz)
        fill.intensity = 0.4
        _set_light_color(fill, (0.95, 0.95, 1.0))
        self.plotter.add_light(fill)
        self._lights.append(fill)

        rim = pv.Light()
        rim.position = (cx, cy + extent * 0.9, cz - extent)
        rim.focal_point = (cx, cy, cz)
        rim.intensity = 0.25
        _set_light_color(rim, (1.0, 1.0, 1.0))
        self.plotter.add_light(rim)
        self._lights.append(rim)

    def _update_ground(self, vertices: np.ndarray) -> None:
        if vertices.size == 0:
            return
        verts = np.asarray(vertices, dtype=np.float32)
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        extent = float(np.linalg.norm(vmax - vmin))
        extent = max(extent, 1e-3)

        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        center = (vmin + vmax) * 0.5
        plane_center = center.copy()
        plane_center[1] = float(vmin[1] - extent * 0.05)

        plane = pv.Plane(
            center=plane_center.tolist(),
            direction=up.tolist(),
            i_size=extent * 2.2,
            j_size=extent * 2.2,
        )
        if self._ground_actor is None:
            self._ground_actor = self.plotter.add_mesh(
                plane,
                color=(0.94, 0.95, 0.97),
                opacity=1.0,
                smooth_shading=False,
                pickable=False,
            )
            try:
                self._ground_actor.SetReceiveShadows(True)
                self._ground_actor.SetCastShadows(False)
                prop = self._ground_actor.GetProperty()
                if hasattr(prop, "SetBackfaceCulling"):
                    prop.SetBackfaceCulling(True)
                if hasattr(prop, "SetAmbient"):
                    prop.SetAmbient(0.6)
                if hasattr(prop, "SetDiffuse"):
                    prop.SetDiffuse(0.4)
            except Exception:
                pass
        else:
            self._ground_actor.GetMapper().SetInputData(plane)
        self._ground_center = plane_center
        self._ground_normal = up
        self._ground_extent = extent

    def _update_shadow(self, vertices: np.ndarray) -> None:
        if vertices.size == 0:
            return
        if self._ground_center is None or self._ground_normal is None or self._ground_extent is None:
            return

        verts = np.asarray(vertices, dtype=np.float32)
        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        radius = 0.35 * max(vmax[0] - vmin[0], vmax[2] - vmin[2], 1e-3)
        shadow_center = self._ground_center + self._ground_normal * (self._ground_extent * 0.002)

        try:
            shadow_geom = pv.Disc(
                center=shadow_center.tolist(),
                normal=self._ground_normal.tolist(),
                inner=0.0,
                outer=radius,
                r_res=48,
                c_res=2,
            )
        except Exception:
            shadow_geom = pv.Plane(
                center=shadow_center.tolist(),
                direction=self._ground_normal.tolist(),
                i_size=radius * 2.0,
                j_size=radius * 2.0,
            )

        if self._shadow_actor is None:
            self._shadow_actor = self.plotter.add_mesh(
                shadow_geom,
                color=(0.0, 0.0, 0.0),
                opacity=0.18,
                smooth_shading=False,
                pickable=False,
                lighting=False,
            )
        else:
            self._shadow_actor.GetMapper().SetInputData(shadow_geom)

    def apply_camera_state(
        self,
        *,
        position: Tuple[float, float, float],
        focal_point: Tuple[float, float, float],
        view_up: Tuple[float, float, float],
        view_angle: float | None = None,
        clipping_range: Tuple[float, float] | None = None,
    ) -> None:
        """Apply explicit camera parameters (e.g., copied from the UI viewport)."""
        cam = self.plotter.camera
        cam.SetPosition(*[float(v) for v in position])
        cam.SetFocalPoint(*[float(v) for v in focal_point])
        cam.SetViewUp(*[float(v) for v in view_up])
        if view_angle is not None:
            cam.SetViewAngle(float(view_angle))
        if clipping_range is not None:
            cam.SetClippingRange(float(clipping_range[0]), float(clipping_range[1]))
        if self._last_vertices is not None:
            self._update_ground(self._last_vertices)

    def render_to_file(self, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Ensure latest mesh updates are rendered before capture.
        self.plotter.render()
        self.plotter.screenshot(out_path, transparent_background=False)

    def close(self) -> None:
        self.plotter.close()
