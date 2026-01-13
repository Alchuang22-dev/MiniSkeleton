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
        self.plotter.set_background(background)

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
        mesh = pv.PolyData(verts, faces_with_count)
        self.mesh_actor = self.plotter.add_mesh(
            mesh,
            color="lightblue",
            opacity=0.9,
            show_edges=False,
            smooth_shading=True,
        )
        if apply_ui_camera:
            self.apply_ui_camera()

        vtk_points = self.mesh_actor.GetMapper().GetInput().GetPoints()
        vtk_array = vtk_points.GetData()
        self._mesh_points = vtk_points
        self._mesh_points_np = vtk_to_numpy(vtk_array)

    def update_vertices(self, vertices: np.ndarray) -> None:
        if self.mesh_actor is None:
            return
        verts = self._apply_rotation(vertices, self._rotation)
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

    def render_to_file(self, out_path: str) -> None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Ensure latest mesh updates are rendered before capture.
        self.plotter.render()
        self.plotter.screenshot(out_path, transparent_background=False)

    def close(self) -> None:
        self.plotter.close()
