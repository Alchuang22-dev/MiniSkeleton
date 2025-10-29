# -*- coding: utf-8 -*-
"""
Mesh I/O & preprocessing utilities for rigging/weights/heat methods.

Design goals
------------
1) General-purpose loader/saver(OBJ/PLY/GLB...), robust to Scenes (multi-geometry).
2) Optional preprocessing switches (center/unit_scale/axis_up/weld/triangulate).
3) Topology utilities: adjacency lists, (uniform) Laplacian (CSR), face areas.
4) Repair helpers: remove degenerate/duplicate faces, unreferenced verts.
5) Non-destructive defaults: NO cow-specific hard-coded transforms by default.

Notes
-----
- Keep transforms optional to avoid embedding dataset-specific bias.
- Use dense np arrays for vertices/faces; expose scipy.sparse for Laplacian.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import trimesh

try:
    import scipy.sparse as sp
except Exception as _:
    sp = None  # Laplacian/adjacency CSR requires scipy; guard with checks.


# -----------------------
# Core data structure
# -----------------------

@dataclass
class Mesh:
    """Minimal mesh container used by rigging pipeline."""
    vertices: np.ndarray          # (N,3) float32/64
    faces: np.ndarray             # (M,3) int32
    normals: Optional[np.ndarray] = None  # (N,3) or None
    uvs: Optional[np.ndarray] = None      # (N,2) per-vertex UV (optional)
    vertex_colors: Optional[np.ndarray] = None  # (N,3|4)

    # Cached bbox (lazy)
    _aabb_min: Optional[np.ndarray] = None
    _aabb_max: Optional[np.ndarray] = None

    # ---------- basic props ----------
    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def aabb(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._aabb_min is None or self._aabb_max is None:
            self._aabb_min = np.min(self.vertices, axis=0)
            self._aabb_max = np.max(self.vertices, axis=0)
        return self._aabb_min, self._aabb_max

    @property
    def bbox_diag(self) -> float:
        a, b = self.aabb
        return float(np.linalg.norm(b - a))

    # ---------- conversions ----------
    def to_trimesh(self) -> trimesh.Trimesh:
        tm = trimesh.Trimesh(
            vertices=self.vertices.copy(),
            faces=self.faces.copy(),
            process=False,
        )
        # Attach optional attrs
        if self.normals is not None:
            tm.vertex_normals = self.normals.copy()
        if self.vertex_colors is not None:
            tm.visual.vertex_colors = self.vertex_colors.copy()
        if self.uvs is not None:
            # trimesh expects per-vertex uv via visual.uv mapped by faces, here we
            # store per-vertex UV; this is sufficient for debug/preview purposes.
            tm.visual.uv = self.uvs.copy()
        return tm

    @staticmethod
    def from_trimesh(tm: trimesh.Trimesh) -> "Mesh":
        v = np.asarray(tm.vertices)
        f = np.asarray(tm.faces, dtype=np.int32)
        n = None
        try:
            if tm.vertex_normals is not None and len(tm.vertex_normals) == len(v):
                n = np.asarray(tm.vertex_normals)
        except Exception:
            pass
        uvs = None
        try:
            if tm.visual is not None and getattr(tm.visual, "uv", None) is not None:
                u = np.asarray(tm.visual.uv)
                if u.ndim == 2 and u.shape[0] == v.shape[0]:
                    uvs = u
        except Exception:
            pass
        colors = None
        try:
            if tm.visual is not None and getattr(tm.visual, "vertex_colors", None) is not None:
                c = np.asarray(tm.visual.vertex_colors)
                if c.ndim == 2 and c.shape[0] == v.shape[0]:
                    colors = c
        except Exception:
            pass
        return Mesh(vertices=v, faces=f, normals=n, uvs=uvs, vertex_colors=colors)

    # ---------- safe utilities ----------
    def ensure_vertex_normals(self, recompute: bool = False) -> None:
        """Compute per-vertex normals if missing or if recompute=True."""
        if (self.normals is None) or recompute:
            tm = self.to_trimesh()
            # trimesh will compute smooth vertex normals
            self.normals = np.asarray(tm.vertex_normals)

    def recenter(self) -> None:
        """Center mesh at origin (optional; NOT applied by default)."""
        c = (self.aabb[0] + self.aabb[1]) * 0.5
        self.vertices = (self.vertices - c[None, :]).astype(self.vertices.dtype)
        # invalidate cache
        self._aabb_min = self._aabb_max = None

    def unit_scale(self, target_diag: float = 1.0) -> None:
        """Scale mesh so that bbox diagonal == target_diag (optional)."""
        diag = self.bbox_diag
        if diag > 0:
            s = float(target_diag / diag)
            self.vertices = (self.vertices * s).astype(self.vertices.dtype)
            self._aabb_min = self._aabb_max = None

    def axis_up(self, up: str = "z", current_up: Optional[str] = None) -> None:
        """
        (Optional) Reorient mesh so that `up` becomes the +axis.
        Only basic cases (x|y|z) supported. No-op by default.
        """
        axes = {"x": 0, "y": 1, "z": 2}
        if current_up is None or up == current_up or up not in axes:
            return
        # Simple heuristic: swap axes; sign assumed positive
        cu, tu = axes[current_up], axes[up]
        v = self.vertices.copy()
        v[:, [tu, cu]] = v[:, [cu, tu]]
        self.vertices = v
        self._aabb_min = self._aabb_max = None

    # ---------- topology metrics ----------
    def face_areas(self) -> np.ndarray:
        v = self.vertices
        f = self.faces
        a = v[f[:, 0]]
        b = v[f[:, 1]]
        c = v[f[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    def is_triangular(self) -> bool:
        return self.faces.ndim == 2 and self.faces.shape[1] == 3

    def summary(self) -> str:
        a, b = self.aabb
        return (f"Mesh(nV={self.n_vertices}, nF={self.n_faces}, "
                f"aabb_min={a.tolist()}, aabb_max={b.tolist()}, "
                f"bbox_diag={self.bbox_diag:.6f})")


# -----------------------
# Loading / Saving
# -----------------------

def _merge_scene(scene: trimesh.Scene) -> trimesh.Trimesh:
    """
    Merge all geometry from a trimesh.Scene into a single Trimesh.
    """
    # Convert to one Trimesh in world coords
    tm = trimesh.util.concatenate(
        [g.copy().apply_transform(scene.graph.get_transform(node))
         for node, g in scene.geometry.items()]
    )
    return tm

def load_mesh(
    path: str,
    *,
    triangulate: bool = True,
    weld: bool = True,
    dedup_faces: bool = True,
    remove_unreferenced: bool = True,
    center: bool = False,
    scale_to_unit: bool = False,
    up_axis: Optional[str] = None,    # e.g., "z"; None = no-op
    current_up: Optional[str] = None, # e.g., "y"; None = unknown
) -> Mesh:
    """
    Load mesh from file with optional preprocessing.
    - NO cow-specific transforms by default (center/scale/up are opt-in).
    """
    path = str(path)
    loaded = trimesh.load(path, force=None, process=False)
    if isinstance(loaded, trimesh.Scene):
        tm = _merge_scene(loaded)
    elif isinstance(loaded, trimesh.Trimesh):
        tm = loaded
    else:
        # Many formats may return a Path/Dict; attempt to dump to scene then sum.
        try:
            tm = loaded.dump().sum()
        except Exception as e:
            raise RuntimeError(f"Unsupported mesh container from {path}: {type(loaded)}") from e

    # Basic cleaning
    if triangulate and hasattr(tm, "triangles") and tm.faces.shape[1] != 3:
        # trimesh will triangulate on creation in most cases, but guard just in case
        tm = tm.triangulate()

    if dedup_faces and hasattr(tm, "remove_duplicate_faces"):
        tm.remove_duplicate_faces()

    if remove_unreferenced and hasattr(tm, "remove_unreferenced_vertices"):
        tm.remove_unreferenced_vertices()

    if weld and hasattr(tm, "merge_vertices"):
        tm.merge_vertices()  # weld close vertices using default tolerance

    # Convert to Mesh struct
    mesh = Mesh.from_trimesh(tm)

    # Optional, user-controlled transforms (not applied by default)
    if center:
        mesh.recenter()
    if scale_to_unit:
        mesh.unit_scale(1.0)
    if up_axis is not None:
        mesh.axis_up(up=up_axis, current_up=current_up)

    # Ensure normals exist for visualization
    mesh.ensure_vertex_normals(recompute=False)
    return mesh

def save_mesh(mesh: Mesh, path: str) -> None:
    """
    Save Mesh to disk. Format inferred from extension.
    Supported (by trimesh): .obj, .ply, .stl, .glb/.gltf (when available).
    """
    tm = mesh.to_trimesh()
    path = str(path)
    ext = Path(path).suffix.lower()
    # For OBJ/PLY, trimesh export is robust.
    # For GLB/GLTF, attributes like colors/uv may need additional care.
    data = tm.export(file_type=ext[1:] if ext.startswith(".") else ext)
    # If export returns bytes, write them directly; else it's a str.
    mode = "wb" if isinstance(data, (bytes, bytearray)) else "w"
    with open(path, mode) as f:
        f.write(data)


# -----------------------
# Topology / Graph utils
# -----------------------

def compute_vertex_adjacency(mesh: Mesh) -> Tuple[List[List[int]], Optional["sp.csr_matrix"]]:
    """
    Build per-vertex adjacency list and (optional) uniform Laplacian in CSR.
    Returns:
        neighbors: list of length N, each is a sorted unique neighbor list.
        L: scipy.sparse.csr_matrix or None if scipy not available.
    """
    n = mesh.n_vertices
    neighbors: List[set] = [set() for _ in range(n)]
    f = mesh.faces
    # undirected edges from triangles
    for i0, i1, i2 in f:
        neighbors[i0].add(i1); neighbors[i0].add(i2)
        neighbors[i1].add(i0); neighbors[i1].add(i2)
        neighbors[i2].add(i0); neighbors[i2].add(i1)

    neigh_lists: List[List[int]] = [sorted(list(s)) for s in neighbors]

    if sp is None:
        return neigh_lists, None

    # Build adjacency A and degree D; Laplacian L = D - A
    rows, cols = [], []
    for i, lst in enumerate(neigh_lists):
        rows.extend([i] * len(lst))
        cols.extend(lst)
    data = np.ones(len(rows), dtype=np.float64)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    # Symmetrize
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(axis=1)).ravel()
    D = sp.diags(deg, 0, shape=(n, n))
    L = D - A
    return neigh_lists, L


# -----------------------
# Repair helpers
# -----------------------

def repair_mesh(
    mesh: Mesh,
    *,
    remove_degenerate: bool = True,
    remove_zero_area: bool = True,
    fill_holes: bool = False
) -> Mesh:
    """
    Apply inexpensive structural repairs via trimesh.
    Note: hole filling is often destructive; keep it opt-in.
    """
    tm = mesh.to_trimesh()
    if remove_degenerate:
        try:
            tm.remove_degenerate_faces()
        except Exception:
            pass
    if remove_zero_area:
        try:
            # filter by area threshold
            areas = tm.area_faces
            keep = areas > 1e-16
            tm.update_faces(keep)
            tm.remove_unreferenced_vertices()
        except Exception:
            pass
    if fill_holes:
        try:
            trimesh.repair.fill_holes(tm)
        except Exception:
            pass
    return Mesh.from_trimesh(tm)


# -----------------------
# Debug / Visualization aids
# -----------------------

def colorize_by_scalar(mesh: Mesh, scalar: np.ndarray, cmap: str = "viridis") -> Mesh:
    """
    Attach per-vertex colors from a scalar field for quick debug preview.
    Requires matplotlib (optional). If not available, falls back to gray.
    """
    try:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=float(np.min(scalar)), vmax=float(np.max(scalar)))
        colors = cm.get_cmap(cmap)(norm(scalar))  # (N,4) float in [0,1]
        colors = (colors * 255.0).astype(np.uint8)
    except Exception:
        colors = np.full((mesh.n_vertices, 4), fill_value=200, dtype=np.uint8)
        colors[:, 3] = 255
    out = Mesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        normals=mesh.normals.copy() if mesh.normals is not None else None,
        uvs=mesh.uvs.copy() if mesh.uvs is not None else None,
        vertex_colors=colors,
    )
    return out


# -----------------------
# Convenience factory
# -----------------------

def make_unit_cube(center: bool = True) -> Mesh:
    """Utility: create a simple cube for tests."""
    tm = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh = Mesh.from_trimesh(tm)
    if not center:
        # move to + quadrant
        mesh.vertices = mesh.vertices + np.array([0.5, 0.5, 0.5], dtype=mesh.vertices.dtype)
        mesh._aabb_min = mesh._aabb_max = None
    return mesh


# -----------------------
# Example usage (doc)
# -----------------------
"""
# Load (generic, no cow-specific transforms):
mesh = load_mesh("data/single/character.obj")

# Optional, if your dataset needs:
mesh = load_mesh("data/single/character.obj", center=True, scale_to_unit=True, up_axis="z", current_up="y")

# Adjacency & Laplacian (for heat/biharmonic weights):
nbrs, L = compute_vertex_adjacency(mesh)

# Ensure normals for preview:
mesh.ensure_vertex_normals()

# Save OBJ
save_mesh(mesh, "out/debug_character.obj")

# Visualize weights as colors:
mesh_colored = colorize_by_scalar(mesh, weights[:, j])
save_mesh(mesh_colored, "out/weights_j.obj")
"""
