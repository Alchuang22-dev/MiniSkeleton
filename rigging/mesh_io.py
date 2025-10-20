# -*- coding: utf-8 -*-
# 关键的骨架/图形矩阵IO总行
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Mesh:
    vertices: np.ndarray  # (N,3)
    faces: np.ndarray     # (M,3) int
    normals: np.ndarray | None = None
    uvs: np.ndarray | None = None

def load_mesh(path: str) -> Mesh:
    """TODO: 读取 OBJ/PLY；保证顶点/面合法；必要时重建法线。"""
    raise NotImplementedError

def save_mesh(mesh: Mesh, path: str) -> None:
    """TODO: 保存为 OBJ/PLY；可选写出权重/骨骼调试属性。"""
    raise NotImplementedError

def compute_vertex_adjacency(mesh: Mesh) -> tuple[list[list[int]], np.ndarray]:
    """TODO: 邻接表/拉普拉斯矩阵构建（为 heat weights 提供支持）。"""
    raise NotImplementedError
