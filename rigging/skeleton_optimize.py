# -*- coding: utf-8 -*-
"""Quadruped skeleton checks and simple optimization heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from rigging.mesh_io import Mesh


@dataclass
class SkeletonCheckReport:
    axis: int
    delta: float
    applied: bool
    spine_nodes: List[int]
    warnings: List[str]


def _build_adjacency(parents: np.ndarray, positions: np.ndarray) -> List[List[Tuple[int, float]]]:
    n = len(parents)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i, p in enumerate(parents):
        if p < 0:
            continue
        w = float(np.linalg.norm(positions[i] - positions[p]))
        adj[i].append((p, w))
        adj[p].append((i, w))
    return adj


def _farthest_node(adj: List[List[Tuple[int, float]]], start: int) -> Tuple[int, List[int], List[float]]:
    n = len(adj)
    parent = [-1] * n
    dist = [-1.0] * n
    dist[start] = 0.0
    stack = [start]
    while stack:
        u = stack.pop()
        for v, w in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + w
                parent[v] = u
                stack.append(v)
    far = max(range(n), key=lambda i: dist[i])
    return far, parent, dist


def _longest_path_nodes(parents: np.ndarray, positions: np.ndarray) -> List[int]:
    if len(parents) == 0:
        return []
    adj = _build_adjacency(parents, positions)
    a, _, _ = _farthest_node(adj, 0)
    b, parent, _ = _farthest_node(adj, a)
    path = []
    cur = b
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    return path


def optimize_quadruped_bind_positions(
    mesh: Mesh,
    parents: np.ndarray,
    positions: np.ndarray,
    *,
    axis: int | None = None,
    threshold_ratio: float = 0.05,
) -> Tuple[np.ndarray, SkeletonCheckReport]:
    """
    Check and apply a simple quadruped skeleton optimization.

    Strategy:
    - Use the longest joint path as an approximate spine.
    - Align spine center to mesh center along the main axis.
    """
    pos = np.array(positions, dtype=np.float32, copy=True)
    vmin = np.min(mesh.vertices, axis=0)
    vmax = np.max(mesh.vertices, axis=0)
    diag = float(np.linalg.norm(vmax - vmin))
    if axis is None:
        axis = int(np.argmax(vmax - vmin))

    spine_nodes = _longest_path_nodes(parents, pos)
    warnings: List[str] = []

    if not spine_nodes:
        return pos, SkeletonCheckReport(axis=axis, delta=0.0, applied=False, spine_nodes=[], warnings=["No joints found."])

    spine_center = float(np.mean(pos[spine_nodes, axis]))
    mesh_center = float((vmin[axis] + vmax[axis]) * 0.5)
    delta = mesh_center - spine_center

    margin = 0.05 * diag
    out_mask = np.any((pos < (vmin - margin)) | (pos > (vmax + margin)), axis=1)
    out_count = int(np.sum(out_mask))
    if out_count > 0:
        warnings.append(f"{out_count} joints are outside the mesh bounds (+/-{margin:.3f}).")

    applied = False
    if abs(delta) > threshold_ratio * diag:
        shift = np.zeros(3, dtype=np.float32)
        shift[axis] = delta
        pos += shift[None, :]
        applied = True
    else:
        warnings.append("Spine center is already near mesh center; no shift applied.")

    report = SkeletonCheckReport(
        axis=axis,
        delta=float(delta),
        applied=applied,
        spine_nodes=spine_nodes,
        warnings=warnings,
    )
    return pos, report
