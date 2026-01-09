# -*- coding: utf-8 -*-
"""GLB export helpers for skeleton-only assets."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from pygltflib import GLTF2, Asset, Node, Scene

from rigging.skeleton import Skeleton


def _matrix_to_gltf(mat: np.ndarray) -> List[float]:
    """Convert a 4x4 row-major matrix to glTF column-major list."""
    return mat.T.reshape(-1).astype(float).tolist()


def export_skeleton_glb(
    skeleton: Skeleton,
    path: str,
    *,
    names: Optional[Iterable[str]] = None,
    parents: Optional[Iterable[int]] = None,
) -> None:
    """Export skeleton joint hierarchy to a GLB file (nodes only)."""
    if names is None:
        names = [j.name for j in skeleton.joints]
    if parents is None:
        parents = [j.parent for j in skeleton.joints]

    names_list = list(names)
    parents_list = [int(p) for p in parents]

    n = len(names_list)
    children_map: List[List[int]] = [[] for _ in range(n)]
    for i, p in enumerate(parents_list):
        if p >= 0:
            children_map[p].append(i)

    nodes: List[Node] = []
    for i, joint in enumerate(skeleton.joints):
        node = Node(
            name=names_list[i],
            children=children_map[i] if children_map[i] else None,
            matrix=_matrix_to_gltf(np.asarray(joint.bind_local, dtype=np.float32)),
        )
        nodes.append(node)

    root_nodes = [i for i, p in enumerate(parents_list) if p < 0]

    gltf = GLTF2(
        asset=Asset(version="2.0"),
        scenes=[Scene(nodes=root_nodes)],
        scene=0,
        nodes=nodes,
    )
    gltf.save(path)
