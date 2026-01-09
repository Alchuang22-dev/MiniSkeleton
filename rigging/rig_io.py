# -*- coding: utf-8 -*-
"""Rig bundle I/O for mesh + skeleton data."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np

from rigging.mesh_io import Mesh
from rigging.skeleton import Skeleton


def _names_array(names: Iterable[str]) -> np.ndarray:
    names = list(names)
    if not names:
        return np.array([], dtype="<U1")
    max_len = max(len(n) for n in names)
    return np.array(names, dtype=f"<U{max_len}")


def save_rig_npz(
    path: str,
    mesh: Mesh,
    skeleton: Skeleton,
    joint_names: list[str],
    joint_parents: list[int],
) -> None:
    """Save a mesh + skeleton bundle to a .npz file."""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    bind_local = np.stack([j.bind_local for j in skeleton.joints], axis=0).astype(np.float32)
    inv_bind = np.stack([j.inv_bind for j in skeleton.joints], axis=0).astype(np.float32)

    names_arr = _names_array(joint_names)
    parents_arr = np.asarray(joint_parents, dtype=np.int32)

    np.savez_compressed(
        path,
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int32),
        joint_names=names_arr,
        parents=parents_arr,
        bind_local=bind_local,
        inv_bind=inv_bind,
    )
