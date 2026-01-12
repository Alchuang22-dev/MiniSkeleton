# -*- coding: utf-8 -*-
"""Simple head-shake keyframe demo for bake_frames."""

from __future__ import annotations

import numpy as np

from rigging.gltf_loader import load_mesh_and_skeleton_from_glb
from rigging.mesh_io import Mesh
from rigging.skeleton import Skeleton, mat4_from_rt, euler_xyz_to_rot
from rigging.weights_heat import HeatWeightsConfig, compute_heat_weights
from rigging.weights_nearest import compute_nearest_bilinear_weights
from scene.asset import RiggedAsset
from scene.choreography import Scene
from scene.timeline import Timeline


PREFERRED_HEAD_JOINTS = ("head", "neck_mid", "neck_root", "neck")
PREFERRED_NECK_JOINTS = ("neck_mid", "neck_root", "neck")


def _pick_joint(names: list[str], preferred: tuple[str, ...]) -> str:
    lower_map = {name.lower(): name for name in names}
    for key in preferred:
        exact = lower_map.get(key.lower())
        if exact:
            return exact
    for key in preferred:
        key_lower = key.lower()
        for name in names:
            if key_lower in name.lower():
                return name
    return names[-1] if names else ""


def _rot_y(angle_rad: float) -> np.ndarray:
    R = euler_xyz_to_rot(0.0, angle_rad, 0.0)
    t = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return mat4_from_rt(R, t)


def build_scene() -> tuple[Scene, float]:
    glb_path = "data/single/spot/spot.glb"
    (
        verts,
        faces,
        names,
        parents,
        positions,
        bind_mats,
    ) = load_mesh_and_skeleton_from_glb(
        glb_path,
        return_bind_mats=True,
    )

    mesh = Mesh(vertices=verts.astype(np.float32), faces=faces.astype(np.int32))
    if bind_mats is not None:
        skeleton = Skeleton.from_bind_matrices(names, parents, bind_mats)
    else:
        skeleton = Skeleton.from_bind_positions(names, parents, positions)

    weights = None
    try:
        cfg = HeatWeightsConfig()
        weights = compute_heat_weights(mesh, skeleton, cfg)
        print("[INFO] Using heat weights (Pinocchio-style).")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Heat weights failed: {exc}; falling back to nearest-bilinear.")
        weights = compute_nearest_bilinear_weights(mesh.vertices, skeleton)
    asset = RiggedAsset(name="spot", mesh=mesh, skeleton=skeleton, weights=weights)

    head_name = _pick_joint(list(names), PREFERRED_HEAD_JOINTS)
    neck_name = _pick_joint(list(names), PREFERRED_NECK_JOINTS)
    timeline = Timeline(duration=1.5)
    if head_name:
        timeline.add_keyframe(head_name, 0.0, _rot_y(-0.35))
        timeline.add_keyframe(head_name, 0.5, _rot_y(0.35))
        timeline.add_keyframe(head_name, 1.0, _rot_y(-0.35))
        timeline.add_keyframe(head_name, 1.5, _rot_y(0.0))
    if neck_name and neck_name != head_name:
        timeline.add_keyframe(neck_name, 0.0, _rot_y(-0.15))
        timeline.add_keyframe(neck_name, 0.5, _rot_y(0.15))
        timeline.add_keyframe(neck_name, 1.0, _rot_y(-0.15))
        timeline.add_keyframe(neck_name, 1.5, _rot_y(0.0))

    scene = Scene()
    scene.add_asset(asset, timeline)
    return scene, timeline.duration


if __name__ == "__main__":
    scene, duration = build_scene()
    print(f"Scene ready. Duration={duration:.2f}s, assets={len(scene.assets)}")
