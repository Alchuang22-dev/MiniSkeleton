# -*- coding: utf-8 -*-
# 绑定骨架方面的函数
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

@dataclass
class Joint:
    name: str
    parent: Optional[str]
    bind_pose: np.ndarray      # (4,4) 关节在绑定时的局部/或全局矩阵（按你的约定）
    offset_matrix: np.ndarray  # (4,4) 逆绑定矩阵（skin bind）



# -----------------------
# Math helpers
# -----------------------

def _eye4(dtype=np.float32) -> np.ndarray:
    return np.eye(4, dtype=dtype)

def mat4_from_rt(R: np.ndarray, t: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Compose 4x4 from 3x3 rotation and 3 translation."""
    M = _eye4(dtype)
    M[:3, :3] = R
    M[:3, 3] = t
    return M

def mat4_translate(t: Sequence[float], dtype=np.float32) -> np.ndarray:
    M = _eye4(dtype)
    M[:3, 3] = np.asarray(t, dtype=dtype)
    return M

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) to 3x3 rotation."""
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy+zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx+zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx+yy)]
    ], dtype=np.float32)
    return R

def euler_xyz_to_rot(rx, ry, rz) -> np.ndarray:
    """Euler XYZ (radians) to rotation matrix."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float32)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float32)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


# -----------------------
# Core data structures
# -----------------------

@dataclass
class Joint:
    """A single joint with parent index and bind/local transforms."""
    name: str
    parent: int  # -1 for root
    bind_local: np.ndarray = field(default_factory=_eye4)  # 4x4 local in bind pose
    inv_bind: np.ndarray = field(default_factory=_eye4)    # 4x4 inverse of bind GLOBAL

    def __post_init__(self):
        # Shape guards
        if self.bind_local.shape != (4,4):
            raise ValueError(f"bind_local must be (4,4), got {self.bind_local.shape}")
        if self.inv_bind.shape != (4,4):
            raise ValueError(f"inv_bind must be (4,4), got {self.inv_bind.shape}")


class Skeleton:
    """
    Generic skeleton with FK and skinning matrix production.
    """
    def __init__(self, joints: Optional[List[Joint]] = None):
        self.joints: List[Joint] = joints if joints is not None else []
        self.name_to_index: Dict[str, int] = {j.name: i for i, j in enumerate(self.joints)}

    # -------- basic props --------
    @property
    def n(self) -> int:
        return len(self.joints)

    def parents(self) -> np.ndarray:
        return np.array([j.parent for j in self.joints], dtype=np.int32)

    # -------- building / editing --------
    def add_joint(self, joint: Joint) -> None:
        self.joints.append(joint)
        self.name_to_index[joint.name] = len(self.joints) - 1

    def joint_index(self, name: str) -> int:
        return self.name_to_index[name]

    def joint_names(self) -> List[str]:
        return [j.name for j in self.joints]

    # -------- bind pose handling --------
    @staticmethod
    def from_bind_positions(
        names: Sequence[str],
        parents: Sequence[int],
        positions: np.ndarray,
        dtype=np.float32
    ) -> "Skeleton":
        """
        Build skeleton from bind joint positions in GLOBAL space.
        Assumptions:
          - `positions[j]` is the GLOBAL position of joint j in bind pose.
          - Parent at index p has position positions[p].
          - Local bind is set so that global bind reproduces these positions with zero rotation.
        """
        names = list(names)
        parents = np.asarray(parents, dtype=np.int32)
        pos = np.asarray(positions, dtype=dtype)  # (J,3)

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions must be (J,3) array")

        J = len(names)
        if not (len(parents) == J and pos.shape[0] == J):
            raise ValueError("names/parents/positions size mismatch")

        joints: List[Joint] = []
        # local bind: translation relative to parent; rotation = I
        for j in range(J):
            p = parents[j]
            if p < 0:
                t = pos[j]
            else:
                t = pos[j] - pos[p]
            bind_local = mat4_translate(t, dtype=dtype)
            joints.append(Joint(name=names[j], parent=p, bind_local=bind_local, inv_bind=_eye4(dtype)))

        skel = Skeleton(joints)
        # compute global bind and inverse-bind
        G_bind = skel.forward_kinematics_local([j.bind_local for j in skel.joints])  # (J,4,4)
        for j in range(J):
            joints[j].inv_bind = np.linalg.inv(G_bind[j]).astype(dtype)
        return skel

    # -------- FK --------
    def forward_kinematics_local(self, local_T: Sequence[np.ndarray]) -> np.ndarray:
        """
        Local transforms -> Global transforms (FK).
        local_T: list/array of J items, each (4,4).
        Returns: (J,4,4) global matrices.
        """
        J = self.n
        if len(local_T) != J:
            raise ValueError("local_T length must equal number of joints")
        parents = self.parents()

        G = np.zeros((J, 4, 4), dtype=np.float32)
        for j in range(J):
            p = parents[j]
            if p < 0:
                G[j] = local_T[j]
            else:
                G[j] = G[p] @ local_T[j]
        return G

    def forward_kinematics_pose(
        self,
        pose_local_over_bind: Sequence[np.ndarray]
    ) -> np.ndarray:
        """
        Apply pose on top of BIND LOCAL:
          global = FK( bind_local @ pose_local_over_bind )
        pose_local_over_bind[j]: (4,4) local delta from bind.
        Returns GLOBAL (J,4,4).
        """
        if len(pose_local_over_bind) != self.n:
            raise ValueError("pose_local_over_bind length must equal number of joints")

        combined_locals = []
        for j, joint in enumerate(self.joints):
            combined_locals.append(joint.bind_local @ pose_local_over_bind[j])

        return self.forward_kinematics_local(combined_locals)

    # -------- Skinning matrices --------
    def skinning_matrices(
        self,
        pose_local_over_bind: Sequence[np.ndarray]
    ) -> np.ndarray:
        """
        Produce skinning matrices per joint for LBS:
          M_skin[j] = G_current[j] @ inv_bind[j]
        where
          G_current = FK( bind_local @ pose_local_over_bind ).
        Returns: (J,4,4)
        """
        G_curr = self.forward_kinematics_pose(pose_local_over_bind)  # (J,4,4)
        M = np.zeros_like(G_curr)
        for j, joint in enumerate(self.joints):
            M[j] = G_curr[j] @ joint.inv_bind
        return M

    # -------- convenient pose constructors --------
    def make_identity_pose(self) -> List[np.ndarray]:
        """Local identity deltas (i.e., bind pose)."""
        return [_eye4(np.float32) for _ in range(self.n)]

    def pose_from_rt(
        self,
        R_list: Sequence[np.ndarray],
        t_list: Sequence[np.ndarray]
    ) -> List[np.ndarray]:
        """Create pose_local_over_bind from per-joint (R,t) in LOCAL space."""
        if len(R_list) != self.n or len(t_list) != self.n:
            raise ValueError("R_list and t_list must match joint count")
        pose = []
        for R, t in zip(R_list, t_list):
            pose.append(mat4_from_rt(np.asarray(R, dtype=np.float32),
                                     np.asarray(t, dtype=np.float32)))
        return pose

    def pose_from_quat_t(
        self,
        q_list: Sequence[np.ndarray],
        t_list: Sequence[np.ndarray]
    ) -> List[np.ndarray]:
        """Create pose from per-joint (quat(x,y,z,w), t)."""
        if len(q_list) != self.n or len(t_list) != self.n:
            raise ValueError("q_list and t_list must match joint count")
        pose = []
        for q, t in zip(q_list, t_list):
            R = quat_to_rot(np.asarray(q, dtype=np.float32))
            pose.append(mat4_from_rt(R, np.asarray(t, dtype=np.float32)))
        return pose

    def pose_from_euler_t(
        self,
        euler_xyz_list: Sequence[Tuple[float,float,float]],
        t_list: Sequence[np.ndarray]
    ) -> List[np.ndarray]:
        """Create pose from per-joint Euler XYZ (rad) and translation."""
        if len(euler_xyz_list) != self.n or len(t_list) != self.n:
            raise ValueError("euler_xyz_list and t_list must match joint count")
        pose = []
        for (rx, ry, rz), t in zip(euler_xyz_list, t_list):
            R = euler_xyz_to_rot(rx, ry, rz)
            pose.append(mat4_from_rt(R, np.asarray(t, dtype=np.float32)))
        return pose


# -----------------------
# Optional: Quadruped autoplace (cow-agnostic)
# -----------------------

def quadruped_auto_place_from_bbox(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    up_axis: str = "y"
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Create a simple quadruped-like joint layout from bbox.
    Returns:
      names:   List[str]
      parents: (J,) int32
      pos_g:   (J,3) float32 GLOBAL bind positions
    Notes:
      - This is a generic template; not tailored to any specific model.
      - up_axis can be 'y' or 'z' depending on dataset convention.
    """
    c = (bbox_min + bbox_max) * 0.5
    L = (bbox_max - bbox_min)

    # axis mapping
    # internal convention: X forward/back, Y up, Z left/right
    # if dataset up is z, we swap y/z when computing offsets
    def ax(v):
        v = np.array(v, dtype=np.float32)
        if up_axis.lower() == "z":
            # swap y <-> z
            v = v[[0,2,1]]
        return v

    x0, y0, z0 = ax(c)
    Lx, Ly, Lz = ax(L)

    root   = np.array([x0 - 0.35*Lx, y0 - 0.10*Ly, z0], dtype=np.float32)
    hips   = np.array([x0 - 0.25*Lx, y0 - 0.05*Ly, z0], dtype=np.float32)
    chest  = np.array([x0 + 0.15*Lx, y0 + 0.00*Ly, z0], dtype=np.float32)
    neck   = np.array([x0 + 0.35*Lx, y0 + 0.10*Ly, z0], dtype=np.float32)
    head   = np.array([x0 + 0.50*Lx, y0 + 0.20*Ly, z0], dtype=np.float32)

    zoff = 0.25 * Lz

    shoulder_L = chest + np.array([0.0, 0.00*Ly, +zoff], dtype=np.float32)
    shoulder_R = chest + np.array([0.0, 0.00*Ly, -zoff], dtype=np.float32)
    elbow_L    = shoulder_L + np.array([0.0, -0.25*Ly, 0.0], dtype=np.float32)
    elbow_R    = shoulder_R + np.array([0.0, -0.25*Ly, 0.0], dtype=np.float32)
    wrist_L    = elbow_L    + np.array([0.0, -0.25*Ly, 0.0], dtype=np.float32)
    wrist_R    = elbow_R    + np.array([0.0, -0.25*Ly, 0.0], dtype=np.float32)

    hip_L      = hips      + np.array([0.0, 0.00*Ly, +zoff], dtype=np.float32)
    hip_R      = hips      + np.array([0.0, 0.00*Ly, -zoff], dtype=np.float32)
    knee_L     = hip_L     + np.array([0.0, -0.30*Ly, 0.0], dtype=np.float32)
    knee_R     = hip_R     + np.array([0.0, -0.30*Ly, 0.0], dtype=np.float32)
    ankle_L    = knee_L    + np.array([0.0, -0.25*Ly, 0.0], dtype=np.float32)
    ankle_R    = knee_R    + np.array([0.0, -0.25*Ly, 0.0], dtype=np.float32)

    names = [
        "root", "spine1", "spine2", "neck", "head",
        "L_shoulder", "L_elbow", "L_wrist",
        "R_shoulder", "R_elbow", "R_wrist",
        "L_hip", "L_knee", "L_ankle",
        "R_hip", "R_knee", "R_ankle",
    ]
    parents = np.array([
        -1, 0, 1, 2, 3,
        2, 5, 6,
        2, 8, 9,
        1,11,12,
        1,14,15,
    ], dtype=np.int32)

    pos = np.stack([
        root, hips, chest, neck, head,
        shoulder_L, elbow_L, wrist_L,
        shoulder_R, elbow_R, wrist_R,
        hip_L, knee_L, ankle_L,
        hip_R, knee_R, ankle_R
    ], axis=0).astype(np.float32)

    # map back if up_axis == "z"
    if up_axis.lower() == "z":
        pos = pos[:, [0,2,1]]

    return names, parents, pos


# -----------------------
# Minimal usage example (comment)
# -----------------------
"""
# Example: build skeleton from bbox of spot_control_mesh.obj (not provided here)
from rigging.mesh_io import load_mesh
from rigging.skeleton import Skeleton, quadruped_auto_place_from_bbox

mesh = load_mesh("data/single/spot_control_mesh.obj", center=False, scale_to_unit=False)
aabb_min, aabb_max = mesh.aabb
names, parents, pos = quadruped_auto_place_from_bbox(aabb_min, aabb_max, up_axis="y")
skel = Skeleton.from_bind_positions(names, parents, pos)

# Identity pose -> skinning matrices
pose = skel.make_identity_pose()
M_skin = skel.skinning_matrices(pose)   # (J,4,4)
# Then LBS: V' = sum_j( w_ij * (M_skin[j] @ V_bind_h) )
"""
