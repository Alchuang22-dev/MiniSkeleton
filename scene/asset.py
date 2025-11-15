# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional

import numpy as np

from rigging.mesh_io import Mesh
from rigging.skeleton import Skeleton
from rigging.lbs import linear_blend_skinning


@dataclass
class RiggedAsset:
    """绑定了骨架和权重的可动画模型."""
    name: str
    mesh: Mesh
    skeleton: Skeleton
    # 形状为 (N, J) 的权重矩阵, N 为顶点数, J 为关节数
    weights: Optional[np.ndarray] = None  # (N, J)

    def bake_frame(self, joint_mats: np.ndarray) -> np.ndarray:
        """调用 LBS, 返回该帧变形后的顶点坐标.

        Parameters
        ----------
        joint_mats:
            形状为 (J, 4, 4) 的蒙皮矩阵(通常是由 Skeleton.skinning_matrices 计算得到),
            对应 ui_simple.py 里的 M_skin。
        """
        if self.weights is None:
            raise ValueError("RiggedAsset.bake_frame() 需要先设置 weights (N, J) 矩阵.")

        joint_mats = np.asarray(joint_mats, dtype=np.float32)
        if joint_mats.ndim != 3 or joint_mats.shape[1:] != (4, 4):
            raise ValueError(
                f"joint_mats 形状应为 (J, 4, 4), 实际为 {joint_mats.shape!r}"
            )

        # 和 ui_simple.py 中的逻辑保持一致:
        #   M_skin = skeleton.skinning_matrices(pose)
        #   deformed_vertices = linear_blend_skinning(..., M_skin, ...)
        deformed_vertices = linear_blend_skinning(
            self.mesh.vertices,
            self.weights,
            joint_mats,
            topk=None,
            normalize=False,
        )
        return deformed_vertices
