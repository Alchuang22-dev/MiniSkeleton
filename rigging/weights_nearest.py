# -*- coding: utf-8 -*-
import numpy as np
from .skeleton import Skeleton

def compute_nearest_bone_weights(verts: np.ndarray, skel: Skeleton, k: int = 2) -> np.ndarray:
    """
    1.3 基础：最近骨骼/双骨插值权重。
    返回 (N_vertices, N_joints) 的稀疏/稠密权重矩阵（可先返回稠密，后续换稀疏）。
    TODO:
      - 定义骨段采样或关节点距离度量
      - 归一化，阈值裁剪
    """
    raise NotImplementedError
