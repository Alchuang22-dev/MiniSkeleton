# -*- coding: utf-8 -*-
import numpy as np
from .mesh_io import Mesh, compute_vertex_adjacency
from .skeleton import Skeleton

def compute_heat_weights(mesh: Mesh, skel: Skeleton, lambda_reg: float = 1e-5) -> np.ndarray:
    """
    1.3 进阶：基于热扩散/拉普拉斯的权重求解（Heat Method / Biharmonic 等变体）。
    TODO:
      - 拉普拉斯构建
      - 每个关节作为源，解泊松/热方程
      - 归一化与平滑
    """
    raise NotImplementedError
