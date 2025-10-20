# -*- coding: utf-8 -*-
from dataclasses import dataclass
import numpy as np
from rigging.mesh_io import Mesh
from rigging.skeleton import Skeleton

@dataclass
class RiggedAsset:
    name: str
    mesh: Mesh
    skeleton: Skeleton
    weights: np.ndarray | None = None  # (N,J)

    def bake_frame(self, joint_mats: np.ndarray) -> np.ndarray:
        """调用 LBS，返回该帧变形后的顶点。"""
        raise NotImplementedError
