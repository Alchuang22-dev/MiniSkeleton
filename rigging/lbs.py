# -*- coding: utf-8 -*-
import numpy as np

def linear_blend_skinning(verts: np.ndarray, weights: np.ndarray, joint_mats: np.ndarray) -> np.ndarray:
    """
    1.4：LBS 蒙皮。输入：
      - verts: (N,3) 绑定姿态的顶点
      - weights: (N,J)
      - joint_mats: (J,4,4) 关节变换(含逆绑定校正)
    输出：
      - deformed_verts: (N,3)
    """
    raise NotImplementedError
