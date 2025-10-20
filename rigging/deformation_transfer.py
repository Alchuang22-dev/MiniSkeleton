# -*- coding: utf-8 -*-
import numpy as np

def deformation_transfer(source_bind: np.ndarray, target_bind: np.ndarray,
                         source_deformed: np.ndarray) -> np.ndarray:
    """
    3：（可选）Deformation Transfer 框架。
    TODO：
      - 三角形局部坐标/梯度场匹配
      - 通过最小二乘把 source 形变映射到 target
    """
    raise NotImplementedError
