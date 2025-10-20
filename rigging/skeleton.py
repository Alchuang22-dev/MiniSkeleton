# -*- coding: utf-8 -*-
# 绑定骨架方面的函数
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional

@dataclass
class Joint:
    name: str
    parent: Optional[str]
    bind_pose: np.ndarray      # (4,4) 关节在绑定时的局部/或全局矩阵（按你的约定）
    offset_matrix: np.ndarray  # (4,4) 逆绑定矩阵（skin bind）

@dataclass
class Skeleton:
    joints: Dict[str, Joint] = field(default_factory=dict)
    hierarchy: Dict[str, List[str]] = field(default_factory=dict)  # parent -> [children]

    def add_joint(self, joint: Joint) -> None:
        """1.1：新增关节并维护层级关系。"""
        raise NotImplementedError

    def set_joint_transform(self, name: str, local_mat4: np.ndarray) -> None:
        """1.2：设置/编辑关节局部变换（UI 拖拽/数值输入）。"""
        raise NotImplementedError

    def forward_kinematics(self) -> Dict[str, np.ndarray]:
        """计算所有关节的全局矩阵（FK）。"""
        raise NotImplementedError

    def to_debug_lines(self) -> np.ndarray:
        """用于视口绘制骨架连线的线段顶点数组。"""
        raise NotImplementedError
