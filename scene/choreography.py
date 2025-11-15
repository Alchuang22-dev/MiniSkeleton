# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .asset import RiggedAsset
from .timeline import Timeline


@dataclass
class AssetFrame:
    """某一时间点下, 单个模型在场景中的状态."""

    asset: RiggedAsset
    vertices: np.ndarray          # 变形后的顶点 (N, 3)
    joint_transforms: np.ndarray  # 当前局部增量 pose (J, 4, 4)


@dataclass
class Scene:
    """简单的多模型场景/编排容器.

    当前实现:
    - 每个 RiggedAsset 绑定一条 Timeline
    - simulate(t) 时, 对所有 asset 采样 Timeline, 得到关节增量 pose
      -> 通过 Skeleton.skinning_matrices 计算蒙皮矩阵
      -> 调用 RiggedAsset.bake_frame() 得到变形后的顶点
    """

    fps: int = 30
    assets: List[RiggedAsset] = field(default_factory=list)
    timelines: List[Timeline] = field(default_factory=list)

    def add_asset(self, asset: RiggedAsset, timeline: Timeline) -> None:
        """多模型装配: 一个 asset 绑定一条 timeline.

        两个列表保持同长度 & 索引对齐。
        """
        if timeline is None:
            raise ValueError("Scene.add_asset() 需要传入有效的 Timeline.")
        if asset.weights is None:
            raise ValueError("Scene.add_asset() 之前需要为 asset 计算并设置权重(weights).")

        self.assets.append(asset)
        self.timelines.append(timeline)

    def simulate(self, t: float) -> List[AssetFrame]:
        """返回时刻 t 的所有模型姿态/顶点(供渲染).

        返回值为 AssetFrame 的列表, 调用方可以将顶点传入渲染管线。
        """
        frames: List[AssetFrame] = []

        for asset, timeline in zip(self.assets, self.timelines):
            if asset.skeleton is None:
                continue

            # 1) 采样时间线, 得到 {joint_name: delta_matrix}
            pose_dict = timeline.sample(t)

            # 2) 将字典映射到按关节索引排列的 (J,4,4) 增量矩阵数组
            J = asset.skeleton.n
            joint_transforms = np.zeros((J, 4, 4), dtype=np.float32)
            joint_transforms[:] = np.eye(4, dtype=np.float32)

            name_to_index = {j.name: idx for idx, j in enumerate(asset.skeleton.joints)}
            for joint_name, mat in pose_dict.items():
                idx = name_to_index.get(joint_name)
                if idx is None:
                    continue
                joint_transforms[idx] = np.asarray(mat, dtype=np.float32)

            # 3) 由 Skeleton 计算蒙皮矩阵, 再调用 LBS
            pose_list = [joint_transforms[j] for j in range(J)]
            skin_mats = asset.skeleton.skinning_matrices(pose_list)  # (J,4,4)
            deformed_vertices = asset.bake_frame(skin_mats)

            frames.append(
                AssetFrame(
                    asset=asset,
                    vertices=deformed_vertices,
                    joint_transforms=joint_transforms,
                )
            )

        return frames
