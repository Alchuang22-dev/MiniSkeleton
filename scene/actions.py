# -*- coding: utf-8 -*-
"""动作/姿态相关的辅助模块.

这个文件提供几类基础能力:

- ActionClip: 基于 Timeline 的动作片段(可循环)
- ActionState: 运行时状态(起始时间、权重、播放速度)
- blend_poses: 对多组姿态做加权混合
- sample_action_states: 在给定全局时间下, 对多个 ActionState 做采样+混合

这些工具可以在 UI 层被进一步封装, 用来实现:
- idle / walk / run 等动作的切换与混合
- 上半身/下半身分层动作(通过只对部分关节写入 keyframe)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np

from .timeline import Timeline


PoseDict = Dict[str, Any]  # {joint_name: transform(通常为 4x4 matrix)}


@dataclass
class ActionClip:
    """基于 Timeline 的可复用动作片段."""

    name: str
    timeline: Timeline
    loop: bool = True

    def sample(self, t_local: float) -> PoseDict:
        """在局部时间 t_local 下采样姿态.

        若 loop=True, 会自动在 [0, duration) 上取模。
        """
        if self.timeline.duration <= 0.0:
            return {}

        if self.loop:
            t_local = float(t_local % self.timeline.duration)
        else:
            t_local = max(0.0, min(float(t_local), float(self.timeline.duration)))

        return self.timeline.sample(t_local)


@dataclass
class ActionState:
    """运行时的动作状态.

    由 UI 或上层逻辑驱动, 用来描述某个动作片段当前的播放进度/权重。
    """

    clip: ActionClip
    start_time: float = 0.0   # 在全局时间轴上的起始时间
    weight: float = 1.0
    speed: float = 1.0        # 播放速度倍数

    def sample(self, global_time: float) -> PoseDict:
        """在给定的全局时间上, 采样该动作的姿态。"""
        local_t = (global_time - self.start_time) * self.speed
        if local_t < 0.0:
            # 尚未开始
            return {}
        return self.clip.sample(local_t)


def blend_poses(poses: List[PoseDict], weights: List[float]) -> PoseDict:
    """将多组姿态按权重做线性混合.

    这里假设:
    - 每个姿态中的 value 都可以被 np.asarray 转为数值数组
    - 若某个关节只出现在部分姿态中, 则仅对出现过的做加权和
    """
    if not poses:
        return {}

    if len(poses) != len(weights):
        raise ValueError("poses 和 weights 长度不一致。")

    weights_arr = np.asarray(weights, dtype=np.float32)
    if np.allclose(weights_arr, 0.0):
        # 避免除以 0, 直接返回第一组姿态
        return poses[0]

    weights_arr = weights_arr / weights_arr.sum()

    blended: Dict[str, np.ndarray] = {}
    for pose, w in zip(poses, weights_arr):
        if w <= 0.0:
            continue
        for joint_name, value in pose.items():
            v = np.asarray(value, dtype=np.float32)
            if joint_name not in blended:
                blended[joint_name] = w * v
            else:
                blended[joint_name] += w * v

    return blended


def sample_action_states(states: List[ActionState], global_time: float) -> PoseDict:
    """在给定全局时间下, 对多条动作状态进行采样并混合.

    通常的使用方式:
        pose = sample_action_states(active_states, now)
    然后再将 pose 映射为 (J,4,4) 数组, 送入 Skeleton.skinning_matrices。
    """
    if not states:
        return {}

    poses: List[PoseDict] = []
    weights: List[float] = []
    for state in states:
        pose = state.sample(global_time)
        if not pose:
            continue
        poses.append(pose)
        weights.append(state.weight)

    if not poses:
        return {}

    return blend_poses(poses, weights)
