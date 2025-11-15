# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np


@dataclass
class Keyframe:
    """单个关键帧.

    Parameters
    ----------
    time:
        时间(秒). 建议使用秒制, 由 fps 决定采样步长.
    values:
        对应的关节姿态字典, 形如 {joint_name: transform}.
        其中 transform 推荐使用 4x4 齐次矩阵(np.ndarray).
    """
    time: float
    values: Dict[str, Any]  # {joint_name: transform/quat/euler/pos...}


@dataclass
class Track:
    """一个关节的一条动画轨."""
    joint_name: str
    keyframes: List[Keyframe] = field(default_factory=list)

    def add_keyframe(self, time: float, value: Any) -> None:
        """向该轨道添加关键帧.

        这里的 value 会自动包装成 {joint_name: value}.
        """
        self.keyframes.append(Keyframe(time=time, values={self.joint_name: value}))
        # 保证时间有序, 便于后续插值
        self.keyframes.sort(key=lambda k: k.time)

    def sample(self, t: float) -> Any:
        """在时间 t 处采样该轨道的值.

        插值策略:
        - 若 t 落在首帧之前, 返回首帧值
        - 若 t 落在末帧之后, 返回末帧值
        - 否则在邻近两个关键帧之间做线性插值

        注意: 这里对任意可转为 np.ndarray 的数值进行线性插值。
        对于旋转等更复杂的插值(例如四元数 slerp)可以后续扩展。
        """
        if not self.keyframes:
            return None

        # clamp 到关键帧时间范围
        kfs = self.keyframes
        if t <= kfs[0].time:
            return kfs[0].values[self.joint_name]
        if t >= kfs[-1].time:
            return kfs[-1].values[self.joint_name]

        # 查找相邻关键帧
        prev_kf = kfs[0]
        next_kf = kfs[-1]
        for i in range(len(kfs) - 1):
            k0, k1 = kfs[i], kfs[i + 1]
            if k0.time <= t <= k1.time:
                prev_kf, next_kf = k0, k1
                break

        v0 = np.asarray(prev_kf.values[self.joint_name], dtype=np.float32)
        v1 = np.asarray(next_kf.values[self.joint_name], dtype=np.float32)

        if np.isclose(prev_kf.time, next_kf.time):
            return v0

        alpha = float((t - prev_kf.time) / (next_kf.time - prev_kf.time))
        return (1.0 - alpha) * v0 + alpha * v1


@dataclass
class Timeline:
    """一条完整的动画时间线, 由多条关节轨道组成."""
    duration: float
    tracks: Dict[str, Track] = field(default_factory=dict)

    def add_keyframe(self, joint_name: str, time: float, value: Any) -> None:
        """在指定关节上添加关键帧."""
        track = self.tracks.get(joint_name)
        if track is None:
            track = Track(joint_name=joint_name)
            self.tracks[joint_name] = track
        track.add_keyframe(time, value)

        # 更新 duration, 以末尾关键帧时间为准
        if time > self.duration:
            self.duration = time

    def sample(self, t: float) -> Dict[str, Any]:
        """插值出时间 t 的骨架姿态字典.

        返回结果形如 {joint_name: transform}.
        """
        if not self.tracks:
            return {}

        # 将 t 限制在 [0, duration] 范围内
        if self.duration > 0.0:
            t = max(0.0, min(float(t), float(self.duration)))

        pose: Dict[str, Any] = {}
        for joint_name, track in self.tracks.items():
            value = track.sample(t)
            if value is not None:
                pose[joint_name] = value
        return pose
