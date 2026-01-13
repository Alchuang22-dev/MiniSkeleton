# -*- coding: utf-8 -*-
"""Auto motion keyframe generators shared by UI demo buttons."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from rigging.skeleton import euler_xyz_to_rot


DEMO_FPS = 30
DEMO_DURATION = 1.5
DEMO_HEAD_YAW = 0.35
DEMO_NECK_YAW = 0.15

WALK_BASE_DURATION = 2.0
WALK_REPEAT = 5
WALK_LEG_SWING = 0.25


def _rotation_y(angle_rad: float) -> np.ndarray:
    return euler_xyz_to_rot(0.0, float(angle_rad), 0.0).astype(np.float32)


def _rotation_x(angle_rad: float) -> np.ndarray:
    return euler_xyz_to_rot(float(angle_rad), 0.0, 0.0).astype(np.float32)

def _rotation_z(angle_rad: float) -> np.ndarray:
    return euler_xyz_to_rot(0.0, 0.0, float(angle_rad)).astype(np.float32)


def _apply_local_rotation(base: np.ndarray, rot: np.ndarray) -> np.ndarray:
    out = np.array(base, copy=True)
    out[:3, :3] = rot @ out[:3, :3]
    return out


def pick_joint_index(joint_names: Iterable[str], preferred: list[str]) -> int | None:
    names = list(joint_names)
    if not names:
        return None
    lower_map = {name.lower(): idx for idx, name in enumerate(names)}
    for key in preferred:
        idx = lower_map.get(key.lower())
        if idx is not None:
            return idx
    for key in preferred:
        key_lower = key.lower()
        for idx, name in enumerate(names):
            if key_lower in name.lower():
                return idx
    return None


def build_head_shake_keyframes(
    joint_names: Iterable[str],
    base_transforms: np.ndarray,
    *,
    fps: int = DEMO_FPS,
    duration: float = DEMO_DURATION,
    head_yaw: float = DEMO_HEAD_YAW,
    neck_yaw: float = DEMO_NECK_YAW,
) -> list[np.ndarray]:
    names = list(joint_names)
    head_idx = pick_joint_index(names, ["head", "neck_mid", "neck_root", "neck"])
    neck_idx = pick_joint_index(names, ["neck_mid", "neck_root", "neck"])
    if head_idx is None and neck_idx is None:
        return []

    frame_count = int(max(1, round(duration * fps)))
    times = np.linspace(0.0, duration, frame_count, endpoint=False)
    keyframes: list[np.ndarray] = []
    base = np.array(base_transforms, copy=True)
    for t in times:
        phase = (t / duration) * (2.0 * np.pi)
        head_angle = float(np.sin(phase) * head_yaw)
        neck_angle = float(np.sin(phase) * neck_yaw)
        frame = np.array(base, copy=True)
        if head_idx is not None:
            frame[head_idx] = _apply_local_rotation(base[head_idx], _rotation_y(head_angle))
        if neck_idx is not None and neck_idx != head_idx:
            frame[neck_idx] = _apply_local_rotation(base[neck_idx], _rotation_y(neck_angle))
        keyframes.append(frame)
    return keyframes


def build_walk_keyframes(
    joint_names: Iterable[str],
    base_transforms: np.ndarray,
    *,
    fps: int,
    duration: float = WALK_BASE_DURATION,
    repeat: int = WALK_REPEAT,
    leg_swing: float = WALK_LEG_SWING,
    close_loop: bool = True,
) -> list[np.ndarray]:
    names = list(joint_names)
    leg_back_left = pick_joint_index(
        names,
        [
            "leg_back_L_hip",
            "leg_back_L_knee",
            "leg_back_L_ankle",
            "leg_back_l",
            "hind_left",
            "back_left",
            "leg1",
        ],
    )
    leg_back_right = pick_joint_index(
        names,
        [
            "leg_back_R_hip",
            "leg_back_R_knee",
            "leg_back_R_ankle",
            "leg_back_r",
            "hind_right",
            "back_right",
            "leg2",
        ],
    )
    leg_front_left = pick_joint_index(
        names,
        [
            "leg_front_L_hip",
            "leg_front_L_knee",
            "leg_front_L_ankle",
            "leg_front_l",
            "front_left",
            "leg3",
        ],
    )
    leg_front_right = pick_joint_index(
        names,
        [
            "leg_front_R_hip",
            "leg_front_R_knee",
            "leg_front_R_ankle",
            "leg_front_r",
            "front_right",
            "leg4",
        ],
    )

    if all(idx is None for idx in (leg_back_left, leg_back_right, leg_front_left, leg_front_right)):
        return []

    frame_count = int(max(1, round(duration * fps)))
    times = np.linspace(0.0, duration, frame_count, endpoint=False)
    base = np.array(base_transforms, copy=True)
    base_keyframes: list[np.ndarray] = []
    for t in times:
        phase = (t / duration) * (2.0 * np.pi)
        swing_a = float(np.sin(phase) * leg_swing)
        swing_b = float(np.sin(phase + np.pi) * leg_swing)
        frame = np.array(base, copy=True)
        # [修复2] 分离前后肢逻辑
        # 前肢：父级是 Root，通常 X 轴是前后摆动
        if leg_front_left is not None:
            frame[leg_front_left] = _apply_local_rotation(base[leg_front_left], _rotation_x(swing_a))
        if leg_front_right is not None:
            frame[leg_front_right] = _apply_local_rotation(base[leg_front_right], _rotation_x(swing_b))
            
        # 后肢：父级是 Spine，如果 X 轴导致横移，尝试改用 Z 轴
        # 注意：这里假设 Z 轴是修正方向。如果变成"自转/扭曲"，请改回 _rotation_x 并尝试 _rotation_y
        if leg_back_right is not None:
            frame[leg_back_right] = _apply_local_rotation(base[leg_back_right], _rotation_z(swing_a))
        if leg_back_left is not None:
            frame[leg_back_left] = _apply_local_rotation(base[leg_back_left], _rotation_z(swing_b))
            
        base_keyframes.append(frame)

    keyframes: list[np.ndarray] = []
    repeat = max(1, int(repeat))
    for _ in range(repeat):
        for frame in base_keyframes:
            keyframes.append(np.array(frame, copy=True))

    if close_loop and base_keyframes:
        keyframes.append(np.array(base_keyframes[0], copy=True))
    return keyframes
