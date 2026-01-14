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
WALK_BACK_PHASE_OFFSET = 0.35
WALK_KNEE_LIFT = 0.18
WALK_ANKLE_LIFT = 0.12
WALK_SPINE_SWAY = 0.05
WALK_NECK_PITCH = 0.08
WALK_HEAD_PITCH = 0.12


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

    leg_back_left_hip = pick_joint_index(
        names,
        ["leg_back_L_hip", "leg_back_l_hip", "hind_left_hip", "back_left_hip", "leg1_hip"],
    )
    leg_back_left_knee = pick_joint_index(
        names,
        ["leg_back_L_knee", "leg_back_l_knee", "hind_left_knee", "back_left_knee", "leg1_knee"],
    )
    leg_back_left_ankle = pick_joint_index(
        names,
        ["leg_back_L_ankle", "leg_back_l_ankle", "hind_left_ankle", "back_left_ankle", "leg1_ankle"],
    )
    leg_back_right_hip = pick_joint_index(
        names,
        ["leg_back_R_hip", "leg_back_r_hip", "hind_right_hip", "back_right_hip", "leg2_hip"],
    )
    leg_back_right_knee = pick_joint_index(
        names,
        ["leg_back_R_knee", "leg_back_r_knee", "hind_right_knee", "back_right_knee", "leg2_knee"],
    )
    leg_back_right_ankle = pick_joint_index(
        names,
        ["leg_back_R_ankle", "leg_back_r_ankle", "hind_right_ankle", "back_right_ankle", "leg2_ankle"],
    )
    leg_front_left_hip = pick_joint_index(
        names,
        ["leg_front_L_hip", "leg_front_l_hip", "front_left_hip", "leg3_hip"],
    )
    leg_front_left_knee = pick_joint_index(
        names,
        ["leg_front_L_knee", "leg_front_l_knee", "front_left_knee", "leg3_knee"],
    )
    leg_front_left_ankle = pick_joint_index(
        names,
        ["leg_front_L_ankle", "leg_front_l_ankle", "front_left_ankle", "leg3_ankle"],
    )
    leg_front_right_hip = pick_joint_index(
        names,
        ["leg_front_R_hip", "leg_front_r_hip", "front_right_hip", "leg4_hip"],
    )
    leg_front_right_knee = pick_joint_index(
        names,
        ["leg_front_R_knee", "leg_front_r_knee", "front_right_knee", "leg4_knee"],
    )
    leg_front_right_ankle = pick_joint_index(
        names,
        ["leg_front_R_ankle", "leg_front_r_ankle", "front_right_ankle", "leg4_ankle"],
    )

    neck_idx = pick_joint_index(names, ["neck_mid", "neck_root", "neck"])
    head_idx = pick_joint_index(names, ["head"])
    spine_idx = pick_joint_index(names, ["body_spine", "spine", "body_root", "body"])

    if all(
        idx is None
        for idx in (
            leg_back_left_hip,
            leg_back_right_hip,
            leg_front_left_hip,
            leg_front_right_hip,
        )
    ):
        return []

    frame_count = int(max(1, round(duration * fps)))
    times = np.linspace(0.0, duration, frame_count, endpoint=False)
    base = np.array(base_transforms, copy=True)
    base_keyframes: list[np.ndarray] = []
    for t in times:
        phase = (t / duration) * (2.0 * np.pi)
        back_phase = phase + WALK_BACK_PHASE_OFFSET
        swing_front = float(np.sin(phase))
        swing_front_opp = float(np.sin(phase + np.pi))
        swing_back = float(np.sin(back_phase))
        swing_back_opp = float(np.sin(back_phase + np.pi))
        lift_front = max(0.0, float(np.sin(phase)))
        lift_front_opp = max(0.0, float(np.sin(phase + np.pi)))
        lift_back = max(0.0, float(np.sin(back_phase)))
        lift_back_opp = max(0.0, float(np.sin(back_phase + np.pi)))
        frame = np.array(base, copy=True)

        if spine_idx is not None:
            frame[spine_idx] = _apply_local_rotation(
                base[spine_idx], _rotation_y(WALK_SPINE_SWAY * np.sin(phase))
            )
        if neck_idx is not None:
            frame[neck_idx] = _apply_local_rotation(
                base[neck_idx], _rotation_x(WALK_NECK_PITCH * np.sin(phase + np.pi / 2.0))
            )
        if head_idx is not None:
            frame[head_idx] = _apply_local_rotation(
                base[head_idx], _rotation_x(WALK_HEAD_PITCH * np.sin(phase + np.pi / 2.0))
            )

        if leg_front_left_hip is not None:
            frame[leg_front_left_hip] = _apply_local_rotation(
                base[leg_front_left_hip], _rotation_x(swing_front * leg_swing)
            )
        if leg_front_left_knee is not None:
            frame[leg_front_left_knee] = _apply_local_rotation(
                base[leg_front_left_knee], _rotation_x(lift_front * WALK_KNEE_LIFT)
            )
        if leg_front_left_ankle is not None:
            frame[leg_front_left_ankle] = _apply_local_rotation(
                base[leg_front_left_ankle], _rotation_x(lift_front * WALK_ANKLE_LIFT)
            )
        if leg_front_right_hip is not None:
            frame[leg_front_right_hip] = _apply_local_rotation(
                base[leg_front_right_hip], _rotation_x(swing_front_opp * leg_swing)
            )
        if leg_front_right_knee is not None:
            frame[leg_front_right_knee] = _apply_local_rotation(
                base[leg_front_right_knee], _rotation_x(lift_front_opp * WALK_KNEE_LIFT)
            )
        if leg_front_right_ankle is not None:
            frame[leg_front_right_ankle] = _apply_local_rotation(
                base[leg_front_right_ankle], _rotation_x(lift_front_opp * WALK_ANKLE_LIFT)
            )

        if leg_back_right_hip is not None:
            frame[leg_back_right_hip] = _apply_local_rotation(
                base[leg_back_right_hip], _rotation_z(swing_back * leg_swing)
            )
        if leg_back_right_knee is not None:
            frame[leg_back_right_knee] = _apply_local_rotation(
                base[leg_back_right_knee], _rotation_z(lift_back * WALK_KNEE_LIFT)
            )
        if leg_back_right_ankle is not None:
            frame[leg_back_right_ankle] = _apply_local_rotation(
                base[leg_back_right_ankle], _rotation_z(lift_back * WALK_ANKLE_LIFT)
            )
        if leg_back_left_hip is not None:
            frame[leg_back_left_hip] = _apply_local_rotation(
                base[leg_back_left_hip], _rotation_z(swing_back_opp * leg_swing)
            )
        if leg_back_left_knee is not None:
            frame[leg_back_left_knee] = _apply_local_rotation(
                base[leg_back_left_knee], _rotation_z(lift_back_opp * WALK_KNEE_LIFT)
            )
        if leg_back_left_ankle is not None:
            frame[leg_back_left_ankle] = _apply_local_rotation(
                base[leg_back_left_ankle], _rotation_z(lift_back_opp * WALK_ANKLE_LIFT)
            )

        base_keyframes.append(frame)

    keyframes: list[np.ndarray] = []
    repeat = max(1, int(repeat))
    for _ in range(repeat):
        for frame in base_keyframes:
            keyframes.append(np.array(frame, copy=True))

    if close_loop and base_keyframes:
        keyframes.append(np.array(base_keyframes[0], copy=True))
    return keyframes
