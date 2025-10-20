# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Keyframe:
    time: float
    values: dict  # {joint_name: transform/quat/euler/pos...}

@dataclass
class Track:
    joint_name: str
    keyframes: List[Keyframe] = field(default_factory=list)

@dataclass
class Timeline:
    duration: float
    tracks: Dict[str, Track] = field(default_factory=dict)

    def sample(self, t: float) -> dict:
        """插值出时间 t 的骨架姿态字典。"""
        raise NotImplementedError
