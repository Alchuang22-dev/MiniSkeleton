# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List
from .asset import RiggedAsset
from .timeline import Timeline

@dataclass
class Scene:
    fps: int = 30
    assets: List[RiggedAsset] = field(default_factory=list)
    timelines: List[Timeline] = field(default_factory=list)

    def add_asset(self, asset: RiggedAsset, timeline: Timeline) -> None:
        """多模型装配：一个 asset 绑定一条 timeline。"""
        raise NotImplementedError

    def simulate(self, t: float):
        """返回时刻 t 的所有模型姿态/顶点（供渲染）。"""
        raise NotImplementedError
