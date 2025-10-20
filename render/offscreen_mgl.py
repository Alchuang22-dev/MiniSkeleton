# -*- coding: utf-8 -*-
import numpy as np

class OffscreenRenderer:
    """基于 moderngl 的离屏渲染，输出 PNG 帧。"""
    def __init__(self, width: int = 1280, height: int = 720):
        # TODO: 创建 context，编译着色器，VBO/IBO 管线
        pass

    def draw_mesh(self, vertices: np.ndarray, faces: np.ndarray, out_path: str) -> None:
        # TODO: 渲染到 FBO 并保存 PNG
        raise NotImplementedError
