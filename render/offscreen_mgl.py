# -*- coding: utf-8 -*-
"""
render/offscreen_mgl.py

用途：
- 使用 moderngl 进行离屏渲染，将三角网格保存为 PNG。
- API 尽量简单，只暴露一个 `OffscreenRenderer.render_mesh(...)`。

注意：
- 这是一个“可运行骨架”，你可以在之后补充：
  - 纹理 / 法线 / 阴影 / 多光源等；
  - 多个 mesh 的同时渲染；
  - 高级相机控制（轨迹球、FOV 等）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import moderngl  # type: ignore
except Exception:  # noqa: BLE001
    moderngl = None

from PIL import Image


@dataclass
class Camera:
    """极简摄像机参数."""

    eye: Tuple[float, float, float] = (2.5, 2.5, 2.5)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_deg: float = 45.0
    near: float = 0.01
    far: float = 100.0


class OffscreenRenderer:
    """使用 moderngl 进行离屏渲染的简单封装."""

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        camera: Optional[Camera] = None,
        background: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        if moderngl is None:
            raise ImportError(
                "OffscreenRenderer 需要 moderngl，请先安装：`pip install moderngl`"
            )

        self.width = int(width)
        self.height = int(height)
        self.camera = camera or Camera()
        self.background = background

        # 创建离屏上下文
        self.ctx = moderngl.create_standalone_context()

        # 创建 FBO
        self.fbo = self.ctx.simple_framebuffer((self.width, self.height))
        self.fbo.use()
        self.fbo.clear(*self.background)

        # 基本的 shader（顶点 + 片元）
        self._prog = self._create_basic_program()
        # 默认光照参数可以写在 shader 内部常量中，后续再抽出来

    # ------------------------------------------------------------------
    # 初始化/辅助
    # ------------------------------------------------------------------
    def _create_basic_program(self):
        """创建一个极简的 Phong-ish shader."""
        vertex_shader = """
        #version 330

        uniform mat4 u_mvp;

        in vec3 in_position;
        in vec3 in_normal;

        out vec3 v_normal;
        out vec3 v_position;

        void main() {
            vec4 world_pos = vec4(in_position, 1.0);
            gl_Position = u_mvp * world_pos;
            v_position = in_position;
            v_normal = normalize(in_normal);
        }
        """

        fragment_shader = """
        #version 330

        in vec3 v_normal;
        in vec3 v_position;

        out vec4 f_color;

        void main() {
            vec3 N = normalize(v_normal);
            vec3 L = normalize(vec3(0.5, 1.0, 0.8));  // 简单单点光源方向
            float diff = max(dot(N, L), 0.0);

            vec3 base = vec3(0.7, 0.7, 0.8);          // 基础颜色
            vec3 color = base * (0.2 + 0.8 * diff);   // 简单漫反射+环境光

            f_color = vec4(color, 1.0);
        }
        """

        return self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
        )

    # ------------------------------------------------------------------
    # 数学相关
    # ------------------------------------------------------------------
    @staticmethod
    def _look_at(
        eye: Tuple[float, float, float],
        target: Tuple[float, float, float],
        up: Tuple[float, float, float],
    ) -> np.ndarray:
        """生成视图矩阵 (4x4)."""
        eye_v = np.array(eye, dtype=np.float32)
        target_v = np.array(target, dtype=np.float32)
        up_v = np.array(up, dtype=np.float32)

        z = eye_v - target_v
        z /= np.linalg.norm(z) + 1e-8

        x = np.cross(up_v, z)
        x /= np.linalg.norm(x) + 1e-8

        y = np.cross(z, x)

        view = np.eye(4, dtype=np.float32)
        view[0, :3] = x
        view[1, :3] = y
        view[2, :3] = z
        view[:3, 3] = -eye_v @ np.array([x, y, z])
        return view

    @staticmethod
    def _perspective(
        fov_deg: float, aspect: float, near: float, far: float
    ) -> np.ndarray:
        """生成透视投影矩阵 (4x4)."""
        f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0
        return proj

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------
    def _draw_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        normals: Optional[np.ndarray] = None,
        model_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """Draw a single mesh into the current framebuffer."""
        v = np.asarray(vertices, dtype=np.float32)
        f = np.asarray(faces, dtype=np.int32)
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"vertices shape should be (N,3), got {v.shape}")
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"faces shape should be (F,3), got {f.shape}")

        if normals is None:
            normals = self._compute_vertex_normals(v, f)
        n = np.asarray(normals, dtype=np.float32)
        if n.shape != v.shape:
            raise ValueError(f"normals shape should match vertices: {v.shape}, got {n.shape}")

        model = np.eye(4, dtype=np.float32) if model_matrix is None else np.asarray(
            model_matrix, dtype=np.float32
        )

        view = self._look_at(self.camera.eye, self.camera.target, self.camera.up)
        proj = self._perspective(
            self.camera.fov_deg,
            aspect=float(self.width) / float(self.height),
            near=self.camera.near,
            far=self.camera.far,
        )
        mvp = proj @ view @ model

        # 组装 buffer
        # 顶点 + 法线 按顺序interleave
        vbo_data = np.hstack([v, n]).astype("f4").tobytes()
        vbo = self.ctx.buffer(vbo_data)
        ibo = self.ctx.buffer(f.astype("i4").tobytes())

        vao_content = [
            (vbo, "3f 3f", "in_position", "in_normal"),
        ]
        vao = self.ctx.vertex_array(self._prog, vao_content, ibo)

        # 设置 uniform
        self._prog["u_mvp"].write(mvp.tobytes())

        # 绘制
        vao.render()

    def render_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        out_path: str,
        normals: Optional[np.ndarray] = None,
        model_matrix: Optional[np.ndarray] = None,
    ) -> None:
        """Render a single mesh and save to PNG."""
        self.fbo.use()
        self.fbo.clear(*self.background)
        self._draw_mesh(vertices, faces, normals=normals, model_matrix=model_matrix)

        data = self.fbo.read(components=4, dtype="f1")
        img = Image.frombytes("RGBA", (self.width, self.height), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        img.save(out_path)

    def render_meshes(
        self,
        meshes: list[tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]],
        out_path: str,
    ) -> None:
        """Render multiple meshes into one PNG.

        meshes: [(vertices, faces, normals, model_matrix), ...]
        """
        self.fbo.use()
        self.fbo.clear(*self.background)
        for vertices, faces, normals, model_matrix in meshes:
            self._draw_mesh(vertices, faces, normals=normals, model_matrix=model_matrix)

        data = self.fbo.read(components=4, dtype="f1")
        img = Image.frombytes("RGBA", (self.width, self.height), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        img.save(out_path)

    @staticmethod
    def _compute_vertex_normals(
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> np.ndarray:
        """CPU 上简单计算顶点法线：按面法线平均."""
        v = vertices
        f = faces
        normals = np.zeros_like(v, dtype=np.float32)

        # 计算每个 face 的法线
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        # 防止零向量
        lengths = np.linalg.norm(fn, axis=1) + 1e-8
        fn /= lengths[:, None]

        # 累加到每个顶点
        for i in range(3):
            np.add.at(normals, f[:, i], fn)

        # 归一化
        lengths = np.linalg.norm(normals, axis=1) + 1e-8
        normals /= lengths[:, None]
        return normals
