# -*- coding: utf-8 -*-
"""
render/onscreen_qt.py

用途：
- 提供一个与 PySide6 配合的“渲染桥接层”，对 UI 层隐藏 moderngl 细节；
- 供 ui/viewport.py 使用，用于在窗口中实时渲染骨骼+网格。

这里主要提供两个类：
- SceneRenderer: 一个纯 Python 的“渲染控制器”，不依赖 Qt；
- QtViewportBridge: 将 SceneRenderer 嵌入到 QOpenGLWidget 里。

注意：
- 这是一个骨架，默认只绘制“当前一帧的单个网格”，渲染逻辑可以根据需要扩展。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import moderngl  # type: ignore
except Exception:  # noqa: BLE001
    moderngl = None

from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from scene.choreography import Scene


@dataclass
class ViewportCamera:
    """在 UI 中使用的摄像机."""

    eye: Tuple[float, float, float] = (2.5, 2.5, 2.5)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    fov_deg: float = 45.0
    near: float = 0.01
    far: float = 100.0


class SceneRenderer:
    """与具体 UI 框架无关的“实时渲染控制器”.

    - 负责创建 moderngl.Context, 编译 shader 等；
    - 提供 draw(time, width, height) 接口供 QtViewportBridge 调用。
    """

    def __init__(self, camera: Optional[ViewportCamera] = None) -> None:
        if moderngl is None:
            raise ImportError(
                "SceneRenderer 需要 moderngl，请先安装：`pip install moderngl`"
            )

        self.camera = camera or ViewportCamera()
        self.ctx: Optional["moderngl.Context"] = None
        self.prog = None

        self._vbo = None
        self._ibo = None
        self._vao = None
        self._vbo_array = None
        self._vertex_capacity = 0
        self._index_capacity = 0

        self.scene: Optional[Scene] = None
        self.current_time: float = 0.0

    # ------------------------------------------------------------------
    # 初始化 GL 资源
    # ------------------------------------------------------------------
    def init_gl(self, ctx) -> None:
        """由 QtViewportBridge 创建好 context 后调用。"""
        self.ctx = ctx
        self.prog = self._create_basic_program()

    def _create_basic_program(self):
        """和 OffscreenRenderer 保持类似风格的简单 shader。"""
        vs = """
        #version 330

        uniform mat4 u_mvp;

        in vec3 in_position;
        in vec3 in_normal;

        out vec3 v_normal;

        void main() {
            gl_Position = u_mvp * vec4(in_position, 1.0);
            v_normal = normalize(in_normal);
        }
        """
        fs = """
        #version 330

        in vec3 v_normal;
        out vec4 f_color;

        void main() {
            vec3 N = normalize(v_normal);
            vec3 L = normalize(vec3(0.5, 1.0, 0.8));
            float diff = max(dot(N, L), 0.0);
            vec3 base = vec3(0.7, 0.7, 0.8);
            vec3 color = base * (0.2 + 0.8 * diff);
            f_color = vec4(color, 1.0);
        }
        """
        return self.ctx.program(vertex_shader=vs, fragment_shader=fs)

    # ------------------------------------------------------------------
    # 绑定场景
    # ------------------------------------------------------------------
    def set_scene(self, scene: Scene) -> None:
        """设置当前要播放的 Scene."""
        self.scene = scene

    # ------------------------------------------------------------------
    # 数学：视图/投影矩阵
    # ------------------------------------------------------------------
    @staticmethod
    def _look_at(eye, target, up) -> np.ndarray:
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
    def _perspective(fov_deg, aspect, near, far) -> np.ndarray:
        f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2 * far * near) / (near - far)
        proj[3, 2] = -1.0
        return proj

    # ------------------------------------------------------------------
    # 渲染主入口
    # ------------------------------------------------------------------
    def draw(self, time: float, width: int, height: int) -> None:
        """在当前 moderngl.Context 中绘制一帧.

        - 由 QtViewportBridge 在 paintGL 中调用；
        - 默认只渲染 scene 中的第一个 asset，后续可扩展为多 mesh。
        """
        if self.ctx is None or self.prog is None:
            return

        self.current_time = time
        self.ctx.viewport = (0, 0, width, height)
        self.ctx.clear(1.0, 1.0, 1.0, 1.0)

        if self.scene is None:
            return

        frames = self.scene.simulate(time)
        if not frames:
            return

        frame = frames[0]
        vertices = frame.vertices
        faces = frame.asset.mesh.faces.astype(np.int32)

        normals = self._compute_vertex_normals(vertices, faces)

        # 组装/复用 buffer
        v = np.asarray(vertices, dtype=np.float32)
        n = np.asarray(normals, dtype=np.float32)
        v_count = v.shape[0]
        idx_count = int(faces.size)

        if self._vbo is None or self._vertex_capacity < v_count:
            self._vertex_capacity = v_count
            self._vbo_array = np.empty((v_count, 6), dtype=np.float32)
            self._vbo = self.ctx.buffer(reserve=self._vbo_array.nbytes)
            self._vao = None
        elif self._vbo_array is None or self._vbo_array.shape[0] != v_count:
            self._vbo_array = np.empty((v_count, 6), dtype=np.float32)

        faces_bytes = faces.astype("i4").tobytes()
        if self._ibo is None or self._index_capacity != idx_count:
            self._index_capacity = idx_count
            self._ibo = self.ctx.buffer(reserve=len(faces_bytes))
            self._ibo.write(faces_bytes)
            self._vao = None
        else:
            self._ibo.write(faces_bytes)

        self._vbo_array[:, :3] = v
        self._vbo_array[:, 3:] = n
        self._vbo.write(self._vbo_array.tobytes())

        if self._vao is None:
            vao_content = [(self._vbo, "3f 3f", "in_position", "in_normal")]
            self._vao = self.ctx.vertex_array(self.prog, vao_content, self._ibo)

        # 计算 MVP
        cam = self.camera
        view = self._look_at(cam.eye, cam.target, cam.up)
        proj = self._perspective(
            cam.fov_deg, aspect=float(width) / float(height or 1), near=cam.near, far=cam.far
        )
        mvp = proj @ view  # 这里暂时不考虑模型矩阵

        self.prog["u_mvp"].write(mvp.astype("f4").tobytes())
        self._vao.render()

    @staticmethod
    def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        v = np.asarray(vertices, dtype=np.float32)
        f = np.asarray(faces, dtype=np.int32)
        normals = np.zeros_like(v, dtype=np.float32)

        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(fn, axis=1) + 1e-8
        fn /= lengths[:, None]

        for i in range(3):
            np.add.at(normals, f[:, i], fn)

        lengths = np.linalg.norm(normals, axis=1) + 1e-8
        normals /= lengths[:, None]
        return normals


class QtViewportBridge(QOpenGLWidget):
    """嵌入 UI 的 OpenGL 视口.

    使用方式（在 ui/viewport.py 中）大致为：

        class SpotViewport(QtViewportBridge):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.set_scene(my_scene)

    然后在 UI 主窗口中将该 widget 放到布局里即可。
    """

    def __init__(self, parent=None) -> None:
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)

        self.renderer = SceneRenderer()
        self._time: float = 0.0  # 可由外部驱动（比如定时器）

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def set_scene(self, scene: Scene) -> None:
        self.renderer.set_scene(scene)
        self.update()

    def set_time(self, t: float) -> None:
        self._time = float(t)
        self.update()

    # ------------------------------------------------------------------
    # QOpenGLWidget 生命周期
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        if moderngl is None:
            return

        # 使用 Qt 的 context 创建 moderngl.Context
        self.ctx = moderngl.create_context()
        self.renderer.init_gl(self.ctx)

    def resizeGL(self, w: int, h: int) -> None:
        _ = (w, h)  # 暂时不用，renderer.draw 会直接根据传入的宽高设置 viewport

    def paintGL(self) -> None:
        if moderngl is None:
            return

        w = max(1, self.width())
        h = max(1, self.height())
        self.renderer.draw(time=self._time, width=w, height=h)
