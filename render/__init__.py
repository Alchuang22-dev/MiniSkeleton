# -*- coding: utf-8 -*-
"""
render 包：负责所有“如何从几何数据渲染出图像”的逻辑。

当前提供：
- OffscreenRenderer: 使用 moderngl 做离屏渲染，输出 PNG。
- QtViewportBridge: 与 PySide6 UI 进行桥接的基础类（在 onscreen_qt 中）。

上层只需要关心：
- 离屏批量渲染时，用 OffscreenRenderer(render.offscreen_mgl)；
- UI 实时预览时，用 QtViewportBridge(render.onscreen_qt) 或其子类。
"""

from .offscreen_mgl import OffscreenRenderer  # noqa: F401