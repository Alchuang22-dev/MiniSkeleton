# -*- coding: utf-8 -*-
"""
tools/bake_frames.py

用途：
- 将一个 Scene 在给定时间段内离屏渲染为一系列 PNG 帧；
- 依赖 render.offscreen_mgl.OffscreenRenderer 和 scene.choreography.Scene。

典型用法（Python 内部调用）::

    from scene.choreography import Scene
    from tools.bake_frames import bake_scene_frames

    scene = build_my_scene()
    bake_scene_frames(scene, out_dir="out/frames/spot", duration=2.0, fps=30)

命令行用法（可选）::

    python -m tools.bake_frames --scene-module examples.single_model_demo \\
        --scene-func build_scene --out out/frames/spot --duration 2.0 --fps 30
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
from typing import Callable, Optional, Tuple

import numpy as np

from scene.choreography import Scene
from render.offscreen_mgl import OffscreenRenderer


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def bake_scene_frames(
    scene: Scene,
    out_dir: str,
    duration: float,
    fps: int = 30,
    width: int = 1024,
    height: int = 1024,
) -> None:
    """对 Scene 进行采样 + 离屏渲染.

    Parameters
    ----------
    scene:
        已构建好的 Scene 对象（包含 RiggedAsset + Timeline）。
    out_dir:
        帧输出目录，会自动创建。文件名为 frame_0000.png, frame_0001.png, ...
    duration:
        动画时长（秒）。
    fps:
        采样帧率。
    width, height:
        输出图像的分辨率。
    """
    _ensure_dir(out_dir)

    total_frames = max(1, int(math.ceil(duration * fps)))
    times = np.linspace(0.0, duration, total_frames, endpoint=False)

    renderer = OffscreenRenderer(width=width, height=height)

    print(f"🎬 开始烘焙帧: {total_frames} 帧, 时长 {duration:.3f}s, fps={fps}")
    print(f"    输出目录: {out_dir}")
    for idx, t in enumerate(times):
        frames = scene.simulate(t)
        if not frames:
            print(f"[WARN] t={t:.3f} 没有可渲染的 AssetFrame，跳过。")
            continue

        # 当前实现：仅渲染第一个 asset，可根据需要扩展为多模型渲染。
        asset_frame = frames[0]
        vertices = asset_frame.vertices
        faces = asset_frame.asset.mesh.faces.astype("int32")

        out_path = os.path.join(out_dir, f"frame_{idx:04d}.png")
        renderer.render_mesh(vertices=vertices, faces=faces, out_path=out_path)
        if idx % 10 == 0 or idx == total_frames - 1:
            print(f"  ▶ [{idx+1}/{total_frames}] t={t:.3f}s -> {out_path}")

    print("✅ 帧烘焙完成。")


def _load_scene_from_entrypoint(
    module_name: str,
    func_name: str,
) -> Tuple[Scene, float]:
    """从 module:function 入口构建 Scene.

    约定：
    - 函数签名为 `def build_scene() -> Scene | (Scene, float)`；
    - 若只返回 Scene，则 duration 由最大 timeline.duration 推断。
    """
    module = importlib.import_module(module_name)
    func: Callable[..., object] = getattr(module, func_name)

    result = func()
    if isinstance(result, Scene):
        scene = result
        # 由 timelines 推断总时长
        if scene.timelines:
            duration = max(t.duration for t in scene.timelines)
        else:
            duration = 1.0
    else:
        scene, duration = result  # type: ignore[misc]

    if not isinstance(scene, Scene):
        raise TypeError(
            f"入口函数 {module_name}.{func_name} 返回值类型错误: {type(scene)!r}"
        )

    return scene, float(duration)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="对 Scene 进行离屏渲染, 输出 PNG 帧序列")
    parser.add_argument(
        "--scene-module",
        type=str,
        required=True,
        help="包含构建场景函数的模块名，例如 examples.single_model_demo",
    )
    parser.add_argument(
        "--scene-func",
        type=str,
        default="build_scene",
        help="构建 Scene 的函数名，默认 build_scene",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="帧输出目录，例如 out/frames/spot",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="动画时长（秒）。若不指定则从 Scene 的 timelines 推断。",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="采样帧率，默认 30 FPS",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="输出图像宽度，默认 1024",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="输出图像高度，默认 1024",
    )

    args = parser.parse_args(argv)

    scene, inferred_duration = _load_scene_from_entrypoint(
        args.scene_module, args.scene_func
    )
    duration = float(args.duration) if args.duration is not None else inferred_duration

    bake_scene_frames(
        scene=scene,
        out_dir=args.out,
        duration=duration,
        fps=int(args.fps),
        width=int(args.width),
        height=int(args.height),
    )


if __name__ == "__main__":
    main()
