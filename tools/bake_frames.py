# -*- coding: utf-8 -*-
"""
tools/bake_frames.py

用途：
- 将一个 Scene 在给定时间段内离屏渲染为一系列 PNG 帧；
- 依赖 render.offscreen_vtk.OffscreenVtkRenderer 和 scene.choreography.Scene。

典型用法（Python 内部调用）::

    from scene.choreography import Scene
    from tools.bake_frames import bake_scene_frames

    scene = build_my_scene()
    bake_scene_frames(scene, out_dir="out/frames/head_shake", duration=2.0, fps=30)

命令行用法（可选）::

    python -m tools.bake_frames --scene-module examples.head_shake_demo \\
        --scene-func build_scene --out out/frames/head_shake --duration 1.5 --fps 30
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
from typing import Callable, Optional, Tuple

import numpy as np

from scene.choreography import Scene
from render.offscreen_vtk import OffscreenVtkRenderer


RENDER_YAW_DEG = -90.0


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _rotation_y_deg(deg: float) -> np.ndarray:
    rad = np.radians(float(deg))
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )

def bake_scene_frames(
    scene: Scene,
    out_dir: str,
    duration: float,
    fps: int = 30,
    width: int = 1024,
    height: int = 1024,
    debug: bool = False,
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

    renderer = OffscreenVtkRenderer(width=width, height=height)
    render_rot = _rotation_y_deg(RENDER_YAW_DEG)
    renderer_ready = False

    print(f"🎬 开始烘焙帧: {total_frames} 帧, 时长 {duration:.3f}s, fps={fps}")
    print(f"    输出目录: {out_dir}")
    if debug:
        print(f"    Renderer: OffscreenVtkRenderer (UI pipeline)")
        print(f"    Render size: {width}x{height}")
        print(f"    Scene assets: {len(scene.assets)}")
        for asset in scene.assets:
            weights = getattr(asset, "weights", None)
            if weights is None:
                weights_info = "None"
            else:
                shape = getattr(weights, "shape", None)
                weights_info = f"{type(weights).__name__} {shape}"
            skel = getattr(asset, "skeleton", None)
            joint_count = skel.n if skel is not None else 0
            print(
                f"    - Asset '{asset.name}': verts={asset.mesh.vertices.shape[0]}, "
                f"joints={joint_count}, weights={weights_info}"
            )

    if debug and scene.timelines:
        for idx, timeline in enumerate(scene.timelines):
            print(
                f"    Timeline[{idx}]: duration={timeline.duration:.3f} "
                f"tracks={len(timeline.tracks)}"
            )
            for joint_name, track in timeline.tracks.items():
                kf_times = [float(kf.time) for kf in track.keyframes]
                print(f"      - {joint_name}: keys={len(kf_times)} times={kf_times}")
    try:
        for idx, t in enumerate(times):
            frames = scene.simulate(t)
            if not frames:
                print(f"[WARN] t={t:.3f} 没有可渲染的 AssetFrame，跳过。")
                continue
    
            if debug:
                for asset_frame in frames:
                    base = asset_frame.asset.mesh.vertices
                    diff_vec = asset_frame.vertices - base
                    delta = float(np.max(np.abs(diff_vec)))
                    mean_delta = float(np.mean(np.linalg.norm(diff_vec, axis=1)))
                    bbox_min = np.min(asset_frame.vertices, axis=0)
                    bbox_max = np.max(asset_frame.vertices, axis=0)
                    eye = np.eye(4, dtype=np.float32)
                    diff = np.linalg.norm(asset_frame.joint_transforms - eye, axis=(1, 2))
                    non_identity = int(np.sum(diff > 1e-6))
                    print(
                        f"  [DEBUG] t={t:.3f}s asset='{asset_frame.asset.name}' "
                        f"non_identity_joints={non_identity} max_vertex_delta={delta:.6f} "
                        f"mean_vertex_delta={mean_delta:.6f}"
                    )
                    print(f"    [DEBUG] bbox_min={bbox_min} bbox_max={bbox_max}")
                    skel = asset_frame.asset.skeleton
                    g_bind = None
                    g_curr = None
                    if skel is not None:
                        bind_locals = [j.bind_local for j in skel.joints]
                        g_bind = skel.forward_kinematics_local(bind_locals)
                        pose_list = [asset_frame.joint_transforms[j] for j in range(skel.n)]
                        g_curr = skel.forward_kinematics_pose(pose_list)
                    if non_identity:
                        for j in np.where(diff > 1e-6)[0][:4]:
                            mat = asset_frame.joint_transforms[j]
                            tvec = mat[:3, 3]
                            r = mat[:3, :3]
                            tr = float(np.trace(r))
                            cos_theta = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
                            angle = float(np.degrees(np.arccos(cos_theta)))
                            print(
                                f"    [DEBUG] joint[{j}] local_delta_t={tvec} "
                                f"delta_angle_deg={angle:.2f}"
                            )
                            if g_bind is not None and g_curr is not None:
                                bind_pos = g_bind[j][:3, 3]
                                curr_pos = g_curr[j][:3, 3]
                                delta_pos = curr_pos - bind_pos
                                name = skel.joints[j].name
                                print(
                                    f"    [DEBUG] joint[{j}] '{name}' "
                                    f"bind_pos={bind_pos} curr_pos={curr_pos} "
                                    f"delta_pos={delta_pos}"
                                )
                    active = np.where(diff > 1e-6)[0]
                    if active.size and hasattr(asset_frame.asset, "weights"):
                        weights = asset_frame.asset.weights
                        for j in active[:6]:
                            name = skel.joints[j].name if skel is not None else str(j)
                            max_w = 0.0
                            count = 0
                            if isinstance(weights, np.ndarray) and weights.ndim == 2:
                                wj = weights[:, j]
                                max_w = float(np.max(wj))
                                count = int(np.sum(wj > 1e-3))
                            elif hasattr(weights, "indices") and hasattr(weights, "weights"):
                                idxs = weights.indices
                                vals = weights.weights
                                mask = idxs == j
                                if np.any(mask):
                                    max_w = float(np.max(vals[mask]))
                                    count = int(np.sum(mask))
                            elif isinstance(weights, tuple) and len(weights) == 2:
                                idxs = np.asarray(weights[0])
                                vals = np.asarray(weights[1])
                                mask = idxs == j
                                if np.any(mask):
                                    max_w = float(np.max(vals[mask]))
                                    count = int(np.sum(mask))
                            print(
                                f"    [DEBUG] joint[{j}] '{name}' max_w={max_w:.4f} "
                                f"influenced_verts={count}"
                            )
    
    
            # Render first asset with VTK (UI pipeline)
            primary = frames[0]
            faces = primary.asset.mesh.faces.astype("int32")
            if not renderer_ready:
                renderer.set_mesh(primary.vertices, faces, rotation=render_rot)
                renderer_ready = True
                if len(frames) > 1:
                    print("[WARN] OffscreenVtkRenderer renders the first asset only.")
            else:
                renderer.update_vertices(primary.vertices)
    
            out_path = os.path.join(out_dir, f"frame_{idx:04d}.png")
            renderer.render_to_file(out_path)
            if idx % 10 == 0 or idx == total_frames - 1:
                print(f"  ▶ [{idx+1}/{total_frames}] t={t:.3f}s -> {out_path}")
    
    finally:
        renderer.close()
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
        help="包含构建场景函数的模块名，例如 examples.head_shake_demo",
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
        help="帧输出目录，例如 out/frames/head_shake",
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug logs (skeleton/weights/deformation)",
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
        debug=bool(args.debug),
    )


if __name__ == "__main__":
    main()
