# -*- coding: utf-8 -*-
"""
tools/export_video.py

用途：
- 将 out/frames/ 下的 PNG 序列合成为 mp4 或 gif；
- 底层调用 ffmpeg（二选一：直接 Popen，或调用 render/make_video.sh）。

基本用法::

    # 直接使用 Python API
    from tools.export_video import frames_to_video
    frames_to_video("out/frames/spot", "out/videos/spot.mp4", fps=30)

    # 命令行
    python -m tools.export_video --frames out/frames/spot --video out/videos/spot.mp4 --fps 30
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import Optional


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def frames_to_video(
    frames_dir: str,
    video_path: str,
    fps: int = 30,
    pattern: str = "frame_%04d.png",
    use_shell_script: bool = False,
) -> None:
    """将帧序列合成为视频或动图.

    Parameters
    ----------
    frames_dir:
        存放帧的目录，例如 out/frames/spot。
    video_path:
        输出视频路径，例如 out/videos/spot.mp4 或 .gif。
    fps:
        帧率。
    pattern:
        帧文件名模式，默认为 frame_%04d.png，对应 ffmpeg 的 -i 参数。
    use_shell_script:
        若为 True，则尝试调用 render/make_video.sh，否则直接调用 ffmpeg。
    """
    _ensure_dir(video_path)

    if use_shell_script:
        script = os.path.join(os.path.dirname(__file__), "..", "render", "make_video.sh")
        script = os.path.abspath(script)
        if not os.path.isfile(script):
            raise FileNotFoundError(f"找不到 make_video.sh: {script}")

        cmd = [
            script,
            frames_dir,
            video_path,
            str(fps),
            pattern,
        ]
    else:
        # 直接用 ffmpeg
        input_pattern = os.path.join(frames_dir, pattern)
        # 对于 mp4 使用 yuv420p 以保证兼容性
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            input_pattern,
        ]
        # 根据扩展名简单判断输出格式
        ext = os.path.splitext(video_path)[1].lower()
        if ext in {".mp4", ".mov"}:
            cmd += [
                "-pix_fmt",
                "yuv420p",
            ]
        elif ext in {".gif"}:
            cmd += [
                "-loop",
                "0",
            ]
        cmd.append(video_path)

    print("▶ 运行命令:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("✅ 视频导出完成:", video_path)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="将离屏渲染的帧序列导出为 mp4/gif")
    parser.add_argument(
        "--frames",
        type=str,
        required=True,
        help="帧目录，例如 out/frames/spot",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="输出视频路径，例如 out/videos/spot.mp4 或 .gif",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="视频帧率，默认 30 FPS",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="frame_%04d.png",
        help="帧文件名模式，默认为 frame_%04d.png",
    )
    parser.add_argument(
        "--use-shell-script",
        action="store_true",
        help="使用 render/make_video.sh 而非直接调用 ffmpeg",
    )

    args = parser.parse_args(argv)

    frames_to_video(
        frames_dir=args.frames,
        video_path=args.video,
        fps=int(args.fps),
        pattern=args.pattern,
        use_shell_script=bool(args.use_shell_script),
    )


if __name__ == "__main__":
    main()
