# -*- coding: utf-8 -*-
import subprocess
from pathlib import Path

def frames_to_video(frames_dir: str, out_path: str, fps: int = 30) -> None:
    """
    使用 ffmpeg 把 frames_dir 下的 png 合成为 mp4/gif。
    注意：要求文件名连续，如 frame_000001.png
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", f"{frames_dir}/frame_%06d.png",
        "-pix_fmt", "yuv420p",
        out_path
    ]
    subprocess.check_call(cmd)
