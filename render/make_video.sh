#!/usr/bin/env bash
# 用 ffmpeg 将帧序列合成为视频或 gif。
#
# 调用方式（与 tools/export_video.py 中保持一致）：
#   ./render/make_video.sh <frames_dir> <video_path> <fps> <pattern>
#
# 示例：
#   ./render/make_video.sh out/frames/spot out/videos/spot.mp4 30 "frame_%04d.png"

set -e

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <frames_dir> <video_path> <fps> [pattern]"
  exit 1
fi

FRAMES_DIR="$1"
VIDEO_PATH="$2"
FPS="$3"
PATTERN="${4:-frame_%04d.png}"

if [ ! -d "$FRAMES_DIR" ]; then
  echo "[ERROR] frames_dir not found: $FRAMES_DIR"
  exit 1
fi

mkdir -p "$(dirname "$VIDEO_PATH")"

INPUT_PATTERN="${FRAMES_DIR}/${PATTERN}"
EXT="${VIDEO_PATH##*.}"
EXT_LOWER="$(echo "$EXT" | tr '[:upper:]' '[:lower:]')"

CMD=(ffmpeg -y -framerate "$FPS" -i "$INPUT_PATTERN")

case "$EXT_LOWER" in
  mp4|mov)
    CMD+=(-pix_fmt yuv420p)
    ;;
  gif)
    CMD+=(-loop 0)
    ;;
  *)
    # 其他格式按 ffmpeg 默认处理
    ;;
esac

CMD+=("$VIDEO_PATH")

echo "Running:" "${CMD[@]}"
"${CMD[@]}"

echo "Done: $VIDEO_PATH"
