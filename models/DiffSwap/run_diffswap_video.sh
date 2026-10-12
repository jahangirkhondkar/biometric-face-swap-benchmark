#!/bin/bash

set -e

SOURCE_IMG=$1
TARGET_VIDEO=$2
OUT_VIDEO=$3
FPS=${4:-30}
MAX_FRAMES=${5:-0}

cd ~/face_swap/DiffSwap

export PYTHONPATH=$PWD:$PYTHONPATH

rm -rf video_work
mkdir -p video_work/frames
mkdir -p video_work/results
mkdir -p outputs

# Extract frames as PNG (lossless / higher quality)
# If MAX_FRAMES > 0, process only that many frames.
# If MAX_FRAMES = 0, process the full video.
if [ "$MAX_FRAMES" -gt 0 ]; then
    ffmpeg -y -i "$TARGET_VIDEO" -frames:v "$MAX_FRAMES" -qscale:v 1 -qmin 1 -qmax 1 video_work/frames/frame_%05d.png
else
    ffmpeg -y -i "$TARGET_VIDEO" -qscale:v 1 -qmin 1 -qmax 1 video_work/frames/frame_%05d.png
fi

# Run DiffSwap frame by frame
for frame in video_work/frames/*.png; do
    name=$(basename "$frame" .png)

    rm -rf data/portrait data/portrait_jpg
    mkdir -p data/portrait_jpg/source
    mkdir -p data/portrait_jpg/target

    cp "$SOURCE_IMG" data/portrait_jpg/source/0.jpg
    cp "$frame" data/portrait_jpg/target/0.png

    python pipeline.py

    # Prefer final full-resolution pasted output
    if [ -f data/portrait/swap_res_ori/diffswap_0.01/0/0.png ]; then
        cp data/portrait/swap_res_ori/diffswap_0.01/0/0.png video_work/results/${name}.png
    else
        cp data/portrait/swap_res/diffswap_0.01/0/0.png video_work/results/${name}.png
    fi
done

# Rebuild video
ffmpeg -y -framerate "$FPS" \
  -i video_work/results/frame_%05d.png \
  -i "$TARGET_VIDEO" \
  -map 0:v:0 -map 1:a? \
  -c:v libx264 -pix_fmt yuv420p \
  -c:a aac -shortest "$OUT_VIDEO"
