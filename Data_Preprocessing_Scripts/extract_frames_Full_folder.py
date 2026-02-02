import os
import cv2
from tqdm import tqdm

# ========================
# Paths (edit if needed)
# ========================

VIDEO_DIR = "/home/mkhondka/face_swap/CanonSwap/results"
OUTPUT_DIR = "/home/mkhondka/Desktop/data_analysis/vdo_to_images/frames_output"

# ========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

video_files = sorted([
    f for f in os.listdir(VIDEO_DIR)
    if f.lower().endswith(".mp4")
])

print(f"Found {len(video_files)} videos.")

for video_name in tqdm(video_files, desc="Processing videos"):
    video_path = os.path.join(VIDEO_DIR, video_name)

    # folder name = video name without extension
    video_id = os.path.splitext(video_name)[0]
    out_dir = os.path.join(OUTPUT_DIR, video_id)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARNING] Cannot open video: {video_name}")
        continue

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # filename: frame_000001.jpg
        frame_name = f"frame_{frame_idx:06d}.jpg"
        frame_path = os.path.join(out_dir, frame_name)

        cv2.imwrite(frame_path, frame)
        saved += 1

    cap.release()
    # Optional: print per-video stats
    # print(f"{video_name}: saved {saved} frames")

print("\nDone. Frames saved in:", OUTPUT_DIR)

