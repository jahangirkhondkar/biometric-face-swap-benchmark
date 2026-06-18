import os
import cv2
import argparse
import subprocess
from pathlib import Path

def detect_largest_face(img, detector):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    # largest detected face
    return max(faces, key=lambda b: b[2] * b[3])

def expand_box(x, y, w, h, img_w, img_h, scale=1.35):
    cx, cy = x + w / 2, y + h / 2
    size = int(max(w, h) * scale)

    x1 = int(max(0, cx - size / 2))
    y1 = int(max(0, cy - size / 2))
    x2 = int(min(img_w, cx + size / 2))
    y2 = int(min(img_h, cy + size / 2))

    return x1, y1, x2, y2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/source/000_best.jpg")
    parser.add_argument("--target_dir", default="data/target")
    parser.add_argument("--output_dir", default="output/full_frames")
    parser.add_argument("--weight", default="checkpoints/blendswap.pth")
    parser.add_argument("--temp_dir", default="temp")
    args = parser.parse_args()

    source_crop_dir = Path(args.temp_dir) / "source_crop"
    target_crop_dir = Path(args.temp_dir) / "target_crop"
    swapped_crop_dir = Path(args.temp_dir) / "swapped_crop"

    source_crop_dir.mkdir(parents=True, exist_ok=True)
    target_crop_dir.mkdir(parents=True, exist_ok=True)
    swapped_crop_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    detector_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(detector_path)

    # -------------------------
    # Prepare source crop once
    # -------------------------
    source_img = cv2.imread(args.source)
    if source_img is None:
        raise FileNotFoundError(f"Cannot read source image: {args.source}")

    src_face = detect_largest_face(source_img, detector)
    if src_face is None:
        raise RuntimeError("No face detected in source image.")

    h_src, w_src = source_img.shape[:2]
    x, y, w, h = src_face
    x1, y1, x2, y2 = expand_box(x, y, w, h, w_src, h_src, scale=1.35)

    source_crop = source_img[y1:y2, x1:x2]
    source_crop = cv2.resize(source_crop, (112, 112))

    source_crop_path = source_crop_dir / "source_112.png"
    cv2.imwrite(str(source_crop_path), source_crop)

    # -------------------------
    # Process target frames
    # -------------------------
    target_paths = sorted(
        list(Path(args.target_dir).glob("*.jpg")) +
        list(Path(args.target_dir).glob("*.png"))
    )

    print(f"Found {len(target_paths)} target frames.")

    for idx, target_path in enumerate(target_paths, 1):
        print(f"[{idx}/{len(target_paths)}] Processing {target_path.name}")

        frame = cv2.imread(str(target_path))
        if frame is None:
            print(f"  Skipping unreadable image: {target_path}")
            continue

        face = detect_largest_face(frame, detector)
        if face is None:
            print(f"  No face detected. Saving original frame.")
            cv2.imwrite(str(Path(args.output_dir) / target_path.name), frame)
            continue

        H, W = frame.shape[:2]
        x, y, w, h = face
        x1, y1, x2, y2 = expand_box(x, y, w, h, W, H, scale=1.35)

        target_crop = frame[y1:y2, x1:x2]
        crop_h, crop_w = target_crop.shape[:2]

        target_crop_256 = cv2.resize(target_crop, (256, 256))

        target_crop_path = target_crop_dir / f"{target_path.stem}_target.png"
        swapped_crop_path = swapped_crop_dir / f"{target_path.stem}_swapped.png"

        cv2.imwrite(str(target_crop_path), target_crop_256)

        # -------------------------
        # Call BlendFace inference.py
        # inference.py takes one source, one target, one output
        # -------------------------
        cmd = [
            "python",
            "inference.py",
            "-w", args.weight,
            "-s", str(source_crop_path),
            "-t", str(target_crop_path),
            "-o", str(swapped_crop_path),
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"  BlendFace failed for {target_path.name}")
            continue

        swapped = cv2.imread(str(swapped_crop_path))
        if swapped is None:
            print(f"  Cannot read swapped crop: {swapped_crop_path}")
            continue

        swapped = cv2.resize(swapped, (crop_w, crop_h))

        # Simple paste-back for first test
        output_frame = frame.copy()
        output_frame[y1:y2, x1:x2] = swapped

        out_path = Path(args.output_dir) / target_path.name
        cv2.imwrite(str(out_path), output_frame)

    print("Done.")

if __name__ == "__main__":
    main()
