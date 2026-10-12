import cv2
import argparse
from pathlib import Path
from insightface.app import FaceAnalysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Target video path")
    parser.add_argument("--output", required=True, help="Output txt path")
    parser.add_argument("--det_size", type=int, default=640)
    args = parser.parse_args()

    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    lines = []
    last_line = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        if len(faces) == 0:
            if last_line is None:
                raise RuntimeError(f"No face detected in first frame: {frame_idx}")
            lines.append(last_line)
            frame_idx += 1
            continue

        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        bbox = face.bbox.astype(int).tolist()
        kps = face.kps.astype(int).reshape(-1).tolist()

        values = bbox + kps
        if len(values) != 14:
            raise RuntimeError(f"Expected 14 values, got {len(values)} at frame {frame_idx}")

        line = ",".join(map(str, values))
        lines.append(line)
        last_line = line
        frame_idx += 1

    cap.release()

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved: {output_path}")
    print(f"Frames/lines: {len(lines)}")


if __name__ == "__main__":
    main()
