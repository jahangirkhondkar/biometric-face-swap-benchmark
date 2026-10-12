import cv2
import argparse
from pathlib import Path
from insightface.app import FaceAnalysis


def make_square_crop(bbox, frame_w, frame_h, scale=1.45):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    size = max(x2 - x1, y2 - y1) * scale

    sx1 = int(round(cx - size / 2))
    sy1 = int(round(cy - size / 2))
    sx2 = int(round(cx + size / 2))
    sy2 = int(round(cy + size / 2))

    sx1 = max(0, sx1)
    sy1 = max(0, sy1)
    sx2 = min(frame_w, sx2)
    sy2 = min(frame_h, sy2)

    crop_w = sx2 - sx1
    crop_h = sy2 - sy1
    side = max(crop_w, crop_h)

    sx2 = min(frame_w, sx1 + side)
    sy2 = min(frame_h, sy1 + side)

    sx1 = max(0, sx2 - side)
    sy1 = max(0, sy2 - side)

    return sx1, sy1, sx2, sy2


def transform_points(values, crop, out_size=512):
    sx1, sy1, sx2, sy2 = crop
    crop_w = sx2 - sx1
    crop_h = sy2 - sy1

    out = []
    for i, v in enumerate(values):
        if i % 2 == 0:
            out.append(int(round((v - sx1) * out_size / crop_w)))
        else:
            out.append(int(round((v - sy1) * out_size / crop_h)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out_video", required=True)
    parser.add_argument("--out_txt", required=True)
    parser.add_argument("--det_size", type=int, default=640)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--scale", type=float, default=1.45)
    args = parser.parse_args()

    out_video = Path(args.out_video)
    out_txt = Path(args.out_txt)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (args.crop_size, args.crop_size)
    )

    lines = []
    last_crop = None
    last_line = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        faces = app.get(frame)

        if len(faces) > 0:
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
            bbox = face.bbox.astype(float).tolist()
            kps = face.kps.astype(float).reshape(-1).tolist()
            crop = make_square_crop(bbox, w, h, args.scale)

            bbox_kps = bbox + kps
            transformed = transform_points(bbox_kps, crop, args.crop_size)
            line = ",".join(map(str, transformed))

            last_crop = crop
            last_line = line
        else:
            if last_crop is None or last_line is None:
                raise RuntimeError(f"No face detected in first frame: {frame_idx}")
            crop = last_crop
            line = last_line

        sx1, sy1, sx2, sy2 = crop
        cropped = frame[sy1:sy2, sx1:sx2]
        cropped = cv2.resize(cropped, (args.crop_size, args.crop_size))

        writer.write(cropped)
        lines.append(line)
        frame_idx += 1

    cap.release()
    writer.release()

    with open(out_txt, "w") as f:
        f.write("\n".join(lines))

    print("Saved video:", out_video)
    print("Saved txt:", out_txt)
    print("Frames/lines:", len(lines))


if __name__ == "__main__":
    main()
