#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np
import cv2
from tqdm import tqdm

# -------------------------
# Utilities
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_image_files(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for n in os.listdir(folder):
        if n.lower().endswith(exts):
            files.append(os.path.join(folder, n))
    files.sort()
    return files

def uniform_subsample(items, k):
    if k <= 0 or k >= len(items):
        return items
    idxs = np.linspace(0, len(items) - 1, k)
    idxs = np.round(idxs).astype(int)
    return [items[i] for i in idxs]

def bbox_area(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def touches_border(b, w, h, eps=2):
    x1, y1, x2, y2 = b
    return (x1 <= eps) or (y1 <= eps) or (x2 >= (w - 1 - eps)) or (y2 >= (h - 1 - eps))

def var_laplacian(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def face_sharpness_from_crop(face_crop):
    """Sharpness on central patch to reduce background defocus impact."""
    if face_crop is None or face_crop.size == 0:
        return 0.0
    ch, cw = face_crop.shape[:2]
    px1, px2 = int(0.2 * cw), int(0.8 * cw)
    py1, py2 = int(0.2 * ch), int(0.8 * ch)
    patch = face_crop[py1:py2, px1:px2]
    if patch.size == 0:
        patch = face_crop
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return var_laplacian(gray)

def _to_np_68(lmk):
    if lmk is None:
        return None
    pts = np.asarray(lmk, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] != 68 or pts.shape[1] < 2:
        return None
    return pts[:, :2].copy()

# -------------------------
# EAR / MAR using standard 68-landmark indexing
# Right eye: 36-41, Left eye: 42-47
# Mouth (outer): 48-59, inner: 60-67
# -------------------------
def _dist(a, b):
    return float(np.linalg.norm(a - b))

def eye_aspect_ratio(eye_pts_6):
    # eye_pts_6 order: p1..p6 around eye
    # EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    p1, p2, p3, p4, p5, p6 = eye_pts_6
    A = _dist(p2, p6)
    B = _dist(p3, p5)
    C = _dist(p1, p4)
    if C < 1e-6:
        return 0.0
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth_outer_12, mouth_inner_8):
    """
    MAR (one common variant):
    use inner mouth vertical distances normalized by mouth width (outer corners).
    """
    # Outer corners: 48 and 54
    left_corner = mouth_outer_12[0]   # idx 48
    right_corner = mouth_outer_12[6]  # idx 54
    mouth_w = _dist(left_corner, right_corner)
    if mouth_w < 1e-6:
        return 0.0

    # Inner mouth verticals (61-67 region):
    # Use (62,66) and (63,65) roughly central vertical gaps
    # mouth_inner indices in this slice: 60..67 -> positions 0..7
    p62 = mouth_inner_8[2]  # landmark 62
    p66 = mouth_inner_8[6]  # landmark 66
    p63 = mouth_inner_8[3]  # landmark 63
    p65 = mouth_inner_8[5]  # landmark 65
    v1 = _dist(p62, p66)
    v2 = _dist(p63, p65)
    return (v1 + v2) / (2.0 * mouth_w)

def parse_args():
    ap = argparse.ArgumentParser("Select best frame per folder (InsightFace only; EAR+MAR).")
    ap.add_argument("--frames_root", required=True, help="Root folder containing per-video frame folders")
    ap.add_argument("--output_dir", required=True, help="Output folder to save best frames")
    ap.add_argument("--min_det_score", type=float, default=0.55)
    ap.add_argument("--max_images_per_folder", type=int, default=80, help="0=all frames")
    ap.add_argument("--det_size", type=int, default=640)
    ap.add_argument("--save_bbox_vis", action="store_true")

    # EAR / MAR thresholds (tuneable)
    ap.add_argument("--ear_thr", type=float, default=0.20,
                    help="Penalize if EAR < this (closed eyes). Typical 0.18-0.23.")
    ap.add_argument("--ear_weight", type=float, default=12.0,
                    help="Strength of eye-closed penalty (increase if still selecting closed eyes).")

    ap.add_argument("--mar_thr", type=float, default=0.35,
                    help="Penalize if MAR > this (wide open mouth). Typical 0.30-0.45.")
    ap.add_argument("--mar_weight", type=float, default=10.0,
                    help="Strength of open-mouth penalty (increase if still selecting open mouth).")

    # Optional: hard reject very bad frames (keeps ranking stable)
    ap.add_argument("--hard_reject", action="store_true",
                    help="If set, skip frames with EAR very low or MAR very high.")
    ap.add_argument("--ear_reject", type=float, default=0.14)
    ap.add_argument("--mar_reject", type=float, default=0.60)

    return ap.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))

    video_ids = [d for d in os.listdir(args.frames_root)
                 if os.path.isdir(os.path.join(args.frames_root, d))]
    video_ids.sort()

    report_path = os.path.join(args.output_dir, "best_frames_report.csv")
    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_id", "best_image_name",
            "det_score", "face_area_ratio", "sharpness", "border_touch",
            "EAR", "MAR", "score", "saved_path"
        ])

        for vid in tqdm(video_ids, desc="Selecting best frames"):
            folder = os.path.join(args.frames_root, vid)
            images = list_image_files(folder)
            if not images:
                writer.writerow([vid, "", "", "", "", "", "", "", "", ""])
                continue

            images_eval = uniform_subsample(images, args.max_images_per_folder)

            best_score = -1e18
            best_path = None
            best_frame = None
            best_bbox_int = None
            best_vals = None

            for img_path in images_eval:
                frame = cv2.imread(img_path)
                if frame is None:
                    continue
                h, w = frame.shape[:2]

                faces = app.get(frame)
                faces = [fc for fc in faces if getattr(fc, "det_score", 0.0) >= args.min_det_score]
                if not faces:
                    continue

                faces.sort(key=lambda fc: bbox_area(fc.bbox.tolist()), reverse=True)
                fc = faces[0]

                det_score = float(getattr(fc, "det_score", 0.0))
                bbox = fc.bbox.tolist()

                # clamp bbox
                x1, y1, x2, y2 = bbox
                x1 = max(0.0, min(x1, w - 1))
                x2 = max(0.0, min(x2, w - 1))
                y1 = max(0.0, min(y1, h - 1))
                y2 = max(0.0, min(y2, h - 1))
                bbox = [x1, y1, x2, y2]

                area_ratio = bbox_area(bbox) / float(w * h)
                border = 1 if touches_border(bbox, w, h, eps=2) else 0

                xi1, yi1, xi2, yi2 = [int(v) for v in bbox]
                xi1 = max(0, min(xi1, w - 2))
                xi2 = max(1, min(xi2, w - 1))
                yi1 = max(0, min(yi1, h - 2))
                yi2 = max(1, min(yi2, h - 1))

                crop = frame[yi1:yi2, xi1:xi2]
                sharp = face_sharpness_from_crop(crop)

                # 68 landmarks from InsightFace
                lmk68 = getattr(fc, "landmark_3d_68", None)
                pts68 = _to_np_68(lmk68)
                if pts68 is None:
                    # If you ever see this a lot, we can fallback to another landmark attribute.
                    continue

                # EAR
                right_eye = pts68[36:42]  # 6 pts
                left_eye = pts68[42:48]
                ear_r = eye_aspect_ratio(right_eye)
                ear_l = eye_aspect_ratio(left_eye)
                ear = min(ear_r, ear_l)  # conservative

                # MAR
                mouth_outer = pts68[48:60]  # 12 pts
                mouth_inner = pts68[60:68]  # 8 pts
                mar = mouth_aspect_ratio(mouth_outer, mouth_inner)

                if args.hard_reject:
                    if ear < args.ear_reject:
                        continue
                    if mar > args.mar_reject:
                        continue

                # penalties
                eye_pen = max(0.0, args.ear_thr - ear)      # closed eyes => big penalty
                mouth_pen = max(0.0, mar - args.mar_thr)    # open mouth => big penalty

                # Score
                score = (
                    (2.0 * det_score) +
                    (3.0 * area_ratio) +
                    (0.002 * sharp) -
                    (0.25 * border) -
                    (args.ear_weight * eye_pen) -
                    (args.mar_weight * mouth_pen)
                )

                if score > best_score:
                    best_score = score
                    best_path = img_path
                    best_frame = frame
                    best_bbox_int = (xi1, yi1, xi2, yi2)
                    best_vals = (det_score, area_ratio, sharp, border, ear, mar, score)

            out_sub = os.path.join(args.output_dir, vid)
            ensure_dir(out_sub)

            if best_path is None:
                writer.writerow([vid, "", "", "", "", "", "", "", "", ""])
                continue

            det_score, area_ratio, sharp, border, ear, mar, score = best_vals
            best_name = os.path.basename(best_path)

            save_path = os.path.join(out_sub, f"{vid}_best.jpg")
            cv2.imwrite(save_path, best_frame)

            if args.save_bbox_vis and best_bbox_int is not None:
                vis = best_frame.copy()
                x1, y1, x2, y2 = best_bbox_int
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"det={det_score:.2f} area={area_ratio:.3f} sharp={sharp:.1f} EAR={ear:.3f} MAR={mar:.3f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                cv2.imwrite(os.path.join(out_sub, f"{vid}_best_vis.jpg"), vis)

            writer.writerow([
                vid, best_name,
                f"{det_score:.4f}", f"{area_ratio:.6f}", f"{sharp:.2f}", border,
                f"{ear:.4f}", f"{mar:.4f}", f"{score:.6f}", save_path
            ])

    print(f"\nDone. Report: {report_path}")
    print(f"Best frames saved under: {args.output_dir}")

if __name__ == "__main__":
    main()

