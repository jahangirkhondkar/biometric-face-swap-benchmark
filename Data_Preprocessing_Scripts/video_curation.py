#!/usr/bin/env python3
import os
import cv2
import csv
import math
import shutil
import argparse
import numpy as np
from tqdm import tqdm

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_videos(input_dir: str):
    vids = []
    for name in os.listdir(input_dir):
        if name.lower().endswith(".mp4"):
            vids.append(os.path.join(input_dir, name))
    vids.sort()
    return vids

def sample_frame_indices(n_frames: int, k: int):
    if n_frames <= 0:
        return []
    if k <= 1:
        return [n_frames // 2]
    # evenly spaced indices across the video (avoid first/last frame)
    idxs = np.linspace(0.05*(n_frames-1), 0.95*(n_frames-1), k)
    idxs = np.clip(np.round(idxs).astype(int), 0, n_frames-1)
    return idxs.tolist()

def var_laplacian(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def bbox_area(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2-x1) * max(0.0, y2-y1)

def clamp_bbox(b, w, h):
    x1, y1, x2, y2 = b
    x1 = max(0.0, min(x1, w-1))
    x2 = max(0.0, min(x2, w-1))
    y1 = max(0.0, min(y1, h-1))
    y2 = max(0.0, min(y2, h-1))
    return [x1, y1, x2, y2]

def touches_border(b, w, h, eps=2):
    x1, y1, x2, y2 = b
    return (x1 <= eps) or (y1 <= eps) or (x2 >= (w-1-eps)) or (y2 >= (h-1-eps))

def eye_line_angle_deg(kps):
    """
    InsightFace returns 5 landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
    Use eye line tilt as weak proxy for roll. (Pose is harder without full headpose.)
    """
    if kps is None or len(kps) < 2:
        return None
    le = kps[0]; re = kps[1]
    dx = float(re[0] - le[0])
    dy = float(re[1] - le[1])
    if abs(dx) < 1e-6:
        return 90.0
    ang = math.degrees(math.atan2(dy, dx))
    return abs(ang)

def parse_args():
    ap = argparse.ArgumentParser(description="Video curation filter (FF++ YouTube RAW)")

    ap.add_argument("--input_dir", required=True, help="Folder containing .mp4")
    ap.add_argument("--output_dir", required=True, help="Output base folder")
    ap.add_argument("--frames_per_video", type=int, default=30)
    ap.add_argument("--min_det_score", type=float, default=0.55)

    # thresholds (tune after pilot)
    ap.add_argument("--min_face_area_ratio", type=float, default=0.03,
                    help="Reject if face bbox/frame area ratio too small in many frames")
    ap.add_argument("--blur_var_threshold", type=float, default=80.0,
                    help="Reject if Laplacian variance on face crop is below this often")
    ap.add_argument("--max_multiface_ratio", type=float, default=0.15)
    ap.add_argument("--max_smallface_ratio", type=float, default=0.40)
    ap.add_argument("--max_blur_ratio", type=float, default=0.50)
    ap.add_argument("--max_outofframe_ratio", type=float, default=0.20)
    ap.add_argument("--max_noface_ratio", type=float, default=0.60)

    # weak pose proxy (roll tilt). If you want stricter pose checks later, add mediapipe headpose.
    ap.add_argument("--max_roll_deg", type=float, default=25.0)
    ap.add_argument("--max_pose_ratio", type=float, default=0.40)

    ap.add_argument("--copy", action="store_true", help="Copy videos to pass/reject (default)")
    ap.add_argument("--move", action="store_true", help="Move videos to pass/reject")
    ap.add_argument("--limit", type=int, default=0, help="Process only N videos (0=all)")
    ap.add_argument("--debug_frames", action="store_true",
                    help="Save a few sampled frames with bbox overlays for rejected videos")

    return ap.parse_args()

def main():
    args = parse_args()

    if args.move and args.copy:
        raise SystemExit("Use only one of --copy or --move.")
    do_move = args.move
    do_copy = (not args.move)  # default safe mode

    ensure_dir(args.output_dir)
    pass_dir = os.path.join(args.output_dir, "pass")
    rej_dir  = os.path.join(args.output_dir, "reject")
    ensure_dir(pass_dir)
    ensure_dir(rej_dir)

    # reject subfolders
    reasons = ["multiface","blur","smallface","pose","outofframe","noface","other"]
    for r in reasons:
        ensure_dir(os.path.join(rej_dir, r))

    # report
    report_path = os.path.join(args.output_dir, "report.csv")

    # InsightFace detector
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    videos = list_videos(args.input_dir)
    if args.limit and args.limit > 0:
        videos = videos[:args.limit]

    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video","decision","primary_reason","all_reasons",
            "frames_sampled","frames_with_face","multiface_ratio",
            "smallface_ratio","blur_ratio","outofframe_ratio","noface_ratio",
            "median_face_area_ratio","median_blur_var"
        ])

        for vp in tqdm(videos, desc="Curating"):
            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                # can't read -> reject
                decision = "reject"
                primary_reason = "other"
                all_reasons = "unreadable"
                dst = os.path.join(rej_dir, "other", os.path.basename(vp))
                if do_move: shutil.move(vp, dst)
                else: shutil.copy2(vp, dst)
                writer.writerow([vp, decision, primary_reason, all_reasons, 0,0,1,1,1,1,1,0,0])
                continue

            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            idxs = sample_frame_indices(n_frames, args.frames_per_video)

            multiface_ct = 0
            smallface_ct = 0
            blur_ct = 0
            outofframe_ct = 0
            noface_ct = 0

            face_area_ratios = []
            blur_vars = []
            pose_bad_ct = 0
            sampled = 0
            frames_with_face = 0

            # for debug
            dbg_saved = 0
            dbg_dir = os.path.join(args.output_dir, "debug_frames", os.path.splitext(os.path.basename(vp))[0])

            for idx in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                sampled += 1
                h, w = frame.shape[:2]

                faces = app.get(frame)
                # filter by score
                faces = [fc for fc in faces if getattr(fc, "det_score", 0.0) >= args.min_det_score]

                if len(faces) == 0:
                    noface_ct += 1
                    continue

                frames_with_face += 1

                if len(faces) >= 2:
                    multiface_ct += 1

                # choose largest face
                faces_sorted = sorted(
                    faces,
                    key=lambda fc: bbox_area(fc.bbox.tolist()),
                    reverse=True
                )
                fc = faces_sorted[0]
                bbox = fc.bbox.tolist()  # x1,y1,x2,y2
                # clamp
                bbox = clamp_bbox(bbox, w, h)

                # face size ratio
                ratio = bbox_area(bbox) / float(w*h)
                face_area_ratios.append(ratio)
                if ratio < args.min_face_area_ratio:
                    smallface_ct += 1

                # out of frame
                if touches_border(bbox, w, h, eps=2):
                    outofframe_ct += 1

                                # blur (on face crop)
                x1,y1,x2,y2 = [int(v) for v in bbox]
                x1 = max(0, min(x1, w-2)); x2 = max(1, min(x2, w-1))
                y1 = max(0, min(y1, h-2)); y2 = max(1, min(y2, h-1))
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    ch, cw = crop.shape[:2]

                    # central patch (focus on face details, avoid background)
                    px1 = int(0.2 * cw)
                    px2 = int(0.8 * cw)
                    py1 = int(0.2 * ch)
                    py2 = int(0.8 * ch)

                    patch = crop[py1:py2, px1:px2]
                    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    bv = var_laplacian(gray)

                    blur_vars.append(bv)
                    if bv < args.blur_var_threshold:
                        blur_ct += 1

                # debug frames for rejected videos (we’ll decide later; just collect a few)
                if args.debug_frames and dbg_saved < 3:
                    # draw bbox and save
                    vis = frame.copy()
                    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(vis, f"ratio={ratio:.3f}", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                    ensure_dir(dbg_dir)
                    cv2.imwrite(os.path.join(dbg_dir, f"frame_{idx:06d}.jpg"), vis)
                    dbg_saved += 1

            cap.release()

            if sampled == 0:
                # couldn't sample -> reject
                decision = "reject"
                primary_reason = "other"
                all_reasons = "nosamples"
            else:
                multiface_ratio = multiface_ct / sampled
                smallface_ratio = smallface_ct / sampled
                blur_ratio = blur_ct / sampled
                outofframe_ratio = outofframe_ct / sampled
                noface_ratio = noface_ct / sampled
                pose_ratio = pose_bad_ct / sampled

                med_area = float(np.median(face_area_ratios)) if face_area_ratios else 0.0
                med_blur = float(np.median(blur_vars)) if blur_vars else 0.0

                fail_reasons = []
                if noface_ratio > args.max_noface_ratio:
                    fail_reasons.append("noface")
                if multiface_ratio > args.max_multiface_ratio:
                    fail_reasons.append("multiface")
                if smallface_ratio > args.max_smallface_ratio:
                    fail_reasons.append("smallface")
                if blur_ratio > args.max_blur_ratio:
                    fail_reasons.append("blur")
                if outofframe_ratio > args.max_outofframe_ratio:
                    fail_reasons.append("outofframe")
                if pose_ratio > args.max_pose_ratio:
                    fail_reasons.append("pose")

                if len(fail_reasons) == 0:
                    decision = "pass"
                    primary_reason = ""
                    all_reasons = ""
                else:
                    decision = "reject"
                    # choose primary reason by priority (you can reorder)
                    priority = ["noface","multiface","smallface","outofframe","blur","pose","other"]
                    primary_reason = next((p for p in priority if p in fail_reasons), fail_reasons[0])
                    all_reasons = "|".join(fail_reasons)

            # copy/move
            base = os.path.basename(vp)
            if decision == "pass":
                dst = os.path.join(pass_dir, base)
            else:
                sub = primary_reason if primary_reason in reasons else "other"
                dst = os.path.join(rej_dir, sub, base)

            if do_move:
                shutil.move(vp, dst)
            else:
                shutil.copy2(vp, dst)

            # write report row
            if sampled == 0:
                writer.writerow([vp, decision, primary_reason, all_reasons, 0,0,1,1,1,1,1,0,0])
            else:
                multiface_ratio = multiface_ct / sampled
                smallface_ratio = smallface_ct / sampled
                blur_ratio = blur_ct / sampled
                outofframe_ratio = outofframe_ct / sampled
                noface_ratio = noface_ct / sampled
                med_area = float(np.median(face_area_ratios)) if face_area_ratios else 0.0
                med_blur = float(np.median(blur_vars)) if blur_vars else 0.0
                writer.writerow([
                    vp, decision, primary_reason, all_reasons,
                    sampled, frames_with_face,
                    f"{multiface_ratio:.4f}", f"{smallface_ratio:.4f}",
                    f"{blur_ratio:.4f}", f"{outofframe_ratio:.4f}",
                    f"{noface_ratio:.4f}", f"{med_area:.6f}", f"{med_blur:.2f}"
                ])

    print(f"\nDone. Report saved to: {report_path}")
    print(f"Pass folder: {pass_dir}")
    print(f"Reject folder: {rej_dir}")
    if do_copy:
        print("Mode: COPY (safe). Use --move when you are confident.")
    else:
        print("Mode: MOVE.")

if __name__ == "__main__":
    main()

