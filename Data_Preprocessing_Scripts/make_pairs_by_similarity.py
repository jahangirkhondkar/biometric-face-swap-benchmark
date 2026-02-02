#!/usr/bin/env python3
import os
import csv
import argparse
import shutil
from typing import List, Tuple, Dict

import numpy as np
import cv2
from tqdm import tqdm


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def find_best_images(best_frames_root: str) -> List[Tuple[str, str]]:
    """
    Returns list of (video_id, image_path) for *_best.jpg inside each video_id folder.
    """
    items = []
    for vid in sorted(os.listdir(best_frames_root)):
        vdir = os.path.join(best_frames_root, vid)
        if not os.path.isdir(vdir):
            continue
        # expected: <vid>_best.jpg
        cand = os.path.join(vdir, f"{vid}_best.jpg")
        if os.path.isfile(cand):
            items.append((vid, cand))
        else:
            # fallback: any *_best.jpg in the folder
            for fn in os.listdir(vdir):
                if fn.endswith("_best.jpg"):
                    items.append((vid, os.path.join(vdir, fn)))
                    break
    return items


def bbox_area(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x)
    if n < eps:
        return x
    return x / n


def parse_args():
    ap = argparse.ArgumentParser("Make disjoint pairs of best frames by ArcFace cosine similarity.")
    ap.add_argument("--best_frames_root", required=True,
                    help="Root dir containing per-video folders with <vid>_best.jpg")
    ap.add_argument("--output_dir", required=True, help="Where to save pair folders + csv")

    ap.add_argument("--det_size", type=int, default=640)
    ap.add_argument("--min_det_score", type=float, default=0.55)

    ap.add_argument("--max_pairs", type=int, default=0,
                    help="0 = make as many disjoint pairs as possible")
    ap.add_argument("--topk", type=int, default=50,
                    help="For each image, only consider top-K neighbors (speed). 0=all (slow).")

    ap.add_argument("--allow_reuse", action="store_true",
                    help="If set, an image can appear in multiple pairs (NOT recommended).")

    ap.add_argument("--copy_mode", choices=["copy", "symlink"], default="copy",
                    help="How to place images into pair folders")
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    # --- Load InsightFace ---
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))

    # --- Gather images ---
    items = find_best_images(args.best_frames_root)
    if not items:
        raise SystemExit(f"No best images found under: {args.best_frames_root}")

    print(f"Found {len(items)} best images.")

    # --- Extract embeddings ---
    vids: List[str] = []
    paths: List[str] = []
    embs: List[np.ndarray] = []
    det_scores: List[float] = []

    bad = 0
    for vid, path in tqdm(items, desc="Extracting embeddings"):
        img = cv2.imread(path)
        if img is None:
            bad += 1
            continue

        faces = app.get(img)
        faces = [fc for fc in faces if getattr(fc, "det_score", 0.0) >= args.min_det_score]
        if not faces:
            bad += 1
            continue

        # choose largest face
        faces.sort(key=lambda fc: bbox_area(fc.bbox.tolist()), reverse=True)
        fc = faces[0]

        emb = getattr(fc, "embedding", None)
        if emb is None:
            bad += 1
            continue

        emb = l2_normalize(np.asarray(emb, dtype=np.float32))
        vids.append(vid)
        paths.append(path)
        embs.append(emb)
        det_scores.append(float(getattr(fc, "det_score", 0.0)))

    if len(embs) < 2:
        raise SystemExit("Not enough valid embeddings to form pairs.")

    if bad > 0:
        print(f"Warning: skipped {bad} images (read/detection/embedding issues). Using {len(embs)}.")

    E = np.vstack(embs)  # (N, D)
    N = E.shape[0]

    # --- Cosine similarity matrix ---
    # since embeddings are L2-normalized, cosine sim = dot product
    S = E @ E.T
    np.fill_diagonal(S, -1.0)

    # --- Build candidate edges ---
    # Option A: consider all edges (i<j)
    # Option B (recommended): top-K neighbors per i
    candidates: List[Tuple[float, int, int]] = []

    if args.topk and args.topk > 0:
        k = min(args.topk, N - 1)
        for i in range(N):
            # partial top-k (descending)
            idx = np.argpartition(-S[i], k)[:k]
            # keep i<j to avoid duplicates
            for j in idx:
                if i < j:
                    candidates.append((float(S[i, j]), i, int(j)))
    else:
        for i in range(N):
            for j in range(i + 1, N):
                candidates.append((float(S[i, j]), i, j))

    # sort edges by similarity descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    # --- Greedy matching ---
    used = np.zeros(N, dtype=bool)
    pairs: List[Tuple[int, int, float]] = []

    for sim, i, j in candidates:
        if not args.allow_reuse:
            if used[i] or used[j]:
                continue
        # accept
        pairs.append((i, j, sim))
        if not args.allow_reuse:
            used[i] = True
            used[j] = True
        if args.max_pairs and args.max_pairs > 0 and len(pairs) >= args.max_pairs:
            break

    if not pairs:
        raise SystemExit("No pairs created. Try increasing --topk or use --allow_reuse.")

    print(f"Created {len(pairs)} pairs.")

    # --- Save pairs ---
    csv_path = os.path.join(args.output_dir, "pairs.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "vid_A", "vid_B", "img_A", "img_B", "cosine_sim", "det_A", "det_B"])
        for idx, (i, j, sim) in enumerate(pairs, start=1):
            vidA, vidB = vids[i], vids[j]
            pathA, pathB = paths[i], paths[j]

            pair_dir = os.path.join(args.output_dir, f"pair_{idx:04d}")
            ensure_dir(pair_dir)

            outA = os.path.join(pair_dir, f"A_{vidA}_best.jpg")
            outB = os.path.join(pair_dir, f"B_{vidB}_best.jpg")

            if args.copy_mode == "symlink":
                # replace if exists
                if os.path.lexists(outA):
                    os.remove(outA)
                if os.path.lexists(outB):
                    os.remove(outB)
                os.symlink(os.path.abspath(pathA), outA)
                os.symlink(os.path.abspath(pathB), outB)
            else:
                shutil.copy2(pathA, outA)
                shutil.copy2(pathB, outB)

            # small meta file (handy for inspection)
            meta_path = os.path.join(pair_dir, "meta.txt")
            with open(meta_path, "w") as mf:
                mf.write(f"pair_id: {idx:04d}\n")
                mf.write(f"vid_A: {vidA}\nvid_B: {vidB}\n")
                mf.write(f"cosine_sim: {sim:.6f}\n")
                mf.write(f"det_A: {det_scores[i]:.4f}\n")
                mf.write(f"det_B: {det_scores[j]:.4f}\n")
                mf.write(f"img_A: {pathA}\n")
                mf.write(f"img_B: {pathB}\n")

            w.writerow([f"{idx:04d}", vidA, vidB, outA, outB, f"{sim:.6f}", f"{det_scores[i]:.4f}", f"{det_scores[j]:.4f}"])

    print(f"Saved pairs to: {args.output_dir}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()

