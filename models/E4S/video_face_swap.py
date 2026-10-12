#!/usr/bin/env python3
import os
import sys
import cv2
import shutil
import tempfile
import argparse
import subprocess
from pathlib import Path
from PIL import Image
import torch

# Import E4S components from original script
import scripts.face_swap as e4s


def run_cmd(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def find_ffmpeg():
    if os.path.exists("/usr/bin/ffmpeg"):
        return "/usr/bin/ffmpeg"
    return "ffmpeg"


def save_bgr(path, frame_bgr):
    cv2.imwrite(str(path), frame_bgr)


def extract_face_square_with_e4s(image_path, output_path):
    """
    Use E4S/FFHQ alignment to create a 1024x1024 square aligned face crop.
    """
    crops, _, _, _ = e4s.crop_and_align_face([("img", str(image_path))])
    crops[0].convert("RGB").save(str(output_path))
    return output_path


def make_e4s_opts(args):
    """
    Build a complete E4S opts object using the original E4S parser.
    This avoids missing attributes such as fsencoder_type, output_size, etc.
    """
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "face_swap.py",
            "--source", args.source_image,
            "--target", args.source_image,
            "--output_dir", str(Path(args.output_video).parent),
            "--checkpoint_path", args.checkpoint_path,
        ]
        if args.lap_bld:
            sys.argv.append("--lap_bld")
        if args.faceParser_name:
            sys.argv += ["--faceParser_name", args.faceParser_name]
        opts = e4s.SwapFacePipelineOptions().parse()
    finally:
        sys.argv = old_argv

    opts.source = args.source_image
    opts.target = args.source_image
    opts.output_dir = str(Path(args.output_video).parent)
    opts.checkpoint_path = args.checkpoint_path
    opts.device = args.device
    opts.lap_bld = args.lap_bld
    opts.faceParser_name = args.faceParser_name

    return opts


def init_models(opts):
    # FaceVid2Vid
    face_vid2vid_cfg = "./pretrained_ckpts/facevid2vid/vox-256.yaml"
    face_vid2vid_ckpt = "./pretrained_ckpts/facevid2vid/00000189-checkpoint.pth.tar"
    e4s.generator, e4s.kp_detector, e4s.he_estimator, e4s.estimate_jacobian = (
        e4s.init_facevid2vid_pretrained_model(face_vid2vid_cfg, face_vid2vid_ckpt)
    )

    # GPEN
    gpen_model_params = {
        "base_dir": "./pretrained_ckpts/gpen/",
        "in_size": 512,
        "model": "GPEN-BFR-512",
        "use_sr": True,
        "sr_model": "realesrnet",
        "sr_scale": 4,
        "channel_multiplier": 2,
        "narrow": 1,
    }
    e4s.GPEN_model = e4s.init_gpen_pretrained_model(model_params=gpen_model_params)

    # Face parser
    if opts.faceParser_name == "default":
        faceParser_ckpt = "./pretrained_ckpts/face_parsing/79999_iter.pth"
        config_path = ""
    elif opts.faceParser_name == "segnext":
        faceParser_ckpt = "./pretrained_ckpts/face_parsing/segnext.small.best_mIoU_iter_140000.pth"
        config_path = "./pretrained_ckpts/face_parsing/segnext.small.512x512.celebamaskhq.160k.py"
    else:
        raise ValueError("faceParser_name must be default or segnext")

    e4s.faceParsing_model = e4s.init_faceParsing_pretrained_model(
        opts.faceParser_name, faceParser_ckpt, config_path
    )
    print("Load pre-trained face parsing models success!")

    # E4S model
    e4s.net = e4s.Net3(opts)
    e4s.net = e4s.net.to(opts.device)
    save_dict = torch.load(opts.checkpoint_path)
    e4s.net.load_state_dict(e4s.torch_utils.remove_module_prefix(save_dict["state_dict"], prefix="module."))
    e4s.net.latent_avg = save_dict["latent_avg"].to(opts.device)
    e4s.net.eval()
    print("Load E4S pre-trained model success!")


def main():
    parser = argparse.ArgumentParser(
        description="E4S source image + target video. Auto square-crop faces, swap, paste back to original video."
    )
    parser.add_argument("--source_image", required=True)
    parser.add_argument("--target_video", required=True)
    parser.add_argument("--output_video", required=True)
    parser.add_argument("--output_face_video", default="", help="Optional video of aligned target square face crops.")
    parser.add_argument("--output_faces_dir", default="", help="Optional directory to save aligned target square crops.")
    parser.add_argument("--work_dir", default="", help="Optional work dir. Empty uses temporary dir.")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--keep_frames", action="store_true")
    parser.add_argument("--lap_bld", action="store_true")
    parser.add_argument("--faceParser_name", default="default", choices=["default", "segnext"])
    parser.add_argument("--checkpoint_path", default="./pretrained_ckpts/e4s/iteration_300000.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    src = Path(args.source_image)
    vid = Path(args.target_video)
    out_video = Path(args.output_video)
    out_video.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise FileNotFoundError(f"Source image not found: {src}")
    if not vid.exists():
        raise FileNotFoundError(f"Target video not found: {vid}")

    ffmpeg = find_ffmpeg()

    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open target video: {vid}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Target video: {width}x{height}, FPS={fps:.3f}, frames={frame_count}")

    opts = make_e4s_opts(args)
    init_models(opts)

    if args.work_dir:
        work_dir = Path(args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="e4s_video_square_"))
        cleanup = not args.keep_frames

    raw_dir = work_dir / "raw_frames"
    crop_dir = work_dir / "target_square_crops"
    swap_dir = work_dir / "swapped_full_frames"
    one_dir = work_dir / "one_frame_swap"
    raw_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    swap_dir.mkdir(parents=True, exist_ok=True)
    one_dir.mkdir(parents=True, exist_ok=True)

    source_square = work_dir / "source_square_1024.png"
    print("Creating square aligned source crop...")
    extract_face_square_with_e4s(src, source_square)

    if args.output_faces_dir:
        Path(args.output_faces_dir).mkdir(parents=True, exist_ok=True)

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    processed = 0
    frame_idx = args.start_frame

    while True:
        if args.max_frames and processed >= args.max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        raw_path = raw_dir / f"frame_{processed:06d}.png"
        save_bgr(raw_path, frame)

        target_square = crop_dir / f"target_square_{processed:06d}.png"
        try:
            extract_face_square_with_e4s(raw_path, target_square)
        except Exception as exc:
            print(f"[WARN] Face alignment failed on frame {frame_idx}: {exc}. Keeping original frame.")
            save_bgr(swap_dir / f"frame_{processed:06d}.png", frame)
            processed += 1
            frame_idx += 1
            continue

        if args.output_faces_dir:
            shutil.copy2(target_square, Path(args.output_faces_dir) / target_square.name)

        frame_out_dir = one_dir / f"frame_{processed:06d}"
        frame_out_dir.mkdir(parents=True, exist_ok=True)

        e4s.faceSwapping_pipeline(
            str(source_square),
            str(raw_path),
            opts,
            save_dir=str(frame_out_dir),
            target_mask=None,
            need_crop=False,
            verbose=False,
            only_target_crop=True,
        )

        candidates = sorted(frame_out_dir.glob("swap_*.png"))
        if not candidates:
            print(f"[WARN] No swapped output found for frame {frame_idx}. Keeping original frame.")
            save_bgr(swap_dir / f"frame_{processed:06d}.png", frame)
        else:
            Image.open(candidates[0]).convert("RGB").save(swap_dir / f"frame_{processed:06d}.png")

        processed += 1
        frame_idx += 1

        if processed % 10 == 0:
            print(f"Processed {processed} frames")

    cap.release()
    print(f"Processed total frames: {processed}")

    if processed == 0:
        raise RuntimeError("No frames were processed.")

    no_audio = work_dir / "e4s_no_audio.mp4"
    run_cmd([
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", str(swap_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(no_audio),
    ])

    try:
        run_cmd([
            ffmpeg, "-y",
            "-i", str(no_audio),
            "-i", str(vid),
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(out_video),
        ])
    except subprocess.CalledProcessError:
        print("[WARN] Audio copy failed. Saving no-audio video instead.")
        shutil.copy2(no_audio, out_video)

    if args.output_face_video:
        face_video = Path(args.output_face_video)
        face_video.parent.mkdir(parents=True, exist_ok=True)
        run_cmd([
            ffmpeg, "-y",
            "-framerate", str(fps),
            "-i", str(crop_dir / "target_square_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(face_video),
        ])

    print(f"Saved full swapped video: {out_video}")
    if args.output_face_video:
        print(f"Saved aligned face/crop video: {args.output_face_video}")

    if cleanup:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"Kept working directory: {work_dir}")


if __name__ == "__main__":
    main()
