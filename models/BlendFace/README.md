# BlendFace – Verified Execution

This document records how BlendFace was successfully executed on our system, with modifications to support full-frame video processing and batch face swapping.

## System
- OS: Linux 6.8.0-106-generic; x86_64
- Python version: 3.10.13
- Virtual Environment: blendface_env
- CPU: Physical core: 6, Total core: 12
- GPU: NVIDIA RTX 4500 Ada Generation
- CUDA (Pytorch): 11.7
- Key Libraries:
  - torch: 1.13.1+cu117
  - torchvision: 0.14.1+cu117
  - numpy: 1.26.4
  - opencv-python: 4.8.1.78

## Installation
Followed the original BlendFace repository installation procedure.

Additional environment adjustments:
- Installed PyTorch 1.13.1 + CUDA 11.7
- Downgraded NumPy to 1.26.4 for compatibility with PyTorch 1.13.1
- Installed OpenCV 4.8.1 to avoid incompatibilities with NumPy 2.x
- Downloaded and placed the required pretrained checkpoints:
  - arcface.pt
  - blendface.pt
  - blendswap.pth

## Modifications
The original BlendFace implementation supports swapping between a single source image and a single target image. To support biometric evaluation on video datasets, an additional preprocessing and batch-processing pipeline was implemented.

### 1. Full-Frame Processing Pipeline

Problem faced:
The original implementation expects aligned face crops as input. Directly supplying full-resolution video frames caused dimensional mismatches and inference failures.

Why:
The BlendFace identity encoder expects fixed-size facial crops rather than full-resolution frames.

### 2. batch_fullframe_swap.py

Problem faced:
The repository provides only image-to-image swapping.

What was changed:
- A custom batch processing script:

Why:
To enable large-scale frame-by-frame face swapping for video datasets and biometric evaluation experiments.

### 3. Temporary Processing Pipeline
Additional directories were introduced:

```bash
swapping/
├── data/
│   ├── source/
│   └── target/
├── temp/
│   ├── source_crop/
│   ├── target_crop/
│   └── swapped_crop/
└── output/
    └── full_frames/
```


Why:
- Store intermediate facial crops
- Preserve original frames
- Simplify debugging and reproducibility


### 4. Video Reconstruction

Problems faced:
- BlendFace only generates swapped images.

What was changed:

- An FFmpeg-based reconstruction step to convert processed frames back into a video.

Example: 
```bash
ffmpeg -framerate 30 \
-i output/full_frames/frame_%06d.jpg \
-c:v libx264 \
-pix_fmt yuv420p \
output/swapped_video.mp4
```

Why:
To generate completely swapped videos for evaluation and demonstration.


## Command Used

Batch Face Swapping:
```bash
python batch_fullframe_swap.py \
    --source data/source/000_best.jpg \
    --target_dir data/target \
    --output_dir output/full_frames \
    --weight checkpoints/blendswap.pth
```

Video Reconstruction:
```bash
ffmpeg -framerate 30 \
-i output/full_frames/frame_%06d.jpg \
-c:v libx264 \
-pix_fmt yuv420p \
output/swapped_video.mp4
```


# BlendFace – Original Repository Reference

This directory contains only the **modified scripts** derived from the original VFace repository.

- Original repository: https://github.com/mapooon/BlendFace
- Paper: Shiohara, K., Yang, X., and Taketomi, T. (2023). BlendFace: Re-designing Identity Encoders for Face-Swapping. In **Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2023)**.
- License: As provided in the original repository

Several additions were made to improve reproducibility and support video-based biometric evaluation.

These changes include:
- automated batch processing of target frames
- face detection and crop generation
- temporary crop management
- full-frame face reinsertion
- video reconstruction using FFmpeg

The original BlendFace repository remains the primary source for the core face-swapping implementation. This directory documents the additional processing steps used in our experimental pipeline for biometric evaluation and large-scale video face-swapping experiments.

The original VFace repository remains the primary source for the full implementation. This directory is included to document the specific script-level changes used in our experimental pipeline.

## Only provided the modified scripts.##
