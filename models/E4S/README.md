# E4S – Verified Execution

This document records how E4S (Editing in Style) was successfully executed on our system **with additional video-processing scripts and environment configuration adjustments**.

The original E4S face-swapping architecture was retained. We developed additional scripts to support image-to-video face swapping, frame processing, face alignment, and video reconstruction.

## System
- OS: Linux 6.8.0-106-generic; x86_64
- Python version: 3.8
- Virtual Environment: Conda (e4s)
- CPU: Physical core: 6, Total core: 12
- GPU: NVIDIA RTX 4500 Ada Generation
- CUDA (PyTorch): 11.3
- Key Libraries:
  - E4S pretrained model
  - FaceVid2Vid
  - GPEN
  - Face parsing (including SegNeXt-based parsing)
    
## Installation
The original E4S installation instructions were followed, with additional environment configuration and custom video-processing scripts.

The core E4S model implementation was not modified.

### 1. Environment Configuration

Problem faced:
The original E4S implementation required compatible PyTorch, CUDA, and custom CUDA extension dependencies.

During execution, GPU extension compilation errors occurred, particularly in the GPEN-related components.

What was changed:
- Created a dedicated Conda environment using Python 3.8.
- Installed the dependencies specified in e4s_env.yaml.
- Configured GCC/G++ 9 for CUDA extension compilation.
- Adjusted the CUDA architecture configuration to support extension compilation in our environment.
- Successfully loaded the required pretrained components.

Why:
To resolve environment compatibility and CUDA extension compilation issues without modifying the original E4S architecture.

### 2. Pretrained Model Configuration

The following pretrained components were configured:
- E4S checkpoint: pretrained_ckpts/e4s/iteration_300000.pt
- Face parsing: pretrained_ckpts/face_parsing/79999_iter.pth
- FaceVid2Vid: pretrained_ckpts/facevid2vid/00000189-checkpoint.pth.tar
- GPEN: pretrained weights under pretrained_ckpts/gpen/

### 3. Image-to-Image Face Swapping

The original E4S image-swapping script was used to verify the installation.

Command used:
```bash
python scripts/face_swap.py \
  --source=example/input/faceswap/source.jpg \
  --target=example/input/faceswap/target.jpg \
  --lap_bld \
  --verbose=True
```
No modification to the original image-swapping model architecture was required.

### 4. Video Face-Swapping Support

Problems faced:
The original image-swapping workflow did not directly provide the complete video-processing procedure required for our biometric evaluation.

Our evaluation required swapping a single source identity into multiple frames of a target video and reconstructing the processed frames into a video.

What was changed:
An additional video-processing script was developed

```bash
video_face_swap.py
```
This script extended the original E4S inference workflow.

Why:
To enable E4S to process video datasets consistently with the other face-swapping models included in our biometric evaluation pipeline.

## Command Used
```bash

# Run Image-to-Image Face Swapping
python scripts/face_swap.py \
  --source=example/input/faceswap/source.jpg \
  --target=example/input/faceswap/target.jpg \
  --lap_bld \
  --verbose=True

# Run Video Face Swapping
python video_face_swap \
  --source_image data/source/000.jpg \
  --target_video data/target/525.mp4 \
  --output_video outputs/000_525_square_swap.mp4 \
  --output_face_video outputs/000_525_target_square_faces.mp4 \
  --lap_bld \
  --max_frames 150

```


# E4S – Original Repository Reference

This directory contains only the **additional scripts developed to extend the original E4S implementation for video face swapping**.

- Original repository: https://github.com/e4s2022/e4s
- Paper: Liu, Z., Li, M., Zhang, Y., Wang, C., Zhang, Q., Wang, J. and Nie, Y., 2023, June. Fine-grained face swapping via regional gan inversion. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 8578-8587). IEEE.
- License: As provided in the original repository

No modifications were made to the core E4S model architecture or pretrained model weights.

The original E4S repository remains the primary source for the complete model implementation.

This directory is included to document the additional scripts and execution adjustments used in our biometric evaluation pipeline.

## Only the additional video-processing scripts are provided.
