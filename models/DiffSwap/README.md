# DiffSwap – Verified Execution

This document records how DiffSwap was successfully executed on our system **with minor compatibility modifications and an additional video-processing script**.

The original DiffSwap face-swapping architecture and pretrained weights were retained. The implementation required resolving dependency compatibility issues, adjusting a Python import, and developing an additional workflow to support image-to-video face swapping.

Both image-to-image and image-to-video face swapping were successfully verified.

## System
- OS: Linux 6.8.0-106-generic; x86_64
- Python version: 3.8
- Virtual Environment: Conda (diffswap_env)
- CPU: Physical core: 6, Total core: 12
- GPU: NVIDIA RTX 4500 Ada Generation
- CUDA (Pytorch): 11.7
- PyTorch: 1.13.1+cu117
- Key Libraries:
  - TensorFlow: 2.13.0
  - MTCNN: 0.1.1
  - dlib: 19.24.2
  - Kornia: 0.7.0
  - PyTorch Lightning: 1.4.2
  - openai-clip: 1.0.1
  - Pillow: 9.5.0

## Installation
The original DiffSwap repository was used as the primary implementation.

However, several environment compatibility issues were encountered during installation and inference.

To execute the pipeline successfully, the following adjustments were required.

### 1. Environment Configuration

Problem faced:
The original DiffSwap implementation required several dependencies with version compatibility constraints.

During installation, conflicts occurred involving TensorFlow, PyTorch Lightning, dlib, Pillow, and other supporting packages.

What was changed:
- Created a dedicated Conda environment using Python 3.8.
- Installed PyTorch 1.13.1 with CUDA 11.7 support.
- Installed compatible versions of TensorFlow, MTCNN, Kornia, and PyTorch Lightning.
- Installed dlib-bin==19.24.2 to resolve dlib installation issues.
- Configured Pillow 9.5.0 and other supporting packages.
- Installed additional dependencies, including SciPy, scikit-image, imutils, transformers, matplotlib, and wandb.

Why:
To establish a compatible environment for running the original DiffSwap implementation without changing its model architecture.

### 2. Video Face-Swapping Support

Problem faced:
The original DiffSwap inference workflow was designed primarily for image-based face swapping.

Our biometric evaluation required processing a target video using a single source face image.

The evaluation also required processing approximately 150 frames per video.

Additional script:
```bash
run_diffswap_video.sh
```

What was changed:
- Accepts a source face image and target video.
- Extracts target video frames.
- Uses PNG frames to avoid additional lossy compression during intermediate processing.
- Organizes source and target images for DiffSwap inference.
- Executes the image-swapping pipeline across video frames.
- Collects the generated swapped frames.
- Reconstructs the processed frames into a video.
- Supports processing up to 150 frames.

Why:
To extend the original image-based DiffSwap implementation for video-based biometric evaluation without modifying the underlying face-swapping architecture.

### 3. Modified Script

```bash
tests/faceswap_portrait.py
```
Type: Modified original script

Modification:
Corrected the import path for the portrait-processing utility.

Purpose:
To resolve Python module import errors during inference.


## Command Used
```bash
# Run Image-to-Image Face Swapping
python pipeline.py

# Run Image-to-Video Face Swapping
bash run_diffswap_video.sh
```
The script prepares the target video frames, executes DiffSwap inference, and reconstructs the resulting swapped frames into a video.

The source image, target video, and frame limit should be configured according to the supplied script.



# DiffSwap – Original Repository Reference

This directory contains only the modified and additional scripts required to execute DiffSwap within our biometric face-swapping evaluation pipeline.

- Original repository: https://github.com/wl-zhao/DiffSwap
- Paper: Zhao, W., Rao, Y., Shi, W., Liu, Z., Zhou, J. and Lu, J., 2023, June. Diffswap: High-fidelity and controllable face swapping via 3d-aware masked diffusion. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 8568-8577). IEEE.
- License: As provided in the original repository

The original DiffSwap model architecture and pretrained model weights were not modified.

Several execution adjustments were required to improve compatibility and extend the model to video processing.

These changes include:
- Configuring a compatible Python, PyTorch, and TensorFlow environment.
- Resolving dependency installation and compatibility issues.
- Correcting the portrait-processing module import.
- Configuring the Python module search path.
- Developing a video-processing Bash script.

The original DiffSwap repository remains the primary source for the complete model implementation.

## Only the modified and additional scripts are provided.
