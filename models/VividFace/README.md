# VividFace – Verified Execution

This document records how VividFace was successfully executed on our system **with environment adjustments, pretrained model path configuration, additional preprocessing scripts, and minor inference modifications**.

The original VividFace model architecture was retained. Additional preprocessing scripts and inference adjustments were used to support custom source images and target videos within our biometric evaluation pipeline.

## System
- OS: Linux 6.8.0-106-generic; x86_64
- Python version: 3.8.20
- Virtual Environment: vividface_env
- CPU: Physical core: 6, Total core: 12
- GPU: NVIDIA RTX 4500 Ada Generation
- PyTorch: 2.4.1+cu121
- CUDA (PyTorch): 12.1
- Key Libraries:
  - diffusers: 0.27.2
  - nvdiffrast: 0.3.0
  - torchvision
  - transformers

## Installation
The original VividFace installation instructions were followed. However, several environment compatibility issues, pretrained model loading problems, and input-processing limitations were encountered.

To execute the pipeline successfully, the following adjustments were made.

### 1. Environment Configuration

Problem faced:
The original implementation required several dependencies, including nvdiffrast, which initially failed during installation because of CUDA extension compilation issues.

What was changed:
- Created a dedicated Python 3.8 environment.
- Installed PyTorch 2.4.1 with CUDA 12.1 support.
- Installed diffusers 0.27.2 and supporting dependencies.
- Resolved the CUDA extension installation issue by installing nvdiffrast version 0.3.0.
- Verified GPU availability through PyTorch.

Why:
To establish a compatible runtime environment for executing the pretrained VividFace inference pipeline.

### 2. Custom Source Image and Target Video Support

Problem faced:
The original example-based inference workflow required a specific directory structure and input format.

Our biometric evaluation required using a custom source face image and target video.

What was changed:
Two additional preprocessing scripts were developed:
```bash
generate_vividface_txt.py
prepare_vividface_target.py
```
These scripts were used to prepare the target video and associated metadata for inference.

Why:
To support our custom source and target datasets without modifying the underlying face-swapping architecture.

### 3. Target Video Cropping

Problem faced:
Direct inference using the original full-resolution target video resulted in a tensor reshaping error:

```bash
shape '[-1, 1, 512, 512]'
```
The model's preprocessing pipeline expected a compatible spatial input size.

What was changed:
- Added target video preprocessing.
- Cropped the target video to a 512 × 512 square format.
- Prepared the source image, target video, and metadata in the directory structure expected by VividFace.

The prepared input structure was:

```bash
my_examples_cropped/
├── faces/
│   └── 1.jpg
├── videos/
│   └── 1.mp4
└── 1.txt
```


Why:
To ensure that the target frames matched the expected model input dimensions and avoid tensor shape mismatches during inference.


### 4. Inference Frame-Length Modification

Problems faced:
The original inference script limited video processing to 80 frames.

Our evaluation required processing at least 150 frames for each target video.

Script Modified:
```bash
infer.py
```

What was changed:

- The frame-length limitation was adjusted to:
```bash
video_length = min(video_length, 152)
```

Why:
To support longer face-swapping sequences for biometric evaluation while preserving the original inference architecture.

For a 25 FPS video, 152 frames correspond to approximately 6.08 seconds.

This adjustment changes the maximum processed sequence length, not the pretrained model weights or network architecture.


## Command Used
```bash
# Generate Target Video Metadata
python generate_vividface_txt.py \
  --video data/target/525.mp4 \
  --output data/target/525.txt

# Run VividFace
python infer.py my_examples_cropped
```


# VividFace – Original Repository Reference

This directory contains the **additional preprocessing scripts and modified inference code** used to execute VividFace with our custom biometric dataset.

- Original repository: https://github.com/deepcs233/VividFace
- Paper: Shao, H., Wang, S., Zhou, Y., Song, G., He, D., Qin, S., Zong, Z., Ma, B., Liu, Y. and Li, H., 2024. Vividface: A diffusion-based hybrid framework for high-fidelity video face swapping. arXiv preprint arXiv:2412.11279.
- License: As provided in the original repository

The original VividFace model architecture and pretrained model weights were not modified.

Several execution adjustments were required to improve compatibility and reproducibility.

These changes include:
- Resolving PyTorch and CUDA extension compatibility issues.
- Installing a compatible version of nvdiffrast.
- Configuring local pretrained model directories.
- Resolving the face3dvae2 model path mismatch.
- Adding preprocessing scripts for custom source images and target videos.
- Preparing square-cropped 512 × 512 target videos.
- Extending the inference frame limit from 80 to 152 frames.

The original VividFace repository remains the primary source for the complete model implementation.

This directory documents the specific modifications and additional scripts used in our biometric face-swapping evaluation pipeline.

## Only the modified and additional scripts are provided.
