# 3dSwap – Verified Execution

This document records how 3dSwap was executed on our system **with additional dataset preparation, camera-pose configuration, and inference adjustments**.

## System
- OS: Linux 6.8.0-106-generic; x86_64
- Python version: 3.8
- Virtual Environment: Conda (3dswap_env)
- CPU: Physical core: 6, Total core: 12
- GPU: NVIDIA RTX 4500 Ada Generation
- Model: 3dSwap
- Key Components:
  - EG3D-based 3D-aware generator
  - Pretrained encoder
  - MLP network
  - ArcFace identity model
  - Camera-pose estimation and processing

## Installation
The original 3dSwap installation instructions were followed, with additional environment configuration, pretrained checkpoint preparation, and dataset adjustments.

### 1. Environment Configuration

Problem faced:
The original 3dSwap implementation required specific dependencies and pretrained models for 3D-aware facial reconstruction and swapping.

What was changed:
- Created a dedicated Conda environment using Python 3.8.
- Configured the required PyTorch and supporting dependencies.
- Prepared the pretrained checkpoints.
- Configured local paths for the models and dataset.
- Prepared camera-pose inputs compatible with the inference pipeline.

Why:
To execute the original 3dSwap pipeline using our local computational environment and facial image dataset.

### 2. Dataset Preparation

Problem faced:
The inference pipeline required facial images and corresponding camera-pose information in a specific directory structure.

What was changed:
Prepared the CelebA-HD dataset structure:
```bash
datasets/
└── CelebA-HD/
    ├── final_crops/
    │   ├── 0.jpg
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── ...
    └── camera_pose/
        ├── 0.npy
        ├── 1.npy
        ├── 2.npy
        └── ...
```
Each image was paired with a corresponding camera-pose file.

Why:
The EG3D-based generator uses camera conditioning to control the 3D-aware rendering process. Therefore, corresponding camera-pose information is required for inference.

### 3. Camera-Pose Processing

Problem faced:
The camera-pose files were not always stored as directly usable numerical arrays.

Some .npy files contained serialized Python objects or dictionaries rather than a flat camera-conditioning vector.

What was changed:
- Inspected the contents and shapes of the camera-pose files.
- Identified the stored camera parameters.
- Prepared the camera-conditioning inputs required by the model.
- Used corresponding image and camera-pose filenames to maintain consistency.

Why:
To ensure that the camera-conditioning information could be loaded and interpreted correctly by the 3D-aware generator.

The EG3D camera-conditioning representation commonly contains 25 values, comprising camera extrinsic and intrinsic parameters.


### 4. 3D-Aware Face-Swapping Inference

Script used:
```bash
run_3dSwap.py
```

What was done:
- Loaded the pretrained generator and associated model components.
- Selected source and target images using dataset indices.
- Loaded the corresponding camera-pose information.
- Executed the optimization-based inference procedure.
- Tested the pipeline using 500 optimization iterations.

Why:
To evaluate the 3D-aware face-swapping procedure using the prepared image dataset and pretrained models.


## Command Used
```bash
python run_3dSwap.py \
  --from_index 0 \
  --to_index 1 \
  --epoch 500 \
  --lr 3e-4 \
  --dataroot datasets/CelebA-HD
```




# VFace – Original Repository Reference

This directory documents the **environment configuration, dataset preparation, camera-pose handling, and inference procedure** used to execute 3dSwap.

- Original repository: https://github.com/lyx0208/3dSwap
- Paper: Li, Y., Ma, C., Yan, Y., Zhu, W. and Yang, X., 2023, June. 3d-aware face swapping. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 12705-12714). IEEE.
- License: As provided in the original repository

No modifications to the original model architecture or pretrained weights are documented.

The original 3dSwap repository remains the primary source for the complete model implementation.

This documentation is included to support reproducibility and record the environment-specific execution procedure used in our biometric face-swapping evaluation pipeline.
Modified or Additional Scripts

The inference workflow used run_3dSwap.py

## Only the modified script is provided.
