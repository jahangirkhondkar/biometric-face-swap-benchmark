# ReliableSwap – Verified Execution

This document records the installation and configuration of ReliableSwap on our system, including **dependency compatibility adjustments, CUDA library configuration, FaceBoxesV2 compilation, and pretrained model preparation**.

The original ReliableSwap model architecture was retained. The implementation required resolving several environment-specific dependency issues before inference could be attempted.

## System
- OS: Linux 6.8.0-106-generic; x86_64
- Python version: 3.8
- Virtual Environment: reliableswap_env
- CPU: Physical core: 6, Total core: 12
- GPU: NVIDIA RTX 4500 Ada Generation
- CUDA (Pytorch): 11.7
- PyTorch: 1.13.1+cu117
- Key Libraries:
  - TensorFlow: 2.10
  - MXNet: 1.9.1
  - InsightFace: 1.0.1
  - OpenCV: 4.8.1
  - NumPy: 1.23.5
  - ONNXRuntime
  - SciPy
  - scikit-image
  - Cython

## Installation
The original ReliableSwap repository was used as the primary implementation.

However, several dependency compatibility issues were encountered during installation.

To prepare the pipeline for execution, the following environment-specific adjustments were required.

### 1. Environment Configuration

Problem faced:
The original ReliableSwap implementation depended on multiple deep learning frameworks and supporting libraries.

Dependency conflicts occurred between PyTorch, TensorFlow, MXNet, NumPy, and the available CUDA libraries.

What was changed:
- Created a dedicated Python 3.8 virtual environment.
- Installed PyTorch 1.13.1 with CUDA 11.7 support.
- Installed TensorFlow 2.10 and MXNet 1.9.1.
- Configured the required supporting libraries.
- Downgraded NumPy to version 1.23.5 to resolve compatibility issues.
- Verified that the main Python dependencies could be imported successfully.

Why:
To establish a compatible runtime environment for ReliableSwap without modifying the original model architecture.

### 2. FaceBoxesV2 Compilation

Problem faced:
ReliableSwap uses face detection and landmark-processing components that require compiled extensions.

During installation, the FaceBoxesV2 non-maximum suppression (NMS) components required compilation.

What was changed:
- Installed the required Cython dependency.
- Compiled the FaceBoxesV2 NMS extension.
- Verified that the compiled components could be imported.

Why:
To enable the original face detection and preprocessing components required by the ReliableSwap inference pipeline.

### 3. InsightFace Compatibility

Problem faced:
The original ReliableSwap implementation required InsightFace and related face recognition dependencies.

Several dependency conflicts occurred during installation.

What was changed:
- Installed InsightFace 1.0.1.
- Resolved the supporting library compatibility issues.
- Verified successful InsightFace import.

Why:
To prepare the face recognition and identity-related components required by the ReliableSwap pipeline.


## Command Used
```bash
python inference/faceshifter_infer.py \
    -s data/source/source.jpg \
    -t data/target/target.mp4 \
    -o outputs/ \
    -n swapped.mp4 \
    -cp inference/extracted_pth/G_mouth1_t38_post.pth \
    --align_source ffhq \
    --align_target ffhq \
```


# ReliableSwap – Original Repository Reference

This directory documents the **environment configuration, dependency fixes, compiled extension preparation, and pretrained model setup** required for ReliableSwap.

- Original repository: https://github.com/ygtxr1997/ReliableSwap
- Pretrained model resources: https://huggingface.co/spaces/ygtxr1997/ReliableSwap_Demo
- Paper: Yuan, G., Li, M., Zhang, Y. and Zheng, H., 2023. Reliableswap: Boosting general face swapping via reliable supervision. arXiv preprint arXiv:2306.05356.
- License: As provided in the original repository

No modifications to the original ReliableSwap model architecture or pretrained model.

The original ReliableSwap repository remains the primary source for the complete model implementation.

This documentation is included to record the environment-specific implementation procedure and support future reproducibility within our biometric face-swapping evaluation pipeline.
