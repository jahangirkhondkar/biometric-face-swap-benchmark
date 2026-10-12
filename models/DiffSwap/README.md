# FaceDancer – Verified Execution

This document records how FaceDancer was executed successfully
on our system **without any modification**.

## System
- OS: Linux 6.8.0-106-generic; x86_64
- Python: 3.8
- Virtual Environment: Conda (facedancer)
- CPU: Physical cores: 6, Total logical cores: 12
- GPU: NVIDIA RTX 4500 Ada Generation
- Framework: TensorFlow
- Execution: TensorFlow CPU-based inference observed during testing
- Key Components:
  - FaceDancer pretrained generator
  - ArcFace-Res50 identity recognition model
  - RetinaFace-Res50 face detection model
  - OpenCV
  - FFmpeg

## Installation
Followed exactly as described in the original repository.

## Command Used
```bash
# Run Image-to-Image Face Swapping
python test_image_swap_multi.py

# Run Image-to-Video Face Swapping
python test_video_swap_multi.py \
  --facedancer_path "./model_zoo/FaceDancer_config_c_HQ.h5" \
  --vid_path "data/target/0127344_angles_15_ft_H.mp4" \
  --swap_source "data/source/0401573_frame.jpg" \
  --vid_output "results/swapped_FaceDancer.mp4"
```


# FaceDancer – Original Repository Reference

This directory contains an **unmodified copy** of the CanonSwap repository.

- Original repository: https://github.com/felixrosberg/FaceDancer
- Paper: Rosberg, F., Aksoy, E.E., Alonso-Fernandez, F. and Englund, C., 2023, January. Facedancer: Pose-and occlusion-aware high fidelity face swapping. In 2023 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (pp. 3443-3452). IEEE.
- License: As provided in the original repository

No source code has been changed. The repository is included here solely to
verify reproducibility and compatibility within our biometric evaluation
pipeline.
