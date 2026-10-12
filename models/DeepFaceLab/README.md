# DeepFaceLab – Verified Execution

This document records how DeepFaceLab was successfully executed on our system **without modification to the core model implementation**, but with environment configuration and execution adjustments.

## System
- OS: Linux 6.8.0-106-generic; x86_64
- Python version: 3.7
- Virtual Environment: Conda (deepfacelab)
- CPU: Physical core: 6, Total core: 12
- GPU: 2 x NVIDIA RTX 4500 Ada Generation (24 GB each)
- CUDA Toolkit: 10.1.243
- TensorFlow: 2.4.0
- cuDNN: 8.0.5
- Model: Quick96 (pretrained initialization)

## Installation
The installation followed the original DeepFaceLab Linux repository instructions, with additional dependency configuration and runtime adjustments required for our hardware environment.

The original implementation was designed around older TensorFlow and CUDA dependencies. Therefore, a dedicated Conda environment was created to support execution.

### 1. Environment Configuration

Problem faced:
DeepFaceLab required legacy Python, TensorFlow, CUDA, and cuDNN dependencies that were not directly compatible with our default system environment.

What was changed:
- Created a dedicated Conda environment using Python 3.7.
- Installed TensorFlow 2.4.0.
- Configured CUDA Toolkit 10.1.243 and compatible cuDNN libraries.
- Adjusted the runtime environment to resolve library compatibility issues.
- Verified that TensorFlow could detect the NVIDIA GPU.

Why:
To establish a compatible environment for running the original DeepFaceLab implementation without modifying the model architecture.

### 2. Face Extraction

Problem faced:
The S3FD face detector encountered runtime failures during GPU-based face extraction.

What was changed:
- Switched the face extraction process from GPU to CPU.
- Used the original S3FD extraction functionality.
- Successfully extracted and aligned faces from the video frames.

Why:
To avoid GPU runtime failures while maintaining the original face detection and alignment procedure.

The CPU-based extraction successfully processed 475 frames during our test.

### 3. Quick96 Model Training

Model Used: Quick96

What done:
- Downloaded the pretrained Quick96 model.
- Prepared the source and target face datasets.
- Initialized training using the pretrained weights.
- Continued training on the prepared face datasets.
- Successfully executed approximately 56,471 training iterations.

Why:
Quick96 provides a pretrained starting point for testing the complete DeepFaceLab face-swapping workflow without training a model entirely from scratch.


### 4. Face Merging and Video Generation

What was done:
- Used the trained Quick96 model for face swapping.
- Generated swapped frames using the original merging pipeline.
- Used overlay-based merging.
- Reconstructed the final swapped video.

Observation:

The face-swapping pipeline executed successfully and generated a swapped video.

However, visible boundaries around the swapped facial region were observed in some frames. These artifacts may be associated with mask blending, color differences, and face alignment.

No changes to the core merging algorithm were introduced.


## Command Used
```bash

# Extract Video Frames
bash 2_extract_image_from_data_src.sh
bash 3_extract_image_from_data_dst.sh

#Extract and Align Faces
bash 4_data_src_extract_faces_S3FD.sh
bash 5_data_dst_extract_faces_S3FD.sh

bash 6_train_Quick96.sh
```




# DeepFaceLab  – Original Repository Reference

This directory documents the verified execution procedure and environment adjustments required to run DeepFaceLab on our system.

- Original Linux repository: https://github.com/nagadit/DeepFaceLab_Linux
- Original DeepFaceLab project: https://github.com/iperov/DeepFaceLab
- Paper: Liu, K., Perov, I., Gao, D., Chervoniy, N., Zhou, W. and Zhang, W., 2023. Deepfacelab: Integrated, flexible and extensible face-swapping framework. Pattern Recognition, 141, p.109628.
- License: Refer to the original repositories and their applicable license terms.

No modifications were made to the core DeepFaceLab model architecture or training algorithms.

The implementation required:

- Configuring a compatible legacy TensorFlow/CUDA environment.
- Resolving runtime dependency issues.
- Switching S3FD face extraction from GPU to CPU.
- Preparing source and target video datasets.
- Training the Quick96 model using pretrained initialization.
- Executing the original merging pipeline to generate swapped video output.

The original DeepFaceLab repositories remain the primary sources for the full implementation.

This documentation is included to support reproducibility and to record the environment-specific execution process used in our biometric face-swapping evaluation project.
