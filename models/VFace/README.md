# VFace – Verified Execution

This document records how VFace was executed successfully
on our system **with mild modification**.

## System
- OS: Linux 6.8.0-106-generic; x86_64
- Python version: 3.10.13
- Virtual Environment: Python Executable
- CPU: Physical core: 6, Total core: 12
- GPU: NVIDIA RTX 4500 Ada Generation
- CUDA (Pytorch): 11.7
- Key Libraries:
  - torch: 1.13.1+cu117
  - torchvision: 0.14.1+cu117
  - transformers: 4.30.2
  - huggingface_hub: 0.16.4

## Installation
Followed exactly as described in the original repository. In addition-

The original VFace code did not run reliably in our environment because of several compatibility and runtime issues. To make the pipeline work end-to-end, four scripts were modified for offline model loading, dependency compatibility, runtime stability, and correct video output behavior.

### 1. modules.py

Problem faced:
The original code loaded CLIP models directly from Hugging Face, which failed in our environment because online model resolution was unreliable.

What was changed:
- Replaced remote CLIP loading with local pretrained model paths
- Updated CLIP text/image/model loaders to use the local cached directory

Why:
To make the project work in offline/local mode and avoid Hugging Face download issues.

### 2. inference.py

Problem faced:
The original script imported the Stable Diffusion safety checker directly, which caused dependency/version errors involving diffusers and huggingface_hub.

What was changed:
- Made the safety checker optional
- Wrapped safety-checker loading in a try/except
- Allowed the script to continue even if that component is unavailable

Why:
To prevent the script from crashing due to environment-specific library mismatches.

### 3. face_swap_utils.py

Problem faced:
The original FFT-based attention fusion caused repeated GPU runtime failures:

cuFFT_INTERNAL_ERROR

What was changed:
- Replaced the FFT-based combine_fft_high_low() operation with a safe weighted blending fallback

Why:
- To eliminate cuFFT instability and make the VFace pipeline runnable on our GPU setup.
- This is a stability workaround, so it prioritizes execution reliability over exact frequency-domain behavior.


### 4. VFace_inference_single.py

Problems faced:
- direct safety-checker dependency caused crashes
- inversion step mismatch caused missing latent files
- cached preprocessing could reuse stale transforms
- the pasted swapped face could appear shifted or shrunk because the script reconstructed the whole frame before compositing

What was changed:

- made safety checker optional
- changed inversion to use:
```bash
inverse_steps = opt.ddim_steps
```
- added support for force reprocessing of cached frames/masks/transforms
- corrected paste-back logic to composite the swapped face onto the true original frame, instead of a reconstructed background frame

Why:
- To make video inference stable and to preserve:
- original frame resolution
- correct swapped-face placement
- consistent preprocessing for new videos.


## Command Used
```bash
python scripts/VFace_inference_single.py \
  --target_video examples/FaceSwap/Videos/525.mp4 \
  --src_image examples/FaceSwap/Source/000_best.jpg \
  --n_frames "$FRAMES" \
  --n_samples 1 \
  --ddim_steps 20 \
  --precision full \
  --force_reprocess
```




# VFace – Original Repository Reference

This directory contains only the **modified scripts** derived from the original VFace repository.

- Original repository: https://github.com/Sanoojan/VFace
- Paper: Baliah, S., Abeysinghe, Y., Thushara, R., Muhammad, K., Dhall, A., Nandakumar, K., and Khan, M. H. (2026). *VFace: A Training-Free Approach for Diffusion-Based Video Face Swapping*. In **Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)**, pp. 4315–4324.
- License: As provided in the original repository

Several scripts were modified to improve reproducibility, environment compatibility, offline model loading, runtime stability, and output alignment within our biometric evaluation pipeline.

These changes include:
- replacing remote model loading with local/offline model paths
- making safety-checker dependencies optional
- introducing a stable fallback for FFT-related runtime failures
- correcting video inference and face paste-back behavior to preserve original frame geometry more reliably

The original VFace repository remains the primary source for the full implementation. This directory is included to document the specific script-level changes used in our experimental pipeline.
