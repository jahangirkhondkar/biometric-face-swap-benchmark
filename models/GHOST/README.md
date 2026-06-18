# GHOST – Verified Execution

This document records how GHOST was executed successfully
on our system **with mild modification**.

## System
- OS: Linux 6.8.0-106-generic; x86_64
- Python version: 3.8.20
- Virtual Environment: Python Executable
- CPU: Physical core: 6, Total core: 12
- GPU: NVIDIA RTX 4500 Ada Generation
- CUDA Driver: 13.0
- Key Libraries:
  - torch: 1.6.0+cu101
  - torchvision: 0.7.0+cu101
  - 0nnx: 1.9.0
  - numpy: 1.23.5
  - mxnet: 1.8.0.post0

## Installation
The repository was installed following the original instructions. However, the original codebase could not run directly in our environment due to dependency incompatibilities and legacy CUDA assumptions.

Several modifications were required to execute the pipeline successfully.

### 1. Dependency Compatibility Fixes

Problem faced:
The original repository depends on older package versions that are no longer compatible with current Python package releases.

Observed issues included:
- ONNX protobuf descriptor errors
- NumPy deprecation issues (np.object)
- MXNet CUDA runtime failures
- unavailable onnxruntime-gpu==1.4.0

What was changed:
Installed compatible versions:
```bash
protobuf==3.20.3
numpy==1.23.5
mxnet==1.8.0.post0
```
Replaced unsupported ONNX Runtime dependencies with versions available for the current package repositories.

Why:
To restore compatibility between ONNX, InsightFace, MXNet, and modern Python package ecosystems.


### 2. CPU Execution Compatibility

Problem faced:
The original implementation assumes CUDA availability throughout the inference pipeline.

Examples:
```python
G.cuda()
netArc.cuda()
torch.from_numpy(...).cuda()
```
On our system:
```bash
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
because PyTorch 1.6 + CUDA 10.1 does not support the NVIDIA RTX 4500 Ada architecture.

What was changed:
- Modified inference scripts to execute on CPU:
```python
device = torch.device("cpu")

G = G.to(device).float()
netArc = netArc.to(device).float()

#Changed InsightFace
ctx_id = -1

```


### 3. Command-Line Argument Fix

What was changed:
- Replaced
```python
parser.add_argument(
    '--image_to_image',
    action='store_true'
)

```

Why:
- To correctly distinguish image-to-image and image-to-video execution modes.


### 4. Video blending code

The image pipeline was converted to CPU execution, but the video blending stage still contained GPU-specific operations.

Original Code:
```python
swap = torch.from_numpy(swap).cuda()
mask = torch.from_numpy(mask).cuda()
full_frame = torch.from_numpy(result_frames[i]).cuda()
mat = torch.from_numpy(tfm_array[j][i]).cuda()

```

Changed To:

```python
swap = torch.from_numpy(swap)
mask = torch.from_numpy(mask)
full_frame = torch.from_numpy(result_frames[i])
mat = torch.from_numpy(tfm_array[j][i])

```

Reason:
The original implementation assumed CUDA availability and silently failed on unsupported hardware.

## Command Used

Image_to_Image

```bash
python inference.py \
  --source_paths input/000_best.jpg \
  --target_image input/525_best.jpg \
  --image_to_image \
  --out_image_name output/ghost_result.jpg
```

For_Video

```bash
python inference.py \
  --source_paths input/000_best.jpg \
  --target_video input/525.mp4 \
  --out_video_name output/ghost_video_result.mp4
```


# GHOST – Original Repository Reference

This directory contains only the **modified scripts** derived from the original GHOST repository.

- Original repository: https://github.com/ai-forever/ghost
- Paper: A. Groshev, A. Maltseva, D. Chesakov, A. Kuznetsov and D. Dimitrov, "GHOST—A New Face Swap Approach for Image and Video Domains," in IEEE Access, vol. 10, pp. 83452-83462, 2022, doi: 10.1109/ACCESS.2022.3196668.
- License: As provided in the original repository

Several scripts were modified to improve reproducibility and environment compatibility.

The original GHOST repository remains the primary source for the full implementation. This directory is included to document the specific script-level changes used in our experimental pipeline.
Only provided the modified scripts.
