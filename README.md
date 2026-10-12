# biometric-face-swap-benchmark

**CITeR Project**: 24F-07C-G
**Project Title**: Study of Identity Preservation in Deepfake Face Swapping Tools and 3D Mask Creation

## Overview
This repository contains the implementation, preprocessing, and evaluation scripts developed and adapted for **CITeR Project 24F-07C-G: Study of Identity Preservation in Deepfake Face Swapping Tools and 3D Mask Creation**.

This repository aggregates verified executions of publicly available
face-swapping models and provides original preprocessing scripts for
biometric evaluation. All third-party models are included without
modification unless explicitly stated and are referenced to their
original repositories. Any future changes or fixes will be clearly
documented.

## Repository Structure

```bash
biometric-face-swap-benchmark/
│
├── Data_preprocessing_scripts/
│
├── models/
│   ├── CanonSwap/
│   ├── DiffFace/
│   ├── REFace/
│   ├── VFace/
│   ├── BlendFace/
│   ├── GHOST/
│   ├── DeepFaceLab/
│   ├── E4S/
│   ├── 3dSwap/
│   ├── VividFace/
│   ├── ReliableSwap/
│   ├── FaceDancer/
│   ├── DiffSwap/
│   └── SimSwap/
│
├── Evaluation_scripts/
│
├── Other_scripts/
│
└── README.md
```

### 1. Data_preprocessing_scripts

Contains original scripts developed for dataset preparation, including video frame extraction, facial image selection, preprocessing, dataset organization, and source-target pair construction.

### 2. models

Contains implementation documentation and modified or additional scripts for 14 publicly available face-swapping models.

Each model directory includes a dedicated README describing the original repository, execution environment, installation procedure, commands used, implementation adjustments, and verification status.

Only modified or newly developed scripts are included where applicable. Unmodified third-party implementations are referenced through their original repositories.

### 3. Evaluation_scripts

Contains scripts for biometric evaluation of generated face-swapped images and videos, including identity similarity analysis, face recognition-based comparisons, and quantitative performance evaluation.

### 4. Other_scripts

Contains additional utilities and supporting scripts used for experimental processing, automation, visualization, and related research activities.

## Models
The following models were verified and executed for the project.

### Face-Swapping Models
```bash
|Serial | Model         | Type            | Status      |                Notes                  |
|-------|---------------|-----------------|-------------|---------------------------------------|
|   1   | CanonSwap     | Face Swapping   | Verified    |               Runs unmodified         |
|-------|---------------|-----------------|-------------|---------------------------------------|
|   2   | DiffFace      | Face Swapping   | Verified    |               Runs unmodified         |
|-------|---------------|-----------------|-------------|---------------------------------------|
|   3   | REFace        | Face Swapping   | Verified    |               Runs unmodified         |
|-------|---------------|-----------------|-------------|---------------------------------------|
|   4   | VFace         | Face Swapping   | Verified    |           Runs with modification      |
|-------|---------------|-----------------|-------------|---------------------------------------|
|   5   | BlendFace     | Face Swapping   | Verified    | Runs unmodifed + add additional step  |
|-------|---------------|-----------------|-------------|---------------------------------------|
|   6   | GHOST         | Face Swapping   | Verified    |           Runs with modification      |
|-------|---------------|-----------------|-------------|---------------------------------------|
|   7   | DeepFaceLab   | Face Swapping   | Verified    | Runs unmodified; environment and CPU  |
|       |               |                 |             |        extraction adjustments         |
|-------|---------------|-----------------|-------------|---------------------------------------|
|   8   | E4S           | Face Swapping   | Verified    | Runs with additional video scripts    |
|-------|---------------|-----------------|-------------|---------------------------------------|
|   9   | 3dSwap        | Face Swapping   | Verified    |          Runs with modification       |
|-------|---------------|-----------------|-------------|---------------------------------------|
|   10  | VividFace     | Face Swapping   | Verified    | Runs with modification and additional |
|       |               |                 |             |        preprocessing scripts          |
|-------|---------------|-----------------|-------------|---------------------------------------|
|   11  | ReliableSwap  | Face Swapping   | Verified    |               Runs unmodified         |
|-------|---------------|-----------------|-------------|---------------------------------------|
|   12  | FaceDancer    | Face Swapping   | Verified    |               Runs unmodified         |
|-------|---------------|-----------------|-------------|---------------------------------------|
|   13  | DiffSwap      | Face Swapping   | Verified    | Runs with minor modification and      |
|       |               |                 |             |    additional video scripts           |
|-------|---------------|-----------------|-------------|---------------------------------------|
|   14  | SimSwap       | Face Swapping   |Verified     |               Runs unmodified         |
|-------|---------------|-----------------|-------------|---------------------------------------|

```
## Reproducibility

This repository emphasizes transparent documentation and reproducibility.

- Original third-party models are properly referenced.
- Modifications and additional scripts are documented in the corresponding model directories.
- Execution environments and commands are provided where applicable.
- Pretrained model weights and original datasets are not redistributed unless permitted by their respective licenses.
- Model execution status is documented separately from biometric evaluation results.

## Research Use and Attribution

This repository is maintained for academic research and biometric security evaluation.

All third-party models, datasets, and pretrained weights remain the intellectual property of their respective authors and are subject to their original licenses and usage conditions.

The original model repositories should be consulted for complete implementations, publications, and licensing information.

## Acknowledgment

This work is conducted as part of **CITeR (Center for Identification Technology Research), Project 24F-07C-G: Study of Identity Preservation in Deepfake Face Swapping Tools and 3D Mask Creation**.
