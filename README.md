# biometric-face-swap-benchmark

This repository aggregates verified executions of publicly available
face-swapping models and provides original preprocessing scripts for
biometric evaluation. All third-party models are included without
modification unless explicitly stated and are referenced to their
original repositories. Any future changes or fixes will be clearly
documented.

## Data Preprocessing Scripts
This folder contains all **original preprocessing scripts**
used for dataset curation, best-frame selection, and pair construction.


## Face-Swapping Models
```markdown
# Face-Swapping Models

Serial  | Model         | Type            | Temporal Modeling | Status      | Notes                  |
|-------|---------------|-----------------|-------------------|-------------|------------------------|
|   1   | CanonSwap     | Face Swapping   | Yes               | Verified    | Runs unmodified        |
|-------|--------------------------------------------------------------------------------------------|
|   2   | DiffFace      | Face Swapping   | Yes               | Verified    | Runs unmodified        |
|-------|--------------------------------------------------------------------------------------------|
|   3   | REFace        | Face Swapping   | Yes               | Verified    | Runs unmodified        |
|-------|-------------------------------------------------------------------------------|
|   4   | VFace         | Face Swapping   | Yes               | Verified    | Runs with modification |
|-------|-------------------------------------------------------------------------------|
|   5   | BlendFace     | Face Swapping   | Yes               | Verified    | Runs unmodifed + add additional step
|-------|-------------------------------------------------------------------------------|
|   6   | GHOST         | Face Swapping   | Yes               | Verified    | Runs with modification |
|-------|-------------------------------------------------------------------------------|
|   7   | DeepFaceLab   | Face Swapping   |
|-------|---------------------------
|   8   | E4S           | Face Swapping   |
|-------|----------------------------
|   9   | 3dSwap        | Face Swapping   |
|-------|-----------------------------
|   10  | VividFace     | Face Swapping   |
|-------|-----------------------------
|   11  | ReliableSwap  | Face Swapping   |
|-------|-------------------------------
|   12  | FaceDancer    | Face Swapping   |
|-------|-------------------------------
|   13  | DiffSwap      | 


