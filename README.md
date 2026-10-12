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
|   14  | SimSwap       | Face Swapping   |
|-------|---------------|-----------------|-------------|---------------------------------------|


