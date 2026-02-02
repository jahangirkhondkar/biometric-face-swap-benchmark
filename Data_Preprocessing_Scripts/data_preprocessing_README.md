# Data Preprocessing Scripts

This directory contains **general-purpose preprocessing scripts** for preparing
video-based face datasets for **biometric face-swapping and identity manipulation
evaluation**.

The pipeline is **dataset-agnostic** and can be applied to any video dataset
containing human faces. It enforces identity clarity, facial quality, and
expression neutrality prior to downstream face-swapping or biometric analysis.

---

## Processing Pipeline

The preprocessing workflow consists of four modular stages:

1. **Video Curation**  
   Filters raw videos using face detection and quality-based criteria to remove
   samples that may introduce identity ambiguity or unstable biometric cues.

2. **Frame Extraction**  
   Extracts frames from curated videos to enable frame-level analysis and
   selection.

3. **Best Frame Selection**  
   Selects a single, biometrically stable frame per video using detection
   confidence, face size, sharpness, and expression-neutrality metrics.

4. **Pair Construction**  
   Constructs face pairs and removes same-identity matches using face embedding
   similarity to ensure identity-distinct evaluation pairs.

Each stage is independent and can be executed or adapted separately.

---

## Scripts

- `video_curation.py`  
  Performs video-level filtering using face detection and quality metrics
  (e.g., multi-face presence, blur, face size, pose, and detection reliability).

- `extract_frames.py`  
  Extracts frames from curated videos for downstream processing.

- `select_best_frame.py`  
  Identifies the most biometrically stable frame per video based on weighted
  quality and expression metrics.

- `make_pairs.py`  
  Generates identity-distinct face pairs using embedding similarity to eliminate
  same-subject matches.

---


## Dependencies

- InsightFace (FaceAnalysis, buffalo_l model pack)
- OpenCV
- NumPy
- SciPy

Exact versions and environment details are documented at the repository level.

---
