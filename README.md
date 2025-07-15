# Objectomaly: Objectness-Aware Refinement for OoD Segmentation with Structural Consistency and Boundary Precision
[Jeonghoon Song](modifying..), [Sunghun Kim](modifiying), [Jaegyun Im](https://github.com/imjaegyun), [Byeongjoon Noh](https://scholar.google.com/citations?hl=ko&user=0mPWzzIAAAAJ)


[[`Paper`](https://arxiv.org/abs/2507.07460)] [[`Dataset`](https://drive.usercontent.google.com/download?id=1NL_ApRB-MjVRrMw6ONYZTe1azXc_71yQ&export=download&authuser=0)] [[`BibTeX`](#Citing-Objectomoly)]
> **Objectomaly** is a post-hoc, training-free refinement framework for Out-of-Distribution (OoD) segmentation. It improves structural consistency and boundary precision by incorporating object-level priors through a three-stage pipeline: CAS, OASC, and MBP.
---
## 🧠 Overview
Semantic segmentation models often struggle with unknown or unexpected objects, especially in safety-critical environments like autonomous driving. Existing OoD methods face challenges like:
- :x: Imprecise boundaries
- :x: Inconsistent scores within object regions
- :x: False positives from background textures

**Objectomaly** addresses these challenges through:
### ☑️ Three-Stage Refinement
1. **Coarse Anomaly Scoring (CAS):**
   Generates an initial anomaly map using a baseline OoD detector (e.g., Mask2Anomaly).
2. **Objectness-Aware Score Calibration (OASC):**
   Refines scores using instance masks from [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) for intra-object consistency.
3. **Meticulous Boundary Precision (MBP):**
   Sharpens contours using Laplacian filtering and Gaussian smoothing.
---

## Installation

See [installation instructions](INSTALL.md).

## Citing Objectomoly
If you use Objectomoly in your research, please use the following BibTeX entry.

```bibtex
@article{song2025objectomaly,
  title={Objectomaly: Objectness-Aware Refinement for OoD Segmentation with Structural Consistency and Boundary Precision},
  author={Song, Jeonghoon and Kim, Sunghun and Im, Jaegyun and Noh, Byeongjoon},
  journal={arXiv preprint arXiv:2507.07460},
  year={2025}
}
```
