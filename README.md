# Objectomaly: Objectness-Aware Refinement for OoD Segmentation with Structural Consistency and Boundary Precision

[Jeonghoon Song](modifying..), [Sunghun Kim](modifiying), [Jaegyun Im](https://github.com/imjaegyun), [Byeongjoon Noh](https://scholar.google.com/citations?hl=ko&user=0mPWzzIAAAAJ)

[[`Paper`](https://arxiv.org/abs/2507.07460)] [[`Dataset`](https://drive.usercontent.google.com/download?id=1NL_ApRB-MjVRrMw6ONYZTe1azXc_71yQ&export=download&authuser=0)] [[`BibTeX`](#Citing-Objectomoly)]

![SAM 2 architecture](fig-src-dat-at.png)

> **Objectomaly** is a post-hoc, training-free refinement framework for Out-of-Distribution (OoD) segmentation. It improves structural consistency and boundary precision by incorporating object-level priors through a three-stage pipeline: CAS, OASC, and MBP.

---

## Overview

Semantic segmentation models often struggle with unknown or unexpected objects, especially in safety-critical environments like autonomous driving. Existing OoD methods face challenges like:

![SAM 2 architecture](image.png)

- Inaccurate boundaries between adjacent objects (left)
- Lack of spatial consistency within anomaly scores of the same object (middle)
- Increase in false positives due to background noise (right)

---

## **Objectomaly** addresses these challenges through
# *Overall Architecture*
### Three-Stage Refinement

1. **Coarse Anomaly Scoring (CAS):**
   Generates an initial anomaly map using a baseline OoD detector (e.g., Mask2Anomaly).
2. **Objectness-Aware Score Calibration (OASC):**
   Refines scores using instance masks from [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) for intra-object consistency.
3. **Meticulous Boundary Precision (MBP):**
   Sharpens contours using Laplacian filtering and Gaussian smoothing.

---

## Quantitative Results

### Pixel-Level Metrics (AuPRC ↑ / FPR<sub>95</sub> ↓)

| Method            | SMIYC AT         | SMIYC OT         | RA               |
| ----------------- | ---------------- | ---------------- | ---------------- |
| RPL               | 88.55 / 7.18     | 96.91 / 0.09     | 71.61 / 17.74    |
| Maskomaly         | 93.40 / 6.90     | 0.96 / 96.14     | 70.90 / 11.90    |
| RbA               | 86.10 / 15.90    | 87.80 / 3.30     | 78.45 / 11.83    |
| UNO               | 96.30 / 2.00     | 93.20 / 0.20     | 82.40 / 9.20     |
| Mask2Anomaly      | 88.70 / 14.60    | 93.30 / 0.20     | 79.70 / 13.45    |
| **Objectomaly**   | **96.64 / 0.62** | **96.99 / 0.07** | **87.19 / 9.92** |

### Component-Level Metrics (sIoU ↑ / PPV ↑ / F1-score ↑)

| Method            | SMIYC AT (sIoU / PPV / F1) | SMIYC OT (sIoU / PPV / F1) |
| ----------------- | -------------------------- | -------------------------- |
| RPL               | 49.77 / 29.96 / 30.16       | 52.62 / 56.65 / 56.69       |
| Maskomaly         | 55.40 / 51.50 / 49.90       | 57.82 / 75.42 / 68.15       |
| RbA               | 56.30 / 41.35 / 42.00       | 47.40 / 56.20 / 50.40       |
| UNO               | **68.01** / 51.86 / 58.87   | 66.87 / 74.86 / 76.32       |
| Mask2Anomaly      | 55.28 / 51.68 / 47.16       | 55.72 / 75.42 / 68.15       |
| **Objectomaly**   | 43.70 / **94.95** / **60.83** | **71.58 / 78.88 / 83.44**   |

---

## Qualitative Results

<p align="center">
  <img src="fig-src-dat-ra.png" alt="Qualitative Examples" width="700">
</p>

## Usage

**Step 1 : git clone**
```
git clone https://github.com/hon121215/Objectomaly.git
cd Objectomaly
```
**Step 2 : environment setup**

- See [installation instructions](INSTALL.md).

**Step 3 : Weight File Download**
After downloading the files below, modify the paths to each of them:

- [best_contrastive.pth](https://drive.usercontent.google.com/download?id=1TO8op0JvEhTzesbo3vcKbkmbPhcVmE47) → /Objectomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml 
- [bt-f-xl.pth](https://drive.usercontent.google.com/download?id=1FaZAKCsTxYE5KBOlRgSb6q3eWFtdkSvp) → /Objectomaly/configs/cityscapes/semantic-segmentation/anomaly_ft.yaml 
- [sam_vit_h_4b8939.pth](https://drive.usercontent.google.com/download?id=1ftcPwAs3zy5cD83Mhxoiw4kgjXmrZW39) → /Objectomaly/anomaly_utils/anomaly_inference.py 

**Step 4 : Dataset Download**

- You can download the anomaly segmentation datasets from the following [🔗link](https://drive.usercontent.google.com/download?id=1NL_ApRB-MjVRrMw6ONYZTe1azXc_71yQ&export=download&authuser=0)

**Dataset Structure**

```
datasets/
├── fs_static
│ ├── images
│ └── labels_masks
├── RoadAnomaly
│ ├── images
│ └── labels_masks
├── RoadAnomaly21
│ ├── images
│ └── labels_masks
├── RoadObstacle21
│ ├── images
│ └── labels_masks
```

After downloading, unzip the files and place them under the `Objectomaly/datasets/` directory.

**Step 5 : Inference**

- Set the argument path for run.sh and run it
```
sh run.sh
```

## Docker Image
**Coming soon**


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

## Acknowledgement

We gratefully acknowledge the following repositories that greatly inspired and supported the development of Objectomaly:

- [Mask2Anomaly](https://github.com/shyam671/Mask2Anomaly-Unmasking-Anomalies-in-Road-Scene-Segmentation)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
