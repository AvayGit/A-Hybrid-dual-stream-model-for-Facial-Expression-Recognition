# Dual-Stream Hybrid FER (ConvNeXt + ViT + GAT) 🚀

[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Coming_Soon-blue.svg)](#) Official PyTorch implementation of the research paper: **"A Dual-Stream Hybrid Architecture Integrating ConvNeXt and Graph-Attention Networks for State-of-the-Art Facial Expression Recognition."**

This repository contains the complete training engine, ablation study framework, and high-precision evaluation scripts for our SOTA Dual-Stream FER model.

## 🌟 Highlights
* **State-of-the-Art Performance:** Achieves **96.44% Overall Accuracy** and a 0.9643 Macro F1-Score on real-world facial expression data.
* **Modality Collapse Defense:** Implements strict pre-fusion `LayerNorm` isolation to ensure high-dimensional visual tensors (768D) do not overpower low-dimensional geometric graphs (128D).
* **Structural Geometry via GAT:** Replaces implicit spatial CNNs with explicit Graph Attention Networks (`MultiheadAttention`), dynamically correlating 6 primary facial muscle distances.
* **Super-Convergence Engine:** Utilizes Differential Learning Rates (protecting the ConvNeXt backbone from catastrophic forgetting), `AdamW`, and `CosineAnnealingWarmRestarts` to reach global optima in just 25 epochs.

---

## 🧠 Architecture Overview

The framework processes facial data through two parallel streams before executing a normalized fusion:

1.  **Stream A (Visual Texture):** A `ConvNeXt-Tiny` spatial prior feeds a 2-layer Vision Transformer (ViT). This allows the network to calculate global self-attention across 49 distinct facial patches.
2.  **Stream B (Geometric Topology):** 6 engineered muscle-distance vectors are modeled as a fully connected graph. A Graph Attention Network (GAT) computes dynamic correlation scores between these structural nodes.

<p align="center">
  <img src="path_to_your_architecture_image.png" alt="Architecture Diagram" width="800">
</p>

---

## 📊 Quantitative Results

Our model demonstrates unprecedented class separability, particularly on highly correlated micro-expressions such as Fear and Anger.

| Class | Precision | Recall | F1-Score | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Angry** | 0.99 | 0.99 | 0.99 | **99.05%** |
| **Disgust** | 0.95 | 0.99 | 0.97 | **98.95%** |
| **Fear** | 0.99 | 0.99 | 0.99 | **99.16%** |
| **Happy** | 0.97 | 0.94 | 0.96 | **94.26%** |
| **Neutral** | 0.93 | 0.93 | 0.93 | **92.52%** |
| **Sad** | 0.94 | 0.94 | 0.94 | **94.01%** |
| **Surprise** | 0.98 | 0.98 | 0.98 | **97.79%** |
| *Overall (N=3425)*| *0.96* | *0.96* | *0.96* | **96.44%** |

---

## ⚙️ Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/YourUsername/Dual-Stream-Hybrid-FER.git](https://github.com/YourUsername/Dual-Stream-Hybrid-FER.git)
cd Dual-Stream-Hybrid-FER
