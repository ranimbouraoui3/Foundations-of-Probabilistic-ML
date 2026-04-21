# Foundations of Probabilistic Machine Learning

This repository serves as a research-oriented portfolio investigating the intersection of **Deep Learning Reliability**, **Uncertainty Quantification (UQ)**, and **Computational Efficiency**. 

Modern AI often suffers from **overconfidence** and **generalization gaps**. These studies focus on moving beyond point-estimate predictions toward systems that are "aware" of what they do not know—a critical requirement for safety-critical and industrial AI applications.

---

## 🔬 Core Research Modules

### 01. Uncertainty Quantification & Robust Vision
**Focus:** *Model Calibration and Out-of-Distribution (OOD) Awareness*

*   **Dynamic Computational Graphs:** Developed a `DynamicCNN` architecture in **PyTorch** that utilizes imperative execution to adapt feature extraction based on input modality (e.g., Grayscale vs. RGB shifts).
*   **Mitigating Overconfidence:** Implemented **Softmax Calibration** and confidence-based rejection thresholds to identify samples lying far from the training distribution.
*   **Noise Robustness:** Evaluated model failure thresholds by injecting **synthetic Gaussian noise** into sensor streams, identifying the point where distribution shifts cause catastrophic forgetting.

### 02. Optimization & Computational Scaling
**Focus:** *Hardware-Aware ML and Training Dynamics*

*   **Streaming Data Pipelines:** Engineered high-performance input pipelines using `tf.data`, utilizing **asynchronous prefetching** and parallel mapping to reduce memory overhead from 30GB to <2GB.
*   **Mixed-Precision Training:** Utilized `mixed_float16` policies to maximize **GPU throughput** and reduce the carbon footprint of deep vision training.
*   **Solver Benchmarking:** Conducted a comparative analysis of **SAGA vs. Adam** solvers, researching convergence rates in both convex (Logistic) and non-convex (MLP) loss surfaces.

### 03. Regression & Predictive Variance
**Focus:** *Bayesian Intuition and Error Characterization*

*   **Uncertainty Estimation:** Investigated the transition from deterministic linear regression to probabilistic modeling, focusing on **predictive variance** as a proxy for model confidence.
*   **Analytical vs. Iterative Solvers:** Implemented and benchmarked **Ordinary Least Squares (OLS)** against iterative Gradient Descent, analyzing the trade-offs between exact solutions and scalable approximations.

---

## 🛠 Technical Stack & Engineering Skills

*   **Frameworks:** PyTorch (Imperative/Dynamic graphs), TensorFlow 2.x (`tf.data`, Mixed Precision).
*   **Domain Expertise:** Computer Vision (OpenCV), Probabilistic ML, Distribution Shifts, Performance Optimization.

---

## 📖 Scientific Motivation

Standard "black-box" models are often unsuited for the real world because they cannot signal when they are likely to be wrong. By researching **Calibration**, **OOD detection**, and **Efficiency**, this repository aims to build a foundation for **Reliable AI** that can be safely deployed in industrial Cyber-Physical Systems (CPS).
