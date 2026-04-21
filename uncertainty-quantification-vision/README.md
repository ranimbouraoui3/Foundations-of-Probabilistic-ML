# 03. Uncertainty Quantification & Robust Vision

This module implements a dynamic Computer Vision pipeline in **PyTorch**, focusing on the transition from deterministic "black-box" classification to **Uncertainty-Aware inference**.

## Scientific Motivation
Standard neural networks are prone to **overconfidence**, often assigning high probability to incorrect or Out-of-Distribution (OOD) samples. For safety-critical industrial applications, a model must be able to signal when an input lies far from the training distribution (Sim-to-Real gap).

## Key Technical Features
1.  **Dynamic Computational Graph:** Leveraged PyTorch's imperative execution to build a `DynamicCNN` that adapts its feature extraction path based on **input modality shifts** (Grayscale vs. RGB).
2.  **Model Calibration Analysis:** Implemented **Softmax Calibration** to assess predictive confidence. The system differentiates between "Certain" and "Ambiguous" samples based on probability density.
3.  **Softmax Rejection Pipeline:** Developed an inference wrapper that rejects samples where the maximum probability falls below a threshold (e.g., 85%), significantly increasing the reliability of the remaining predictions.
4.  **Stochastic Robustness Testing:** Evaluated model failure modes by injecting **synthetic Gaussian noise** into sensor streams, identifying the exact noise variance ($\sigma$) where the model's predictive capacity collapses.

## Methodology & Evaluation
*   **Architecture:** Custom Modular CNN with modality-specific entry layers.
*   **Optimization:** Adam with Cross-Entropy Loss.
*   **Metrics:** Beyond standard accuracy, I report **Rejection Rates** and **Certain-Set Accuracy** to provide a complete picture of model reliability.

## Interpretations
This study demonstrates that low-confidence predictions correlate strongly with noisy or OOD inputs. This serves as a foundational step toward more advanced **Probabilistic ML** (Bayesian Neural Networks or Gaussian Processes) used in high-stakes research and industrial Cyber-Physical Systems.
