# Uncertainty Quantification & Dynamic Vision

This module implements a dynamic Computer Vision pipeline in **PyTorch**, focusing on the transition from deterministic classification to **Uncertainty-Aware** inference.

## Scientific Motivation
Standard neural networks are often overconfident in their predictions. For safety-critical applications (such as those researched in Prof. Solin's group), it is vital that a model can identify when an input lies far from the training distribution.

## Key Technical Features
1.  **Dynamic Computational Graph**: The `DynamicCNN` architecture uses PyTorch's imperative nature to adapt its feature extraction path based on input channel modality (Grayscale vs. RGB).
2.  **Model Calibration Analysis**: Implementation of a **Predictive Confidence Threshold**. The model identifies "Uncertain" samples by analyzing the Softmax probability distribution.
3.  **Ambiguity Handling**: By rejecting samples where the maximum probability is below a threshold (e.g., 85%), we significantly improve the reliability of the remaining predictions.

## Methodology
*   **Architecture**: Custom CNN with modular entry-layers for different modalities.
*   **Dataset**: MNIST (Handwritten digits).
*   **Optimization**: Adam optimizer with Cross-Entropy Loss.
*   **Evaluation**: Beyond standard accuracy, we report **Rejection Rates** and **Certain-Set Accuracy**.

## Interpretations
The results demonstrate that low-confidence predictions often correlate with ambiguous or noisy inputs. This study serves as a foundational step toward more advanced Probabilistic ML techniques, such as **Bayesian Neural Networks** or **Gaussian Processes**, which model uncertainty directly in the parameter space.