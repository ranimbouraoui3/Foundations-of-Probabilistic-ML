# Probabilistic ML Foundations

This repository contains a series of foundational studies in Machine Learning, 
focusing on the transition from deterministic engineering to uncertainty-aware research. 
These projects serve as technical evidence for my application to the **AScI Summer Research Program.**

## Repository Structure

### [01. Regression & Variance Study](./01-Regression-Variance-Study)
*   **Focus:** Predictive variance and Bayesian intuition.
*   **Description:** Moving beyond standard Linear Regression to visualize model uncertainty using ensemble-based variance on housing data.

### [02. Uncertainty Quantification in Vision](./02-Uncertainty-Quantification-CV)
*   **Focus:** Model calibration and out-of-distribution (OOD) awareness.
*   **Description:** Implementation of a Dynamic CNN architecture in PyTorch. Explores softmax-based confidence thresholding to identify ambiguous samples.

### [03. Optimization & Scalability Benchmarks](./03-Optimization-Benchmarks)
*   **Focus:** Training dynamics and Transfer Learning.
*   **Description:** A comparative study of convergence rates across different optimizers (SAGA vs. Adam) and a benchmark of lightweight vs. deep architectures (MobileNet vs. ResNet).

## Technical Stack
*   **Languages:** Python
*   **Frameworks:** PyTorch, JAX (Foundations), TensorFlow/Keras
*   **Tools:** NumPy, Pandas, Matplotlib, OpenCV