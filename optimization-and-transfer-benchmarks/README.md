# Optimization & Scalability Benchmarks

This module focuses on the computational foundations of Deep Learning. Before moving to probabilistic modeling, it is essential to understand the dynamics of model convergence and the trade-offs between architectural complexity and inference efficiency.

## 1. Convergence Study: Adaptive vs. Stochastic Gradient Methods
**Source:** Comparative analysis of Logistic Regression and Multi-Layer Perceptrons (MLP).

In this study, I analyzed the convergence behavior of two distinct optimization paradigms:
*   **SAGA (Stochastic Average Gradient):** A solver used for its efficiency in high-dimensional sparse problems.
*   **Adam (Adaptive Moment Estimation):** The standard for deep learning, utilizing adaptive learning rates.

### Key Observations:
*   **Rate of Convergence:** While SAGA reaches a stable minimum faster for linear models, Adam demonstrates superior resilience in navigating the non-convex loss surfaces of deep MLPs.
*   **Warm-Start Dynamics:** I implemented a "warm-start" training loop to monitor the log-loss at every iteration, visualizing the plateauing effect as the model approaches global minima.

## 2. Transfer Learning: MobileNetV2 vs. ResNet50
**Source:** Architectural benchmarking on the CIFAR-10 dataset.

This project investigates the trade-offs between **parameter efficiency** and **representational power**, a critical consideration for scaling models in research.

### Benchmark Details:
*   **MobileNetV2:** Benchmarked as a lightweight architecture (inverted residuals) for mobile-efficient inference.
*   **ResNet50:** Benchmarked for its deep feature extraction capabilities via skip connections.
*   **Technique:** Both models were initialized with `ImageNet` weights. I froze the base convolutional layers and appended a custom classification head to evaluate feature reuse efficiency.

### Results:
| Model | Parameter Count | Training Speed | Validation Accuracy |
| :--- | :--- | :--- | :--- |
| MobileNetV2 | ~2.2M | High | X.X% |
| ResNet50 | ~23.5M | Moderate | Y.Y% |

## Research Significance
Understanding optimizer stability and architectural trade-offs is fundamental to the **AScI Project 4207**, particularly when dealing with **large-scale models** and **generative modeling (Diffusion)**. Efficient inference starts with selecting the right backbone and optimizer, and these benchmarks provide the empirical evidence for those decisions.

---
**Technologies Used:**
*   **Frameworks:** TensorFlow/Keras, Scikit-learn
*   **Data Analysis:** Pandas, NumPy, Matplotlib
*   **Techniques:** Transfer Learning, Hyperparameter Tuning, Convergence Analysis