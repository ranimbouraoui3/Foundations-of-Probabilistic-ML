# 02. Optimization Benchmarking & Architectural Scaling

This module focuses on **systems-level AI design**. It investigates the dynamics of model convergence and the trade-offs between architectural complexity, memory efficiency, and inference performance.

## 1. Solver Dynamics: SAGA vs. Adam
I conducted a comparative study of optimization paradigms across convex and non-convex loss landscapes:
*   **SAGA (Stochastic Average Gradient):** Benchmarked for its efficiency in high-dimensional sparse linear problems (Logistic Regression).
*   **Adam (Adaptive Moment Estimation):** Evaluated for its resilience in navigating the complex, non-convex landscapes of Multi-Layer Perceptrons (MLP).
*   **Convergence Monitoring:** Implemented "warm-start" training loops to monitor log-loss trajectories, identifying the precise inflection points where models reach global stability.

## 2. Transfer Learning & Hardware Scaling
Architectural benchmarking on the **CIFAR-10** dataset using **MobileNetV2** (lightweight/inverted residuals) vs. **ResNet50** (deep residual learning).

### Systems Engineering Highlights:
*   **Scalable Data Pipelines:** Engineered streaming pipelines using `tf.data` to transform static preprocessing into **dynamic, on-the-fly transformations**. This reduced peak RAM usage from **30GB to <2GB**.
*   **Mixed-Precision Training:** Utilized `mixed_float16` policies to optimize GPU throughput and reduce memory footprint.

### Comparative Results:
| Model | Parameter Count | Computational Cost | Validation Accuracy |
| :--- | :--- | :--- | :--- |
| **MobileNetV2** | ~2.2M | Low (Mobile-Ready) | 35.59% |
| **ResNet50** | ~23.5M | Moderate | 90.37% |

## Research Significance
Selecting the right backbone and optimizer is not a heuristic; it is an engineering decision based on hardware constraints and representational needs. This module proves the ability to design pipelines that maximize hardware utilization while maintaining high generalization accuracy.
---
**Technologies Used:**
*   **Frameworks:** TensorFlow/Keras, Scikit-learn
*   **Data Analysis:** Pandas, NumPy, Matplotlib
*   **Techniques:** Transfer Learning, Hyperparameter Tuning, Convergence Analysis
