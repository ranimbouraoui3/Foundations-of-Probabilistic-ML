# 01. Optimization Foundations & Linear Systems

This module explores the mathematical foundations of parameter estimation and the trade-offs between analytical and iterative optimization methods. Using the Ames Housing dataset, this study moves beyond high-level APIs to implement core learning algorithms from first principles.

## Key Research Highlights

*   **Iterative Optimization:** Developed a multi-parameter **Gradient Descent** solver from scratch. This involved deriving partial derivatives for the MSE loss function and implementing a manual update loop to navigate the loss surface.
*   **Analytical Solvers (OLS):** Implemented the **Ordinary Least Squares (OLS)** method using the Normal Equation ($a = (X^T X)^{-1} X^T y$). I utilized NumPy matrix algebra to benchmark the speed and precision of exact solutions against iterative convergence.
*   **Statistical Evaluation:** Conducted correlation analyses on high-dimensional features to determine predictive power. Utilized **RMSE (Root Mean Squared Error)** to evaluate the bias-variance tradeoff across training and test distributions.

## Research Significance
Understanding the delta between iterative solvers (essential for non-convex Deep Learning) and analytical solutions (optimal for linear systems) is a prerequisite for advanced machine learning research. This module provides the empirical foundation for understanding how data distribution impacts solver stability.
