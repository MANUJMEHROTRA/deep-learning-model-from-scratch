# Linear Regression — from scratch with NumPy

## Overview

Linear Regression is the simplest supervised learning algorithm. It models the relationship between one or more input features **X** and a continuous output **y** by fitting a straight line (or hyperplane in higher dimensions):

```
ŷ = X · w + b
```

where **w** are the learned weights and **b** is the bias term.

---

## Mathematical Foundation

### Hypothesis

Given an input vector **x** ∈ ℝᵈ:

```
ŷ = w₁x₁ + w₂x₂ + ... + wᵈxᵈ + b
```

In matrix form across all *n* training samples:

```
ŷ = Xw + b          X ∈ ℝⁿˣᵈ, w ∈ ℝᵈ, b ∈ ℝ
```

### Loss Function — Mean Squared Error (MSE)

```
MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²
```

MSE is convex with a unique global minimum, making it well-suited for gradient descent.

### Gradients

Differentiating MSE with respect to **w** and **b**:

```
∂L/∂w = (2/n) Xᵀ(ŷ - y)
∂L/∂b = (2/n) Σ(ŷ - y)
```

### Gradient Descent Update Rule

At each iteration *t*:

```
w ← w - α · ∂L/∂w
b ← b - α · ∂L/∂b
```

where **α** is the learning rate.

### Evaluation — R² (Coefficient of Determination)

```
R² = 1 - SS_res / SS_tot

SS_res = Σ(yᵢ - ŷᵢ)²      (residual sum of squares)
SS_tot = Σ(yᵢ - ȳ)²       (total sum of squares)
```

R² = 1 means a perfect fit; R² = 0 means the model is no better than predicting the mean.

---

## Implementation Details

File: [linear_regression.py](linear_regression.py)

| Component | Description |
|---|---|
| `fit(X, y)` | Runs gradient descent for `n_iterations` steps, records MSE at each step |
| `predict(X)` | Computes `Xw + b` |
| `score(X, y)` | Returns R² |
| `loss_history` | List of MSE values per iteration — useful for plotting convergence |

### Key Design Choices

- **Vectorised gradients** — all gradient computations use NumPy matrix operations rather than Python loops, making it fast even for larger datasets.
- **No explicit regularisation** — this is vanilla OLS. Adding L2 regularisation (Ridge) would require adding `λw` to the weight gradient.
- **Single learning rate** — adaptive methods (Adam, RMSProp) are not included to keep the implementation transparent.

---

## Step-by-Step Walkthrough

```
1. Initialise  w = 0, b = 0

2. For each iteration:
   a. Forward pass:   ŷ = Xw + b
   b. Compute error:  e = ŷ - y
   c. Compute grads:  dw = (2/n) Xᵀe,  db = (2/n) Σe
   d. Update params:  w -= α·dw,        b -= α·db
   e. Record MSE

3. Return fitted model
```

---

## Usage

```python
import numpy as np
from linear_regression import LinearRegression

# Create and train
model = LinearRegression(learning_rate=0.05, n_iterations=500)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(model.score(X_test, y_test))   # R²
print(model.loss_history[-1])         # Final MSE
```

Run the built-in demo:

```bash
python linear_regression.py
```

---

## Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `learning_rate` | `0.01` | Too high → divergence; too low → slow convergence |
| `n_iterations` | `1000` | More iterations → closer to optimum (assuming stable α) |

---

## Common Pitfalls

- **Feature scaling** — gradient descent converges much faster when features are standardised (zero mean, unit variance). Without scaling, a feature with range [0, 1000] will dominate the gradient.
- **Learning rate tuning** — if loss increases or oscillates, reduce α. If loss decreases painfully slowly, increase α.
- **Collinearity** — highly correlated features make `w` unstable. Use Ridge regression or PCA in that case.

---

## Complexity

| Phase | Time | Space |
|---|---|---|
| Training | O(n · d · T) | O(d) |
| Prediction | O(n · d) | O(d) |

where *n* = samples, *d* = features, *T* = iterations.