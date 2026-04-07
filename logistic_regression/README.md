# Logistic Regression — from scratch with NumPy

## Overview

Logistic Regression is a supervised classification algorithm that models the **probability** that an input belongs to a particular class. Despite the name it is a classifier, not a regressor. It squashes a linear combination of features through the **sigmoid** function to produce values in (0, 1).

---

## Mathematical Foundation

### Sigmoid (Logistic) Function

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

Properties:
- Output is always in (0, 1) — interpretable as a probability.
- Differentiable everywhere: σ'(z) = σ(z)(1 − σ(z)).
- Symmetric about z = 0: σ(0) = 0.5.

### Hypothesis

```
z  = Xw + b                     (linear score)
ŷ  = σ(z) = P(y=1 | x; w, b)   (predicted probability)
```

The predicted class is:

```
class = 1  if ŷ ≥ threshold (default 0.5)
        0  otherwise
```

### Loss Function — Binary Cross-Entropy

```
L = -(1/n) Σᵢ [ yᵢ log(ŷᵢ) + (1 − yᵢ) log(1 − ŷᵢ) ]
```

Binary cross-entropy is the negative log-likelihood of a Bernoulli distribution. It penalises confident wrong predictions much more than uncertain ones (log(0) → ∞).

### Gradients

A remarkable result: the gradients of cross-entropy + sigmoid simplify to the same form as linear regression:

```
∂L/∂w = (1/n) Xᵀ(ŷ − y)
∂L/∂b = (1/n) Σ(ŷ − y)
```

### Gradient Descent Update

```
w ← w − α · ∂L/∂w
b ← b − α · ∂L/∂b
```

---

## Implementation Details

File: [logistic_regression.py](logistic_regression.py)

| Component | Description |
|---|---|
| `fit(X, y)` | Runs gradient descent; records loss per iteration |
| `predict_proba(X)` | Returns P(y=1 \| x) for each sample |
| `predict(X)` | Thresholds probabilities → 0/1 labels |
| `score(X, y)` | Returns classification accuracy |

### Numerically Stable Sigmoid

The naive `1 / (1 + exp(-z))` overflows for large negative z. The implementation uses:

```python
σ(z) = 1 / (1 + e⁻ᶻ)       for z ≥ 0
σ(z) = eᶻ / (1 + eᶻ)        for z < 0
```

Both branches are mathematically identical but avoid overflow.

### Epsilon Clipping in Cross-Entropy

Probabilities are clipped to [ε, 1−ε] (ε = 1e-15) before taking the log, preventing `log(0)` = −∞.

---

## Step-by-Step Walkthrough

```
1. Initialise  w = 0, b = 0

2. For each iteration:
   a. Linear score:    z  = Xw + b
   b. Sigmoid:         ŷ  = σ(z)
   c. Error:           e  = ŷ − y
   d. Gradients:       dw = (1/n) Xᵀe,  db = (1/n) Σe
   e. Update:          w -= α·dw,         b -= α·db
   f. Record loss

3. Return fitted model
```

---

## Usage

```python
import numpy as np
from logistic_regression import LogisticRegression

model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
model.fit(X_train, y_train)

probs  = model.predict_proba(X_test)   # probabilities
labels = model.predict(X_test)         # 0/1 labels
acc    = model.score(X_test, y_test)   # accuracy
```

Run the built-in demo:

```bash
python logistic_regression.py
```

---

## Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `learning_rate` | `0.01` | Step size for gradient descent |
| `n_iterations` | `1000` | Training iterations |
| `threshold` | `0.5` | Decision boundary for class prediction |

**Changing the threshold** trades off precision vs. recall:
- Lower threshold → more positives predicted → higher recall, lower precision.
- Higher threshold → fewer positives predicted → lower recall, higher precision.

---

## Extension to Multi-class

For K > 2 classes, replace sigmoid with **softmax** and binary cross-entropy with **categorical cross-entropy**. Each class gets its own weight vector; the output is a probability distribution over K classes (One-vs-Rest or Softmax regression).

---

## Common Pitfalls

- **Feature scaling** — logistic regression with gradient descent converges faster when features are standardised.
- **Class imbalance** — accuracy is a misleading metric when classes are imbalanced. Use precision, recall, F1, or AUC-ROC instead.
- **Linear decision boundary** — logistic regression draws a linear boundary in feature space. Non-linear problems require feature engineering or a different model.

---

## Complexity

| Phase | Time | Space |
|---|---|---|
| Training | O(n · d · T) | O(d) |
| Prediction | O(n · d) | O(d) |

where *n* = samples, *d* = features, *T* = iterations.