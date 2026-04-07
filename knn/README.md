# k-Nearest Neighbours (KNN) — from scratch with NumPy

## Overview

k-Nearest Neighbours is a **non-parametric, lazy learning** algorithm. It makes no assumptions about the underlying data distribution and defers all computation to prediction time. It works for both classification and regression.

**Core idea**: to predict for a new point, find the *k* training points closest to it and aggregate their labels.

---

## Mathematical Foundation

### Distance Metrics

**Euclidean (L2) distance** — the default:

```
d(a, b) = √( Σᵢ (aᵢ − bᵢ)² )
```

**Manhattan (L1) distance**:

```
d(a, b) = Σᵢ |aᵢ − bᵢ|
```

L1 is more robust to outliers; L2 penalises large individual differences more heavily.

### Prediction

Given a query point **x**:

1. Compute distance from **x** to every training point.
2. Select the *k* points with smallest distance: the **neighbourhood** N_k(x).
3. Aggregate:

**Classification** — majority vote:
```
ŷ = argmax_c  |{xᵢ ∈ N_k(x) : yᵢ = c}|
```

**Regression** — mean of neighbours:
```
ŷ = (1/k) Σ_{xᵢ ∈ N_k(x)} yᵢ
```

---

## Implementation Details

File: [knn.py](knn.py)

| Component | Description |
|---|---|
| `fit(X, y)` | Stores training data (no computation) |
| `predict(X)` | Computes distance matrix, finds k-NN, aggregates |
| `score(X, y)` | Accuracy (classification) or R² (regression) |

### Vectorised Distance Computation (Euclidean)

Pairwise squared Euclidean distance is computed without explicit loops using the identity:

```
||a − b||² = ||a||² + ||b||² − 2(a · b)
```

This reduces an O(n_query · n_train · d) loop to efficient NumPy matrix operations:

```python
sq_X     = sum(X²,     axis=1, keepdims=True)   # (n_q, 1)
sq_train = sum(X_tr²,  axis=1)                  # (n_train,)
cross    = X @ X_train.T                          # (n_q, n_train)
dist_sq  = sq_X + sq_train - 2 * cross           # (n_q, n_train)
```

---

## Step-by-Step Walkthrough

```
Training (fit):
  Store X_train and y_train — no computation.

Prediction for a single query point x:
  1. Compute d(x, xᵢ)  for all xᵢ ∈ X_train
  2. Sort distances ascending
  3. Take indices of top-k smallest distances
  4. Retrieve y_train[top-k indices]
  5. Classification → majority vote
     Regression    → mean
```

---

## Usage

```python
import numpy as np
from knn import KNearestNeighbours

# Classification
clf = KNearestNeighbours(k=5, task='classification', distance='euclidean')
clf.fit(X_train, y_train)
labels = clf.predict(X_test)
print(clf.score(X_test, y_test))   # accuracy

# Regression
reg = KNearestNeighbours(k=7, task='regression')
reg.fit(X_train, y_train)
values = reg.predict(X_test)
print(reg.score(X_test, y_test))   # R²
```

Run the built-in demo:

```bash
python knn.py
```

---

## Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `k` | `3` | Number of neighbours. Small k → complex boundary (overfitting); large k → smoother boundary (underfitting) |
| `task` | `'classification'` | Switches aggregation between majority vote and mean |
| `distance` | `'euclidean'` | Distance metric used for neighbour selection |

### Choosing k

- **k = 1**: Each point is its own nearest neighbour on the training set → training accuracy = 100%, but very high variance.
- **k = n**: Always predicts the majority class → maximum bias, zero variance.
- **Cross-validation** is the standard way to select k.

---

## The Curse of Dimensionality

In high dimensions, all points become roughly equidistant from a query point. This makes "nearest" neighbours uninformative. KNN degrades significantly beyond ~20 dimensions. Mitigations:

- Dimensionality reduction (PCA, t-SNE) before applying KNN.
- Feature selection to remove irrelevant dimensions.
- Distance metric learning.

---

## Common Pitfalls

- **Feature scaling is critical** — a feature with range [0, 1000] will completely dominate Euclidean distance over a feature with range [0, 1]. Always standardise features.
- **Prediction cost** — O(n · d) per query point. For large datasets, use approximate nearest neighbour structures (KD-trees, ball trees, FAISS).
- **Ties** — when multiple classes have the same vote count, the winner depends on implementation. Consider k that avoids ties for binary problems (use odd k).

---

## Complexity

| Phase | Time | Space |
|---|---|---|
| Training | O(1) | O(n · d) |
| Prediction (brute-force) | O(n_query · n_train · d) | O(n_query · n_train) |

where *n* = training samples, *d* = features.