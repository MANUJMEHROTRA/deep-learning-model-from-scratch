# Decision Tree — from scratch with NumPy

## Overview

A Decision Tree is a supervised learning algorithm that makes predictions by recursively partitioning the feature space using a series of binary questions. The result is a tree structure where:

- **Internal nodes** test a feature against a threshold.
- **Branches** correspond to the outcome of that test.
- **Leaf nodes** hold the final prediction (class label or numeric value).

Decision trees are the building block for powerful ensemble methods (Random Forest, Gradient Boosting).

---

## Mathematical Foundation

### Impurity Measures

The quality of a split is measured by how much it reduces **impurity** in the child nodes.

**Gini Impurity** (used for classification):

```
Gini(S) = 1 − Σₖ pₖ²
```

where pₖ = fraction of samples in S belonging to class k. A pure node (all one class) has Gini = 0; maximum impurity (uniform distribution) approaches 1 − 1/K.

**Variance / MSE** (used for regression):

```
MSE(S) = Var(y) = (1/n) Σᵢ (yᵢ − ȳ)²
```

### Information Gain (Impurity Reduction)

For a split that partitions set S into left subset S_L and right subset S_R:

```
Gain = Impurity(S) − [ |S_L|/|S| · Impurity(S_L) + |S_R|/|S| · Impurity(S_R) ]
```

The split that maximises Gain is chosen at each node.

### Leaf Predictions

| Task | Leaf Value |
|---|---|
| Classification | Majority class among samples at the leaf |
| Regression | Mean of target values at the leaf |

---

## The Recursive Splitting Algorithm

```
BuildTree(S, depth):
  if stopping_criterion(S, depth):
    return Leaf(predict(S))

  Find the (feature, threshold) pair that maximises Gain on S
  if no valid split exists:
    return Leaf(predict(S))

  S_L = { x ∈ S : x[feature] ≤ threshold }
  S_R = { x ∈ S : x[feature] >  threshold }

  left  = BuildTree(S_L, depth + 1)
  right = BuildTree(S_R, depth + 1)
  return InternalNode(feature, threshold, left, right)
```

### Stopping Criteria

| Criterion | Description |
|---|---|
| `max_depth` | Stop when the node is at this depth |
| `min_samples_split` | Stop if the node has fewer than this many samples |
| `min_samples_leaf` | Reject any split where a child would have fewer samples than this |
| Pure node | Stop if impurity = 0 (all labels identical) |
| No gain | Stop if no split improves impurity |

---

## Implementation Details

File: [decision_tree.py](decision_tree.py)

### Classes

| Class | Description |
|---|---|
| `_Node` | Internal data structure representing a node (internal or leaf) |
| `DecisionTree` | Public API: fit, predict, score |

### `DecisionTree` Methods

| Method | Description |
|---|---|
| `fit(X, y)` | Recursively builds the tree by calling `_build` |
| `predict(X)` | Traverses the tree for each sample |
| `score(X, y)` | Accuracy (classification) or R² (regression) |
| `get_depth()` | Returns the actual depth of the built tree |

### Split Candidate Generation

Rather than trying every unique value of a feature as a threshold, the implementation uses **midpoints between consecutive unique values**:

```python
thresholds = np.unique(X[:, feat])
midpoints  = (thresholds[:-1] + thresholds[1:]) / 2
```

This is equivalent in outcome to testing each value but reduces the number of thresholds evaluated.

### Feature Subsampling (`max_features`)

Setting `max_features` to a value less than the total number of features causes each node to consider only a random subset of features. This is the key mechanism in **Random Forests** — it decorrelates individual trees in the ensemble.

---

## Step-by-Step Walkthrough

```
fit(X, y):
  root = _build(X, y, depth=0)

_build(X, y, depth):
  1. Check stopping criteria → return leaf if met

  2. For each feature (or random subset if max_features set):
       For each midpoint threshold:
         Compute weighted child impurity
         Compute Gain = parent_impurity − weighted_child_impurity
         Track best (feature, threshold, gain)

  3. If best gain <= 0: return leaf

  4. Split X, y at (best_feature, best_threshold)

  5. node.left  = _build(X_left,  y_left,  depth+1)
     node.right = _build(X_right, y_right, depth+1)

  6. Return node

predict(X):
  For each sample x:
    Start at root
    While not at leaf:
      If x[node.feature] <= node.threshold: go left
      Else: go right
    Return leaf.value
```

---

## Usage

```python
import numpy as np
from decision_tree import DecisionTree

# Classification
clf = DecisionTree(task='classification', max_depth=6)
clf.fit(X_train, y_train)
labels = clf.predict(X_test)
print(clf.score(X_test, y_test))   # accuracy
print(clf.get_depth())              # actual tree depth

# Regression
reg = DecisionTree(task='regression', max_depth=5)
reg.fit(X_train, y_train)
values = reg.predict(X_test)
print(reg.score(X_test, y_test))   # R²
```

Run the built-in demo:

```bash
python decision_tree.py
```

---

## Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `task` | `'classification'` | Switches impurity measure and leaf prediction |
| `max_depth` | `None` | Maximum tree depth; `None` = grow until pure |
| `min_samples_split` | `2` | Minimum samples at a node to attempt a split |
| `min_samples_leaf` | `1` | Minimum samples each child must have after split |
| `max_features` | `None` | Number of features per split (None = all) |

### Controlling Overfitting

Decision trees are prone to overfitting (memorising training data). The primary controls are:

1. **`max_depth`** — the most impactful parameter. A depth-3 tree is often already competitive and interpretable.
2. **`min_samples_leaf`** — prevents leaves with very few noisy samples.
3. **`min_samples_split`** — prevents splitting tiny nodes.
4. **Pruning** — post-hoc removal of branches that don't improve validation performance (not implemented here; consider cost-complexity pruning).

---

## Advantages and Limitations

**Advantages:**
- No feature scaling required — splits are based on thresholds, not distances.
- Handles mixed feature types (categorical can be binarised).
- Interpretable — can be visualised and inspected.
- Non-linear decision boundaries.
- Fast prediction: O(depth) per sample.

**Limitations:**
- High variance — small changes in data can produce a very different tree.
- Greedy splits — the locally optimal split at each node is not globally optimal.
- Axis-aligned splits — cannot directly model diagonal decision boundaries without feature engineering.
- Best used as a component in ensembles (Random Forest, XGBoost) rather than alone.

---

## Complexity

| Phase | Time | Space |
|---|---|---|
| Training | O(n · d · n · log n) ≈ O(n² · d · log n) worst case | O(n) (recursive stack) |
| Prediction | O(depth) per sample | O(1) |

where *n* = samples, *d* = features. In practice much faster because splits reduce n rapidly.