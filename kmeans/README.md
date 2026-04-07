# K-Means Clustering — from scratch with NumPy

## Overview

K-Means is an **unsupervised** clustering algorithm. It partitions *n* data points into *k* non-overlapping clusters by iteratively assigning points to the nearest centroid and updating centroids to be the mean of their assigned points.

Unlike supervised methods, K-Means has no labels — it discovers structure in the data.

---

## Mathematical Foundation

### Objective Function (Inertia)

K-Means minimises the **within-cluster sum of squared distances** (WCSS), also called inertia:

```
J = Σⱼ₌₁ᵏ  Σ_{xᵢ ∈ Cⱼ}  ||xᵢ − μⱼ||²
```

where **μⱼ** is the centroid of cluster *j* and *Cⱼ* is the set of points assigned to it.

This objective is **NP-hard** to optimise globally, but the iterative algorithm finds a good local minimum.

### Algorithm Steps

```
1. Initialise k centroids  μ₁, ..., μₖ

2. Repeat until convergence:
   a. Assignment step:
      Assign each xᵢ to its nearest centroid:
      cᵢ = argmin_j  ||xᵢ − μⱼ||²

   b. Update step:
      Recompute each centroid as the mean of its assigned points:
      μⱼ = (1/|Cⱼ|) Σ_{xᵢ ∈ Cⱼ} xᵢ

3. Convergence: centroid shift < tolerance, or max iterations reached
```

### Convergence Guarantee

The algorithm is guaranteed to converge (inertia is non-increasing at every step), but not to a global minimum — it converges to a local minimum that depends on initialisation.

---

## K-Means++ Initialisation

Naïve random initialisation often leads to poor local minima. **K-Means++** (Arthur & Vassilvitskii, 2007) uses a smarter initialisation:

```
1. Choose the first centroid uniformly at random from X.

2. For each subsequent centroid:
   a. Compute d(xᵢ)² = min distance² from xᵢ to any existing centroid.
   b. Sample the next centroid with probability proportional to d(xᵢ)².

3. Repeat until k centroids are chosen.
```

This spreads the initial centroids apart, leading to better and more consistent solutions. It provides an O(log k) approximation guarantee on inertia.

---

## Implementation Details

File: [kmeans.py](kmeans.py)

| Component | Description |
|---|---|
| `fit(X)` | Runs `n_init` independent K-Means runs, keeps the best (lowest inertia) |
| `predict(X)` | Assigns new points to the nearest fitted centroid |
| `fit_predict(X)` | `fit` + return `labels_` |
| `centroids_` | Final centroid positions (k × d array) |
| `labels_` | Cluster assignment for each training point |
| `inertia_` | Final WCSS value |

### Vectorised Assignment

Assignment is computed without loops using broadcasting:

```python
diffs   = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (n, k, d)
sq_dist = np.sum(diffs**2, axis=2)                            # (n, k)
labels  = np.argmin(sq_dist, axis=1)                          # (n,)
```

### Multiple Restarts (`n_init`)

Each run starts from a different random initialisation. The run with the lowest inertia is returned. This is the same strategy used by scikit-learn (default `n_init=10`).

---

## Step-by-Step Walkthrough

```
For each of n_init independent runs:

  1. K-Means++ init → k initial centroids

  2. Loop (max_iterations):
     a. Assign each point to nearest centroid
     b. Recompute centroids as cluster means
     c. Compute centroid shift ||new − old||
     d. If shift < tol: stop early

  3. Record inertia

Return centroids / labels from the run with minimum inertia.
```

---

## Usage

```python
import numpy as np
from kmeans import KMeans

model = KMeans(k=3, random_state=42)
labels = model.fit_predict(X)

print(model.centroids_)   # (k, d) array of centroid positions
print(model.inertia_)     # final WCSS
print(model.n_iter_)      # iterations taken by best run

# Assign new points to clusters
new_labels = model.predict(X_new)
```

Run the built-in demo:

```bash
python kmeans.py
```

---

## Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `k` | `3` | Number of clusters. Must be specified in advance |
| `max_iterations` | `300` | Maximum assignment/update cycles per run |
| `tol` | `1e-4` | Early stopping when centroid shift falls below this |
| `n_init` | `10` | Number of independent restarts |
| `random_state` | `None` | Seed for reproducibility |

### Choosing k — The Elbow Method

Plot inertia vs. k. Inertia decreases monotonically; the "elbow" where the rate of decrease sharply slows suggests the best k:

```
k=1: high inertia
k=2: drops sharply
k=3: still drops significantly   ← elbow here (if 3 true clusters)
k=4: marginal improvement
...
```

The **silhouette score** is another metric: higher is better, ranges from −1 to 1.

---

## Limitations

- **k must be specified** in advance. If unknown, use the elbow method, silhouette analysis, or algorithms like DBSCAN or GMM that infer the number of clusters.
- **Assumes spherical clusters** — K-Means uses Euclidean distance, so it naturally fits roughly spherical, equally-sized clusters. It struggles with elongated, irregular, or very different-sized clusters.
- **Sensitive to outliers** — outliers pull centroids away from the true cluster centres. Consider removing outliers or using K-Medoids.
- **Finds local minima** — multiple restarts (`n_init`) mitigate this.

---

## Complexity

| Phase | Time | Space |
|---|---|---|
| One iteration | O(n · k · d) | O(n · k · d) |
| Full training | O(n · k · d · T · n_init) | O(n · d + k · d) |

where *n* = samples, *k* = clusters, *d* = features, *T* = iterations.