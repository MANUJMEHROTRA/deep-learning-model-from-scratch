# DBSCAN — from scratch with NumPy

## Overview

**Density-Based Spatial Clustering of Applications with Noise (DBSCAN)** is an unsupervised clustering algorithm that discovers clusters of arbitrary shape by identifying dense regions of points separated by sparser areas. Unlike K-Means it does **not** require specifying the number of clusters in advance and naturally identifies **noise/outlier** points.

---

## Core Concepts

DBSCAN classifies every point into exactly one of three roles:

| Role | Definition |
|---|---|
| **Core point** | Has ≥ `min_samples` neighbours within radius `eps` (including itself) |
| **Border point** | Within `eps` of a core point, but has < `min_samples` neighbours of its own |
| **Noise point** | Not within `eps` of any core point — labelled **−1** |

---

## Mathematical Foundation

### Neighbourhood

```
N_eps(p) = { q ∈ D : dist(p, q) ≤ eps }
```

### Core Point Condition

```
p is a core point  ⟺  |N_eps(p)| ≥ min_samples
```

### Density-Reachability

Point **q** is *directly density-reachable* from **p** if:

```
q ∈ N_eps(p)   AND   p is a core point
```

Point **q** is *density-reachable* from **p** if there exists a chain:

```
p = p₁ → p₂ → ... → pₙ = q
```

where each consecutive pair is directly density-reachable.

### Density-Connectivity

Points **p** and **q** are *density-connected* if there exists a point **o** from which **both** p and q are density-reachable. A cluster is a maximal set of mutually density-connected points.

---

## Algorithm

```
DBSCAN(D, eps, min_samples):
  labels ← UNVISITED for all points
  cluster_id ← 0

  for each point P in D:
    if P is already visited: continue

    N ← N_eps(P)                         // region query

    if |N| < min_samples:
      labels[P] ← NOISE                  // tentative noise
    else:
      cluster_id ← cluster_id + 1
      EXPAND_CLUSTER(P, N, cluster_id)

EXPAND_CLUSTER(P, N, cluster_id):
  labels[P] ← cluster_id
  queue ← N                              // BFS frontier

  while queue not empty:
    Q ← dequeue(queue)

    if labels[Q] == NOISE:
      labels[Q] ← cluster_id            // border point
      continue

    if labels[Q] ≠ UNVISITED: continue  // already processed

    labels[Q] ← cluster_id
    N' ← N_eps(Q)
    if |N'| ≥ min_samples:              // Q is a core point
      queue ← queue ∪ N'               // expand further
```

The BFS (breadth-first search) expansion guarantees that every density-reachable point is discovered from each core point.

---

## Implementation Details

File: [dbscan.py](dbscan.py)

| Component | Description |
|---|---|
| `_pairwise_distances(X)` | Full (n × n) distance matrix — vectorised for Euclidean |
| `_region_query(dist_matrix, idx)` | Returns indices of all points within `eps` of point `idx` |
| `_expand_cluster(...)` | BFS expansion from a core point |
| `fit(X)` | Main loop: classifies all points and assigns cluster labels |
| `fit_predict(X)` | Convenience: `fit` + return `labels_` |

### Attributes After Fitting

| Attribute | Description |
|---|---|
| `labels_` | Cluster label per point (−1 = noise, 0..K−1 = cluster id) |
| `core_sample_indices_` | Indices of all core points |
| `components_` | Feature vectors of core points |
| `n_clusters_` | Number of clusters found |

### Vectorised Euclidean Distance

```python
sq   = sum(X², axis=1, keepdims=True)        # (n, 1)
dist = sqrt(sq + sq.T − 2 · X @ X.T)         # (n, n)
```

This avoids explicit Python loops, reducing the O(n² d) computation to pure NumPy operations.

---

## Step-by-Step Walkthrough

```
1. Compute (n × n) pairwise distance matrix.

2. Initialise all labels as UNVISITED.

3. For each point P:
   a. If already labelled: skip.
   b. Find N_eps(P).
   c. If |N_eps(P)| < min_samples: label P as NOISE.
   d. Else: increment cluster_id, BFS-expand from P.

4. BFS expansion:
   - Label P with cluster_id.
   - For each neighbour Q in the queue:
       * If Q was NOISE: make it a border point (re-label).
       * If Q is UNVISITED: label it, check if it is a core point,
         and if so add its neighbours to the queue.
       * If Q already has a cluster label: skip.
```

---

## Usage

```python
import numpy as np
from dbscan import DBSCAN

model = DBSCAN(eps=1.0, min_samples=5, metric='euclidean')
labels = model.fit_predict(X)

print(model.n_clusters_)            # number of clusters found
print((labels == -1).sum())         # noise points
print(model.core_sample_indices_)   # indices of core points
```

Run the built-in demo:

```bash
python dbscan.py
```

---

## Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `eps` | `0.5` | Neighbourhood radius. Too small → everything is noise; too large → everything merges into one cluster |
| `min_samples` | `5` | Density threshold. Higher → fewer, denser clusters; lower → more clusters, less noise |
| `metric` | `'euclidean'` | Distance function (`'euclidean'` or `'manhattan'`) |

### Choosing `eps` — the k-distance plot

1. For each point, compute distance to its k-th nearest neighbour (k = min_samples − 1).
2. Sort and plot these distances.
3. Look for the "elbow" — this is a good eps value.

### Choosing `min_samples`

A common rule of thumb: `min_samples ≥ D + 1` where D is the number of features. For noisy data, larger values work better.

---

## Why DBSCAN Beats K-Means on Non-Convex Data

K-Means assumes spherical, equally-sized clusters (Voronoi cells). DBSCAN makes no such assumption:

```
K-Means on rings → fails (assigns half of each ring to wrong cluster)
DBSCAN on rings  → succeeds (follows density contours exactly)
```

---

## Limitations

- **Quadratic memory and time** — the pairwise distance matrix is O(n²). For large n, spatial indexes (KD-tree, ball tree) are used to make region queries O(log n), bringing total complexity to O(n log n).
- **Single global eps** — DBSCAN cannot handle clusters with very different densities. HDBSCAN (hierarchical extension) addresses this.
- **Sensitivity to hyperparameters** — choosing eps and min_samples requires domain knowledge or the k-distance plot.
- **High dimensions** — as with KNN, the curse of dimensionality makes density estimation unreliable beyond ~20 dimensions.

---

## Complexity

| Phase | Time | Space |
|---|---|---|
| Distance matrix (naïve) | O(n² · d) | O(n²) |
| BFS expansion | O(n²) worst case | O(n) |
| With spatial index | O(n log n) | O(n) |

where *n* = samples, *d* = features.