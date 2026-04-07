# Machine Learning Models from Scratch

Pure NumPy implementations of seven foundational machine learning algorithms — no scikit-learn, no TensorFlow, no shortcuts. Each model is built from first principles to make the underlying mathematics transparent.

---

## Models

| # | Algorithm | Task | Key Idea |
|---|---|---|---|
| 1 | [Linear Regression](linear_regression/) | Regression | Minimise MSE via gradient descent |
| 2 | [Logistic Regression](logistic_regression/) | Classification | Sigmoid + binary cross-entropy |
| 3 | [k-Nearest Neighbours](knn/) | Classification & Regression | Majority vote / mean of k closest points |
| 4 | [K-Means Clustering](kmeans/) | Unsupervised Clustering | Iterative assignment and centroid update |
| 5 | [Decision Tree](decision_tree/) | Classification & Regression | Recursive binary splitting on impurity gain |
| 6 | [DBSCAN](dbscan/) | Unsupervised Clustering | Density-based cluster expansion with noise detection |
| 7 | [ANN via HNSW](hnsw/) | Approximate Nearest Neighbour | Hierarchical multi-layer navigable graph |

---

## Repository Structure

```
.
├── linear_regression/
│   ├── linear_regression.py
│   └── README.md
├── logistic_regression/
│   ├── logistic_regression.py
│   └── README.md
├── knn/
│   ├── knn.py
│   └── README.md
├── kmeans/
│   ├── kmeans.py
│   └── README.md
├── decision_tree/
│   ├── decision_tree.py
│   └── README.md
├── dbscan/
│   ├── dbscan.py
│   └── README.md
└── hnsw/
    ├── hnsw.py
    └── README.md
```

---

## Quick Start

No installation beyond NumPy and pandas is needed.

```bash
pip install numpy pandas
```

Run any model's built-in demo:

```bash
python linear_regression/linear_regression.py
python logistic_regression/logistic_regression.py
python knn/knn.py
python kmeans/kmeans.py
python decision_tree/decision_tree.py
python dbscan/dbscan.py
python hnsw/hnsw.py
```

---

## Model Summaries

### 1. Linear Regression

Models a continuous target as a weighted sum of features. Trained by minimising Mean Squared Error (MSE) with gradient descent.

```
ŷ = Xw + b        Loss = (1/n) Σ (y − ŷ)²
```

Evaluation metric: **R²** (coefficient of determination).

[Full details →](linear_regression/README.md)

---

### 2. Logistic Regression

Binary classifier that squashes a linear score through the sigmoid function to produce a probability.

```
ŷ = σ(Xw + b)     Loss = -(1/n) Σ [y log ŷ + (1−y) log(1−ŷ)]
```

Numerically stable sigmoid; supports adjustable decision threshold.

[Full details →](logistic_regression/README.md)

---

### 3. k-Nearest Neighbours

Lazy learner — stores all training data and defers computation to inference time. Supports Euclidean and Manhattan distances, classification (majority vote) and regression (mean).

Vectorised distance computation using the identity `||a−b||² = ||a||² + ||b||² − 2aᵀb`.

[Full details →](knn/README.md)

---

### 4. K-Means Clustering

Unsupervised algorithm that partitions data into k clusters by minimising within-cluster sum of squares (inertia). Uses **k-means++ initialisation** and multiple restarts to avoid poor local minima.

[Full details →](kmeans/README.md)

---

### 5. Decision Tree

Recursive binary splitting algorithm for classification (Gini impurity) and regression (MSE). Supports `max_depth`, `min_samples_split`, `min_samples_leaf`, and `max_features` (enabling use as a base learner in Random Forests).

[Full details →](decision_tree/README.md)

---

### 6. DBSCAN

Density-based clustering that discovers clusters of **arbitrary shape** and labels outliers as noise — no need to specify the number of clusters. Expands clusters via BFS from core points (points with ≥ `min_samples` neighbours within radius `eps`).

```
core point  → |N_eps(p)| ≥ min_samples
border point → within eps of a core point
noise        → label −1
```

Handles non-convex clusters (rings, crescents) that K-Means cannot separate.

[Full details →](dbscan/README.md)

---

### 7. ANN via HNSW

Approximate Nearest Neighbour search using a **Hierarchical Navigable Small World** graph — the algorithm powering Pinecone, Weaviate, pgvector, and FAISS. Achieves **O(log n)** average query time with >95% Recall@K.

The index is a multi-layer graph: upper layers act as long-range "highways", layer 0 holds all nodes with dense local links. Search greedy-descends from the top and does a full beam search at layer 0.

```
Recall@10  ef=10  → ~85%
Recall@10  ef=50  → ~96%
Recall@10  ef=200 → ~99%
```

[Full details →](hnsw/README.md)

---

## Design Principles

- **NumPy only** — all computation uses vectorised array operations; no ML libraries.
- **Consistent API** — every supervised model exposes `fit`, `predict`, and `score`.
- **Self-contained demos** — run each file directly to see the model train and evaluate on synthetic data.
- **Readable over clever** — code is written to match the mathematical description, not to squeeze out every last microsecond.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | All numerical computation |
| `pandas` | Data loading / manipulation (available, not required by core models) |