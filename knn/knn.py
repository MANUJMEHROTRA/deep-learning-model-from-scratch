import numpy as np
from collections import Counter


class KNearestNeighbours:
    """
    k-Nearest Neighbours for classification and regression.

    Parameters
    ----------
    k : int
        Number of nearest neighbours to consider.
    task : str
        'classification' uses majority vote; 'regression' uses mean of neighbours.
    distance : str
        Distance metric — 'euclidean' (L2) or 'manhattan' (L1).
    """

    def __init__(
        self,
        k: int = 3,
        task: str = "classification",
        distance: str = "euclidean",
    ):
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        if distance not in ("euclidean", "manhattan"):
            raise ValueError("distance must be 'euclidean' or 'manhattan'")
        self.k = k
        self.task = task
        self.distance = distance
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between X (n_query, d) and
        self._X_train (n_train, d).

        Returns (n_query, n_train) distance matrix.
        """
        if self.distance == "euclidean":
            # ||a - b||² = ||a||² + ||b||² - 2 a·b  (vectorised, avoids loops)
            sq_X = np.sum(X ** 2, axis=1, keepdims=True)           # (n_q, 1)
            sq_train = np.sum(self._X_train ** 2, axis=1)           # (n_train,)
            cross = X @ self._X_train.T                             # (n_q, n_train)
            dist_sq = sq_X + sq_train - 2 * cross
            # Clamp small negatives from floating-point errors
            return np.sqrt(np.maximum(dist_sq, 0))
        else:  # manhattan
            # Loop over query points — acceptable for small datasets
            return np.array(
                [np.sum(np.abs(self._X_train - x), axis=1) for x in X]
            )

    def _predict_single(self, distances: np.ndarray) -> float | int:
        k_idx = np.argsort(distances)[: self.k]
        k_labels = self._y_train[k_idx]
        if self.task == "classification":
            return Counter(k_labels).most_common(1)[0][0]
        return float(np.mean(k_labels))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighbours":
        """Store training data — KNN is a lazy learner, no real training."""
        self._X_train = np.array(X, dtype=float)
        self._y_train = np.array(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels / values for each row in X."""
        if self._X_train is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X = np.array(X, dtype=float)
        dist_matrix = self._compute_distances(X)
        return np.array([self._predict_single(row) for row in dist_matrix])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Accuracy for classification, R² for regression.
        """
        y_pred = self.predict(X)
        if self.task == "classification":
            return float(np.mean(y_pred == y))
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_res / ss_tot)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(7)

    # --- Classification demo ---
    n = 300
    X0 = rng.standard_normal((n // 2, 2)) + np.array([-2, -2])
    X1 = rng.standard_normal((n // 2, 2)) + np.array([2, 2])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])

    idx = rng.permutation(n)
    X, y = X[idx], y[idx]
    split = int(0.8 * n)

    clf = KNearestNeighbours(k=5, task="classification")
    clf.fit(X[:split], y[:split])
    print(f"[Classification] k=5 accuracy: {clf.score(X[split:], y[split:]):.4f}")

    # --- Regression demo ---
    Xr = rng.uniform(0, 2 * np.pi, (n, 1))
    yr = np.sin(Xr[:, 0]) + rng.standard_normal(n) * 0.2

    idx = rng.permutation(n)
    Xr, yr = Xr[idx], yr[idx]

    reg = KNearestNeighbours(k=7, task="regression")
    reg.fit(Xr[:split], yr[:split])
    print(f"[Regression]     k=7 R²:        {reg.score(Xr[split:], yr[split:]):.4f}")