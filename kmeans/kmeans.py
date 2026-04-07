import numpy as np


class KMeans:
    """
    K-Means clustering with k-means++ initialisation.

    Parameters
    ----------
    k : int
        Number of clusters.
    max_iterations : int
        Maximum number of assignment / update iterations.
    tol : float
        Convergence tolerance — stops early if centroid shift < tol.
    n_init : int
        Number of independent runs; best inertia is kept.
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        k: int = 3,
        max_iterations: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state: int | None = None,
    ):
        self.k = k
        self.max_iterations = max_iterations
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = float("inf")
        self.n_iter_: int = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_centroids_plusplus(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """K-means++ centroid initialisation."""
        n_samples = X.shape[0]
        first_idx = rng.integers(n_samples)
        centroids = [X[first_idx]]

        for _ in range(1, self.k):
            # Squared distances from each point to the nearest centroid
            sq_dists = np.array(
                [min(np.sum((x - c) ** 2) for c in centroids) for x in X]
            )
            probs = sq_dists / sq_dists.sum()
            cumulative = np.cumsum(probs)
            r = rng.random()
            idx = np.searchsorted(cumulative, r)
            centroids.append(X[idx])

        return np.array(centroids)

    def _assign_labels(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each sample to its nearest centroid (vectorised)."""
        # (n_samples, k) matrix of squared distances
        diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]   # (n, k, d)
        sq_dists = np.sum(diffs ** 2, axis=2)                        # (n, k)
        return np.argmin(sq_dists, axis=1)

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        inertia = 0.0
        for j in range(self.k):
            members = X[labels == j]
            if len(members) > 0:
                inertia += float(np.sum((members - centroids[j]) ** 2))
        return inertia

    def _run_once(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        """Single run of k-means. Returns (centroids, labels, inertia, n_iter)."""
        centroids = self._init_centroids_plusplus(X, rng)

        for iteration in range(1, self.max_iterations + 1):
            labels = self._assign_labels(X, centroids)

            new_centroids = np.array(
                [
                    X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                    for j in range(self.k)
                ]
            )

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < self.tol:
                break

        inertia = self._compute_inertia(X, labels, centroids)
        return centroids, labels, inertia, iteration

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Cluster data matrix X.

        X : (n_samples, n_features)
        """
        X = np.array(X, dtype=float)
        rng = np.random.default_rng(self.random_state)

        best_centroids: np.ndarray | None = None
        best_labels: np.ndarray | None = None
        best_inertia = float("inf")
        best_n_iter = 0

        for _ in range(self.n_init):
            centroids, labels, inertia, n_iter = self._run_once(X, rng)
            if inertia < best_inertia:
                best_centroids = centroids
                best_labels = labels
                best_inertia = inertia
                best_n_iter = n_iter

        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new samples to the nearest fitted centroid."""
        if self.centroids_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        return self._assign_labels(np.array(X, dtype=float), self.centroids_)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels for X."""
        self.fit(X)
        return self.labels_


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(21)

    # Three well-separated Gaussian blobs
    centres = [(-4, 0), (0, 4), (4, 0)]
    X = np.vstack([rng.standard_normal((100, 2)) + c for c in centres])
    true_labels = np.repeat([0, 1, 2], 100)

    model = KMeans(k=3, random_state=21)
    pred = model.fit_predict(X)

    # Cluster purity (permutation-invariant)
    from itertools import permutations

    def purity(true, pred, k):
        best = 0
        for perm in permutations(range(k)):
            mapped = np.array([perm[p] for p in pred])
            acc = np.mean(mapped == true)
            best = max(best, acc)
        return best

    print(f"Inertia        : {model.inertia_:.2f}")
    print(f"Iterations     : {model.n_iter_}")
    print(f"Cluster purity : {purity(true_labels, pred, 3):.4f}")
    print(f"Centroids      :\n{model.centroids_}")