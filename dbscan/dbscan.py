import numpy as np
from collections import deque


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Clusters are dense regions separated by lower-density regions. Points
    that do not belong to any cluster are labelled as noise (label = -1).

    Parameters
    ----------
    eps : float
        Radius of the neighbourhood around each point.
    min_samples : int
        Minimum number of points (including the point itself) required
        within eps to qualify as a core point.
    metric : str
        Distance metric — 'euclidean' (L2) or 'manhattan' (L1).
    """

    # Internal sentinel values
    _UNVISITED = -2
    _NOISE = -1

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
    ):
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        # Set after fit
        self.labels_: np.ndarray | None = None
        self.core_sample_indices_: np.ndarray | None = None
        self.n_clusters_: int = 0
        self.components_: np.ndarray | None = None   # core point vectors

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the full (n, n) pairwise distance matrix.

        Euclidean uses the identity ||a-b||² = ||a||² + ||b||² - 2 a·b
        to avoid an explicit Python loop.
        """
        if self.metric == "euclidean":
            sq = np.sum(X ** 2, axis=1, keepdims=True)   # (n, 1)
            dist_sq = sq + sq.T - 2.0 * (X @ X.T)        # (n, n)
            return np.sqrt(np.maximum(dist_sq, 0.0))
        else:  # manhattan
            # |a - b|_1 — vectorised via broadcasting (memory: O(n² d))
            return np.sum(np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :]), axis=2)

    def _region_query(self, dist_matrix: np.ndarray, idx: int) -> np.ndarray:
        """Return indices of all points within eps of point idx."""
        return np.where(dist_matrix[idx] <= self.eps)[0]

    def _expand_cluster(
        self,
        dist_matrix: np.ndarray,
        labels: np.ndarray,
        seed_idx: int,
        cluster_id: int,
        seed_neighbors: np.ndarray,
    ) -> None:
        """
        BFS expansion from a core point.

        Starting from seed_idx, follow density-reachability links until
        no more points can be added to cluster_id.
        """
        labels[seed_idx] = cluster_id
        queue: deque[int] = deque(seed_neighbors)

        while queue:
            point = queue.popleft()

            if labels[point] == self._NOISE:
                # Border point: add to cluster, do not expand further
                labels[point] = cluster_id
                continue

            if labels[point] != self._UNVISITED:
                # Already processed (either in this or another cluster)
                continue

            labels[point] = cluster_id
            neighbors = self._region_query(dist_matrix, point)

            if len(neighbors) >= self.min_samples:
                # point is a core point — add its unseen neighbors to queue
                for nb in neighbors:
                    if labels[nb] == self._UNVISITED or labels[nb] == self._NOISE:
                        queue.append(nb)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "DBSCAN":
        """
        Run DBSCAN on feature matrix X.

        X : (n_samples, n_features)
        """
        X = np.array(X, dtype=float)
        n_samples = X.shape[0]

        dist_matrix = self._pairwise_distances(X)
        labels = np.full(n_samples, self._UNVISITED, dtype=int)
        cluster_id = 0

        for idx in range(n_samples):
            if labels[idx] != self._UNVISITED:
                continue  # already processed

            neighbors = self._region_query(dist_matrix, idx)

            if len(neighbors) < self.min_samples:
                labels[idx] = self._NOISE
            else:
                cluster_id += 1
                self._expand_cluster(dist_matrix, labels, idx, cluster_id, neighbors)

        # Remap from 1-indexed cluster IDs to 0-indexed (noise stays -1)
        final_labels = np.where(labels == self._NOISE, -1, labels - 1)

        self.labels_ = final_labels
        self.n_clusters_ = cluster_id
        self.core_sample_indices_ = np.array(
            [i for i in range(n_samples) if len(self._region_query(dist_matrix, i)) >= self.min_samples],
            dtype=int,
        )
        self.components_ = X[self.core_sample_indices_]
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels. Noise points have label -1."""
        return self.fit(X).labels_


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(99)

    # Three dense circular blobs + scattered noise
    def make_blob(centre, n, spread):
        return rng.standard_normal((n, 2)) * spread + np.array(centre)

    X_blobs = np.vstack([
        make_blob((-4,  0), 100, 0.6),
        make_blob(( 0,  4), 100, 0.6),
        make_blob(( 4,  0), 100, 0.6),
    ])
    X_noise = rng.uniform(-7, 7, size=(30, 2))
    X = np.vstack([X_blobs, X_noise])

    # Shuffle
    idx = rng.permutation(len(X))
    X = X[idx]

    model = DBSCAN(eps=1.0, min_samples=5, metric="euclidean")
    labels = model.fit_predict(X)

    print(f"Number of clusters  : {model.n_clusters_}")
    print(f"Noise points        : {(labels == -1).sum()}")
    print(f"Core points         : {len(model.core_sample_indices_)}")

    for cid in range(model.n_clusters_):
        n = (labels == cid).sum()
        print(f"  Cluster {cid}: {n} points")

    # ── Ring-shaped cluster demo (shows advantage over K-Means) ──────────
    # Evenly spaced angles guarantee no gaps larger than eps
    angles_inner = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    angles_outer = np.linspace(0, 2 * np.pi, 150, endpoint=False)
    ring_inner = np.column_stack([np.cos(angles_inner) * 1.0, np.sin(angles_inner) * 1.0])
    ring_outer = np.column_stack([np.cos(angles_outer) * 3.0, np.sin(angles_outer) * 3.0])
    ring_inner += rng.standard_normal((120, 2)) * 0.08
    ring_outer += rng.standard_normal((150, 2)) * 0.08
    X_ring = np.vstack([ring_inner, ring_outer])

    ring_model = DBSCAN(eps=0.35, min_samples=4)
    ring_labels = ring_model.fit_predict(X_ring)

    print(f"\n[Ring demo] Clusters found : {ring_model.n_clusters_}  (expected 2)")
    print(f"[Ring demo] Noise points   : {(ring_labels == -1).sum()}")