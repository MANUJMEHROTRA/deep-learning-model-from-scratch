import numpy as np
from collections import Counter


# ---------------------------------------------------------------------------
# Internal node / leaf representation
# ---------------------------------------------------------------------------

class _Node:
    """A node in the decision tree."""

    __slots__ = (
        "feature_idx",
        "threshold",
        "left",
        "right",
        "value",
        "impurity",
        "n_samples",
    )

    def __init__(
        self,
        feature_idx: int | None = None,
        threshold: float | None = None,
        left: "_Node | None" = None,
        right: "_Node | None" = None,
        value=None,
        impurity: float = 0.0,
        n_samples: int = 0,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value          # Leaf prediction (class or mean)
        self.impurity = impurity
        self.n_samples = n_samples

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------

class DecisionTree:
    """
    Decision Tree for classification (Gini impurity) and regression (MSE).

    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    max_depth : int | None
        Maximum tree depth. None means unlimited.
    min_samples_split : int
        Minimum samples required to split a node.
    min_samples_leaf : int
        Minimum samples required in each child after a split.
    max_features : int | None
        Number of features to consider at each split (None = all features).
    """

    def __init__(
        self,
        task: str = "classification",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | None = None,
    ):
        if task not in ("classification", "regression"):
            raise ValueError("task must be 'classification' or 'regression'")
        self.task = task
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root: _Node | None = None
        self.n_features_: int = 0
        self.classes_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Impurity / leaf-value helpers
    # ------------------------------------------------------------------

    def _gini(self, y: np.ndarray) -> float:
        n = len(y)
        if n == 0:
            return 0.0
        counts = np.bincount(y.astype(int))
        probs = counts / n
        return float(1.0 - np.sum(probs ** 2))

    def _mse(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        return float(np.var(y))

    def _impurity(self, y: np.ndarray) -> float:
        return self._gini(y) if self.task == "classification" else self._mse(y)

    def _leaf_value(self, y: np.ndarray):
        if self.task == "classification":
            return Counter(y.tolist()).most_common(1)[0][0]
        return float(np.mean(y))

    # ------------------------------------------------------------------
    # Split search
    # ------------------------------------------------------------------

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[int | None, float | None, float]:
        n_samples, n_features = X.shape
        best_feature, best_threshold, best_gain = None, None, -float("inf")
        parent_impurity = self._impurity(y)

        # Feature subsampling (for random forest compatibility)
        feature_indices = np.arange(n_features)
        if self.max_features is not None and self.max_features < n_features:
            feature_indices = np.random.choice(
                n_features, self.max_features, replace=False
            )

        for feat in feature_indices:
            thresholds = np.unique(X[:, feat])
            # Use midpoints between consecutive unique values as candidate splits
            if len(thresholds) == 1:
                continue
            midpoints = (thresholds[:-1] + thresholds[1:]) / 2

            for thr in midpoints:
                left_mask = X[:, feat] <= thr
                right_mask = ~left_mask

                n_left = left_mask.sum()
                n_right = right_mask.sum()

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]

                # Weighted impurity reduction (information gain)
                child_impurity = (
                    n_left / n_samples * self._impurity(y_left)
                    + n_right / n_samples * self._impurity(y_right)
                )
                gain = parent_impurity - child_impurity

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_threshold = thr

        return best_feature, best_threshold, best_gain

    # ------------------------------------------------------------------
    # Tree building
    # ------------------------------------------------------------------

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n_samples = len(y)
        impurity = self._impurity(y)

        # Stopping criteria
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or impurity == 0.0
        ):
            return _Node(value=self._leaf_value(y), n_samples=n_samples, impurity=impurity)

        feat, thr, gain = self._best_split(X, y)

        # No valid split found
        if feat is None or gain <= 0:
            return _Node(value=self._leaf_value(y), n_samples=n_samples, impurity=impurity)

        left_mask = X[:, feat] <= thr
        node = _Node(
            feature_idx=feat,
            threshold=thr,
            impurity=impurity,
            n_samples=n_samples,
        )
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        """
        Build the decision tree from training data.

        X : (n_samples, n_features)
        y : (n_samples,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.n_features_ = X.shape[1]
        if self.task == "classification":
            self.classes_ = np.unique(y)
        self.root = self._build(X, y, depth=0)
        return self

    def _traverse(self, x: np.ndarray, node: _Node):
        if node.is_leaf:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels / values for each row in X."""
        if self.root is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        X = np.array(X, dtype=float)
        return np.array([self._traverse(x, self.root) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy for classification, R² for regression."""
        y_pred = self.predict(X)
        if self.task == "classification":
            return float(np.mean(y_pred == y))
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_res / ss_tot)

    def get_depth(self) -> int:
        """Return the actual depth of the built tree."""
        def _depth(node: _Node | None) -> int:
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.root)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(3)

    # --- Classification demo (3-class) ---
    from itertools import product

    centres = [(-3, 0), (0, 3), (3, 0)]
    X_cls = np.vstack([rng.standard_normal((80, 2)) + np.array(c) for c in centres])
    y_cls = np.repeat([0, 1, 2], 80)
    idx = rng.permutation(len(y_cls))
    X_cls, y_cls = X_cls[idx], y_cls[idx]
    split = int(0.8 * len(y_cls))

    clf = DecisionTree(task="classification", max_depth=6)
    clf.fit(X_cls[:split], y_cls[:split])
    print(f"[Classification] Accuracy : {clf.score(X_cls[split:], y_cls[split:]):.4f}")
    print(f"[Classification] Depth    : {clf.get_depth()}")

    # --- Regression demo ---
    Xr = rng.uniform(0, 2 * np.pi, (200, 1))
    yr = np.sin(Xr[:, 0]) + rng.standard_normal(200) * 0.15
    idx = rng.permutation(200)
    Xr, yr = Xr[idx], yr[idx]

    reg = DecisionTree(task="regression", max_depth=5)
    reg.fit(Xr[:160], yr[:160])
    print(f"[Regression]     R²        : {reg.score(Xr[160:], yr[160:]):.4f}")
    print(f"[Regression]     Depth     : {reg.get_depth()}")