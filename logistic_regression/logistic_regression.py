import numpy as np
import pandas as pd


class LogisticRegression:
    """
    Binary Logistic Regression trained with gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for each gradient descent update.
    n_iterations : int
        Number of gradient descent iterations.
    threshold : float
        Decision threshold for converting probabilities to class labels.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        threshold: float = 0.5,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.threshold = threshold
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def _binary_cross_entropy(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Train the model on feature matrix X and binary label vector y.

        X : (n_samples, n_features)
        y : (n_samples,)  — values must be 0 or 1
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            z = X @ self.weights + self.bias
            y_pred = self._sigmoid(z)

            # Gradients of binary cross-entropy
            error = y_pred - y
            dw = (1 / n_samples) * X.T @ error
            db = (1 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.loss_history.append(self._binary_cross_entropy(y, y_pred))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for class 1."""
        if self.weights is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels (0 or 1)."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return classification accuracy."""
        return float(np.mean(self.predict(X) == y))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Synthetic linearly separable dataset
    n = 300
    X0 = rng.standard_normal((n // 2, 2)) + np.array([-1.5, -1.5])
    X1 = rng.standard_normal((n // 2, 2)) + np.array([1.5, 1.5])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n // 2), np.ones(n // 2)])

    # Shuffle
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]

    split = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)

    print(f"Learned weights  : {model.weights}")
    print(f"Learned bias     : {model.bias:.4f}")
    print(f"Accuracy (test)  : {model.score(X_test, y_test):.4f}")
    print(f"Final loss       : {model.loss_history[-1]:.4f}")