import numpy as np
import pandas as pd


class LinearRegression:
    """
    Ordinary Least Squares Linear Regression via gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size for each gradient descent update.
    n_iterations : int
        Number of gradient descent iterations.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Train the model on feature matrix X and target vector y.

        X : (n_samples, n_features)
        y : (n_samples,)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            y_pred = self._forward(X)

            # Gradients of MSE w.r.t. weights and bias
            error = y_pred - y
            dw = (2 / n_samples) * X.T @ error
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            self.loss_history.append(self._mse(y, y_pred))

        return self

    def _forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted values for feature matrix X."""
        if self.weights is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        return self._forward(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination R²."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)

    # Synthetic dataset: y = 3x1 + 1.5x2 + noise
    n = 200
    X = rng.standard_normal((n, 2))
    y = 3 * X[:, 0] + 1.5 * X[:, 1] + rng.standard_normal(n) * 0.5

    # Train / test split
    split = int(0.8 * n)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression(learning_rate=0.05, n_iterations=500)
    model.fit(X_train, y_train)

    print(f"Learned weights : {model.weights}")
    print(f"Learned bias    : {model.bias:.4f}")
    print(f"R² on test set  : {model.score(X_test, y_test):.4f}")
    print(f"Final MSE       : {model.loss_history[-1]:.4f}")