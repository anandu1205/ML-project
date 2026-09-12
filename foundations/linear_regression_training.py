import numpy as np
from numpy.typing import NDArray


class Solution:

    learning_rate = 0.01

    def get_derivative(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
        N: int,
        X: NDArray[np.float64],
        desired_weight: int
    ) -> float:

        return -2 * np.dot(
            ground_truth - model_prediction,
            X[:, desired_weight]
        ) / N

    def get_model_prediction(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        return np.squeeze(np.matmul(X, weights))

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        weights = np.array(initial_weights, dtype=np.float64)

        for _ in range(num_iterations):

            # Forward pass
            prediction = self.get_model_prediction(X, weights)

            # Calculate and apply gradient for each weight
            for j in range(len(weights)):

                gradient = self.get_derivative(
                    prediction,
                    Y,
                    len(X),
                    X,
                    j
                )

                weights[j] -= self.learning_rate * gradient

        return np.round(weights, 5)
