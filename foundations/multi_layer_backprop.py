import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # Forward pass
        z1 = x @ W1.T + b1          # pre-activation layer 1
        a1 = np.maximum(0, z1)       # ReLU activation
        z2 = a1 @ W2.T + b2          # output (predictions)
        loss = np.mean((z2 - y_true) ** 2)

        # Backward pass
        n = len(y_true) if y_true.ndim > 0 else 1
        dz2 = 2 * (z2 - y_true) / n  # dL/dz2
        dW2 = dz2.reshape(-1, 1) @ a1.reshape(1, -1)  # dL/dW2
        db2 = dz2                      # dL/db2

        da1 = dz2.reshape(1, -1) @ W2  # dL/da1
        da1 = da1.flatten()
        dz1 = da1 * (z1 > 0).astype(float)  # ReLU derivative
        dW1 = dz1.reshape(-1, 1) @ x.reshape(1, -1)  # dL/dW1
        db1 = dz1                      # dL/db1

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist(),
        }