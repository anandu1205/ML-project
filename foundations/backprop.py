import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = np.dot(x, w) + b
        y_hat = 1.0 / (1.0 + np.exp(-z))

        error = y_hat - y_true
        sigmoid_deriv = y_hat * (1.0 - y_hat)
        delta = error * sigmoid_deriv

        dL_dw = np.round(delta * x, 5)
        dL_db = round(float(delta), 5)

        return (dL_dw, dL_db)