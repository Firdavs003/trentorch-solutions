import numpy as np


def linear(input: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    t_weight = np.transpose(weight)
    if bias is not None:
        output = np.matmul(input, t_weight) + bias
    else:
        output = np.matmul(input, t_weight)
    return output
