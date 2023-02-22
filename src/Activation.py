import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    return np.max(0, x)

def softmax(x: np.ndarray) -> np.ndarray:
    sum_exp = np.sum(np.exp(x))
    return np.exp(x) / sum_exp

def linear(x: np.ndarray) -> np.ndarray:
    return x

