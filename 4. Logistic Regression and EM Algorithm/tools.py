import numba as nb
import numpy as np

@nb.jit
def initialize():
    pi = np.full((10, 1), 0.1, dtype=np.float64)
    mu = np.random.rand(28 * 28, 10).astype(np.float64)
    mu_prev = np.zeros((28 * 28, 10), dtype=np.float64)
    Z = np.full((10, 60000), 0.1, dtype=np.float64)
    return pi, mu, mu_prev, Z
