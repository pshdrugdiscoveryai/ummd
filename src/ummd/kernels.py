import numpy as np

class GaussianKernel:
    def __init__(self, gamma):
        self.gamma = gamma
    def __call__(self, x, y):
        return np.exp(-self.gamma * np.linalg.norm(x - y) ** 2)