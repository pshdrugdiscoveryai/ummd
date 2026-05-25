from scipy.spatial.distance import cdist, pdist, squareform
import numpy as np

median_dist = lambda x, y: 1 / (np.median(cdist(x, y, metric='sqeuclidean')))
gaussian_kernel = lambda x, y, gamma: np.exp(-gamma * cdist(x, y, metric='sqeuclidean'))


def MMD(x, y, kernel_function=gaussian_kernel, gamma=None):
    if gamma is None:
        gamma = median_dist(x, y)

    def mean_kernel_matrix(x, y, gamma):
        return np.mean(kernel_function(x, y, gamma))
    
    return mean_kernel_matrix(x, x, gamma) + mean_kernel_matrix(y, y, gamma) - 2 * mean_kernel_matrix(x, y, gamma)

    
