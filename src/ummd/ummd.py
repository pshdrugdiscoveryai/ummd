"""
UMMD: Unique Maximum Mean Discrepancy

"""

import time
from tracemalloc import start
from scipy.spatial.distance import cdist, pdist
import numpy as np


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        wrapper.time_taken = end - start
        return res
    return wrapper


@timer
def kernel_matrix(x, y, gammas):
    """
    Calculates the kernel distance matrix using a gaussian kernel with given bandwidths.

    args: 
        x: np.array of distribution x with m values, of shape [m, d] where d is the dimensions of a joint distribution
        y: np.array of distribution x with n values, of shape [n, d] where d is the dimensions of a joint distribution
        gammas: np.array of internal RBF precisions (gamma = 1/(2*sigma**2)) of shape [bandwidths, ]

    returns:
        K: np.array matrix of shape [bandwidths, m, n]
    """
    assert isinstance(gammas, np.ndarray) and gammas.ndim == 1, "Gammas must be a 1D array of bandwidths."


    D = cdist(x, y, metric='sqeuclidean')                # [m, n]
    K = np.exp(-gammas[:, None, None] * D[None, :, :])   # [bandwidths, m, n]
    return K                                      



@timer
def calc_MMD(K: np.array, s: np.array):
    """
    Calculates the biased MMD statistic given a kernel distance matrix and a sample weighting vector.

    args:
        K: Kernel distance matrix of shape [bandwidths, m, n]
        s: Sample weighting vector of shape [m + n, ] representing class frequency (1/m) and negated indexes from y (-1/n)

    returns: res: np.array
        MMD values for each tested bandwidth of shape [bandwidths, ]
    """
    return (s @ K @ s.T)



def perm_MMD(K, s, rng, n_permutations=999):
    """
    Calculates the biased MMD statistic for n_permutations given a kernel distance matrix and permuting a sample weighting vector.

    args:
        K: Kernel distance matrix of shape [bandwidths, m, n]
        s: Sample weighting vector of shape [m + n, ] representing class frequency (1/m) and negated indexes from y (-1/n)

    returns: res: np.array
        MMD values for each tested bandwidth of shape [n_permutations, bandwidths]
    """
    S = np.repeat(s[np.newaxis, :], repeats=n_permutations, axis=0) # [permutations, m + n]
    S = rng.permuted(S, axis=1)                                  
    perms = np.sum((S @ K) * S, 2)                                  # [bandwidths, permutations]
    return np.moveaxis(perms, 1, 0)                                 # [permutations, bandwidths]


def perm_uMMD(K, x_idx, y_idx, rng, n_permutations=999):
    """
    Calculates the biased MMD statistic for n_permutations given a kernel distance matrix and permuting a sample weighting vector.

    args:
        K: Kernel distance matrix of shape [bandwidths, m, n]
        xy_idx: Sample index vector of position in unique measurement matrix, shape [m + n, ] representing class frequency (1/m) and negated indexes from y (-1/n)

    returns: res: np.array
        MMD values for each tested bandwidth of shape [n_permutations, bandwidths]
    """

    xy_idx = np.concatenate((x_idx, y_idx))
    m = len(x_idx)
    n = len(y_idx)
    u = K.shape[-1]

    S = np.repeat(xy_idx[np.newaxis, :], repeats=n_permutations, axis=0)                    # [permutations, m + n]
    S = rng.permuted(S, axis=1)
    X = S[:, :m]                                                                            # [permutations, m]                                                                         # [permutations, n]
    
    # Vectorising bincount requires an offset trick. Add a new u index for each permutation and then bincount that.
    # Reshape that back to the original dimensions and you get the counts of the unique indexes for each permutation.
    # U_y can be easily calculated per permutation since U_x + U_y must = U_xy (the total counts of each unique value across x and y, which doesn't change with permutation).
    U_xy = np.bincount(xy_idx, minlength=u)                                 # [u, ]
    offsets = np.arange(n_permutations)[:, None] * u                        # [permutations, 1]
    U_x = np.bincount((X + offsets).ravel(),                                # [permutations * m, ]
                  minlength=n_permutations * u).reshape(n_permutations, u)  # [permutations, u]
    U_y = U_xy - U_x                                                        # [permutations, u]                     

    U = np.divide(U_x, m) - np.divide(U_y, n)                               # [permutations, u]

    perms = np.sum((U @ K) * U, 2)                                          # [bandwidths, permutations]
    return np.moveaxis(perms, 1, 0)                                         # [permutations, bandwidths]



def get_bandwidths(xy, n=10):
    """
    Generates a geometric grid of n sigma length-scales spanning the range of
    pairwise Euclidean distances in the pooled sample xy.

    args:
        xy: np.array of pooled samples of shape [m + n, d].
        n: (int) number of bandwidths to generate.

    returns:
        sigmas: np.array of sigma length-scales of shape [n, ].
    """
    D = pdist(xy, 'euclidean')
    lambda_min, lambda_max = D.min(), D.max()
    t = np.arange(n) / (n - 1)
    sigmas = (lambda_min / 2) * ((2 * lambda_max) / (lambda_min / 2)) ** t
    return sigmas


def cauchy_combination(p_vals, weight_distribution='uniform'):

    p_vals = np.clip(p_vals, 1e-30, 1 - 1e-30) # Avoid extreme p-values that can cause numerical issues

    norm = lambda x: x / np.sum(x)
    match weight_distribution:
        case 'uniform':
            w = norm(np.ones(len(p_vals)))
        case 'left':
            w = norm(1/np.arange(1, len(p_vals)+1))
        case 'right':
            w = norm(1/np.arange(len(p_vals), 0, -1))
        case 'centered':
            mid = (len(p_vals) - 1) / 2 
            w = norm(np.exp(-0.5 * ((np.arange(len(p_vals)) - mid) / (mid / 2)) ** 2))

    # Cauchy combination formula        
    T = np.sum(w * np.tan((0.5 - p_vals) * np.pi))
    cauchy_p = 0.5 - (np.arctan(T) / np.pi)
    return cauchy_p


@timer
def generate_ummd_input(x, y):
    """
    """
    unique_values, inverse = np.unique(np.concatenate((x, y), axis=0), axis=0, return_inverse=True)
    x_idx = inverse[:len(x)]
    y_idx = inverse[len(x):]
    return unique_values, x_idx, y_idx



@timer
def MMD(x, y, unique=True, bandwidths=None, n_permutations=999, perm_batch_size=999, cauchy_weighting="centered", seed=11):
    """
    Calculates the MMD of two distributions with optional p_values.

    args:
        x: np.array of distribution x with m values, of shape [m, d] where d is the dimensions of a joint distribution.
        y: np.array of distribution x with n values, of shape [n, d] where d is the dimensions of a joint distribution.
        unique: (bool) whether to use the unique value optimisation, which can be much faster for discrete data with many repeated values. Default: True.
        bandwidths: kernel bandwidths as sigma length-scales (same units as the data). One of:
            None or "median" -> median heuristic (median pairwise Euclidean distance of the pooled unique sample);
            int -> generate that many bandwidths spanning the pooled pairwise distances (see get_bandwidths);
            1D np.array -> the sigma values to test.
            Each sigma is converted internally to an RBF gamma via gamma = 1 / (2 * sigma**2),
            i.e. the kernel is k(x, y) = exp(-||x - y||**2 / (2 * sigma**2)). Default: None.
        n_permuations: (int) number of permutations to approximate p-value. Default: 999.
        perm_batch_size: (int) number of permutations to calculate in each batch. Default: 999.
        cauchy_weighting: If testing multiple bandwidths use a cauchy correction to aggregate into a single p-value; 
            weighting options ["centered", "uniform", "left", "right", None]. 
            Default: "centered", None will return p-values per bandwidth.
        seed: (int) random seed for reproducibility. Default: 11.

    returns:
        results object with attributes:
        'bandwidths': bandwidths for the RBF kernel.
        'n_permutations': number of permutations used to approximate p-value.
        'biased_MMD': MMD statistic per bandwidth.
        'p-values_per_bandwidth': permuation derived p-values for each bandwidth tested.
        'cauchy_method': method used for cauchy combination.
        'p-value': cauchy adjusted p-value.
    """

    # Check for 2d array
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]


    m = len(x)
    n = len(y)

    xy = np.concatenate((x, y), axis=0)     # [(m + n), d]

    # Resolve bandwidths
    if bandwidths is None or bandwidths == "median" or bandwidths <= 1:
        bandwidths = np.array([np.median(pdist(np.unique(xy, axis=0), metric='euclidean'))])
    elif isinstance(bandwidths, (int, np.integer)):
        bandwidths = get_bandwidths(np.unique(xy, axis=0), n=bandwidths)
    elif isinstance(bandwidths, np.ndarray):
        assert bandwidths.ndim == 1, "Bandwidths must be a 1D array of bandwidths."
    else:
        raise ValueError("Bandwidths must be None, 'median', an int number of bandwidths to generate, or a 1D np.array of bandwidths.")

    # Convert bandwidths to gammas
    gammas = 1.0 / (2.0 * bandwidths ** 2)


    # Calulate MMD
    if unique:
        unique_values, x_idx, y_idx = generate_ummd_input(x, y)  # [u, d], [m, ], [n, ]   
        u = len(unique_values)     
        K = kernel_matrix(unique_values, unique_values, gammas)  # [bandwidths, u, u]
        s_x = np.bincount(x_idx, minlength=u)/m                  # [u, ]
        s_y = np.bincount(y_idx, minlength=u)/n                  # [u, ] 
        s = s_x - s_y                                            # [u, ]
    else:
        K = kernel_matrix(xy, xy, gammas)       # [bandwidths, (m + n), (m + n)]

        s_x = np.ones(m)/m                      # [m, ]
        s_y = np.ones(n)/n * -1                 # [n, ] 
        s = np.concatenate((s_x, s_y))          # [(m + n), ]


    # Define results output dictionary
    res = {
        "bandwidths": bandwidths,
        "n_permutations": n_permutations,
        "biased_MMD": None,
        "p-values_per_bandwidth": None,
        "cauchy_method": cauchy_weighting,
        "p-value": None
    }

    
    obs = calc_MMD(K, s)                        # [bandwidths, ]
    res['biased_MMD'] = obs

    if n_permutations > 0:
        # NOTE: p-values will not be identical for a given seed and n_permutations between unique=True and unique=False.
        # The unique and brute-force paths sample the same permutation null but realize different draws at a given seed, 
        # so p-values differ by O(1/√B) Monte-Carlo error (independent of repeats); they converge with increasing n_permutations.
        
        batches = np.arange(0, n_permutations, perm_batch_size)
        rng = np.random.default_rng(seed)
        perms = np.empty((n_permutations, len(bandwidths)))  # [permutations, bandwidths]

        # Batch permutations
        for batch_start in batches:
            n_batch = min(perm_batch_size, n_permutations - batch_start)
            if unique:
                perms[batch_start:batch_start+n_batch] = perm_uMMD(K, x_idx, y_idx, rng=rng, n_permutations=n_batch)     # [batch_size, bandwidths]
            else:
                perms[batch_start:batch_start+n_batch] = perm_MMD(K, s, rng=rng, n_permutations=n_batch)                 # [batch_size, bandwidths]

        p_values = (np.sum(perms.round(10) >= obs.round(10), axis=0) + 1) / (n_permutations + 1)   # [bandwidths, ]
        res['p-values_per_bandwidth'] = p_values.round(6)

        # cauchy combination of p-values across bandwidths
        if cauchy_weighting is not None and len(bandwidths) > 1:
            if isinstance(cauchy_weighting, str):
                assert cauchy_weighting in ["uniform", "left", "right", "centered"], "Invalid cauchy weighting method. Must be one of ['uniform', 'left', 'right', 'centered']."

            res['p-value'] = cauchy_combination(p_values, weight_distribution=cauchy_weighting)


    return res