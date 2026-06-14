"""UMMD: a space and time efficient Maximum Mean Discrepancy two-sample test implementation for data with repeated sample values.

Maximum Mean Discrepancy (MMD) is a kernel-based test for whether two samples
are drawn from the same distribution. The naive kernel matrix costs O(N^2) in
time and memory; this implementation collapses repeated observations and works
over the U unique values instead, giving O(U^2), which can be a huge improvement
for data with many repeated values. Significance is assessed by permutation, with optional
testing over multiple RBF bandwidths aggregated via the Cauchy combination test.

Main entry point
----------------
MMD : run the (unique) MMD two-sample test and return statistics and p-values.


Example
-------
>>> import numpy as np
>>> from ummd import MMD
>>> rng = np.random.default_rng(0)
>>> x = rng.integers(2, 7, size=200)
>>> y = rng.integers(-5, 2, size=200)
>>> res = MMD(x, y, n_permutations=999, bandwidths=5, cauchy_weighting='centered')
>>> res["p-value"]

References
----------
Gretton et al. (2012), A Kernel Two-Sample Test.
Schrab et al. (2023), MMD Aggregated Two-Sample Test.
Liu and Xie (2019), Cauchy Combination Test.
"""

from scipy.spatial.distance import pdist
import numpy as np
import warnings

from ._utils import timer
from .kernels import resolve_kernel


@timer
def calc_MMD(K: np.array, s: np.array):
    """Calculate the biased MMD statistic given a kernel distance matrix and a sample weighting vector.

    Parameters
    ----------
    K : np.ndarray, shape (b, m, n)
        Kernel distance matrix.
    s : np.ndarray, shape (m + n,)
        Sample weighting vector representing class frequency (1/m) and negated indexes from y (-1/n).

    Returns
    -------
    res : np.ndarray, shape (b,)
        MMD values for each tested bandwidth.
    """
    return s @ K @ s.T


def perm_MMD(K, s, rng, n_permutations=999):
    """Calculate the biased MMD statistic across n_permutations.

    Parameters
    ----------
    K : np.ndarray, shape (b, m, n)
        Kernel distance matrix.
    s : np.ndarray, shape (m + n,)
        Sample weighting vector representing class frequency (1/m) and negated indexes from y (-1/n).
    rng : np.random.Generator
        Random number generator for permutation.
    n_permutations : int, optional
        Number of permutations to perform (default is 999).

    Returns
    -------
    res : np.ndarray, shape (n_permutations, b)
        MMD values for each tested bandwidth.
    """
    S = np.repeat(
        s[np.newaxis, :], repeats=n_permutations, axis=0
    )  # [permutations, m + n]
    S = rng.permuted(S, axis=1)
    perms = np.sum((S @ K) * S, 2)  # [bandwidths, permutations]
    return np.moveaxis(perms, 1, 0)  # [permutations, bandwidths]


def perm_uMMD(K, x_idx, y_idx, rng, n_permutations=0):
    """Calculate the biased MMD statistic for n_permutations of unique values.

    Requires a np.bincount across u * n_permutations over perm_MMD function; this adds time and space complexity
    but reclaims improved efficiency in cases with many repeated values.

    Parameters
    ----------
    K : np.ndarray, shape (b, u, u)
        Kernel distance matrix of unique values where ``u`` is the number of unique values.
    x_idx : np.ndarray, shape (m,)
        Sample index vector for the first distribution.
    y_idx : np.ndarray, shape (n,)
        Sample index vector for the second distribution.
    rng : np.random.Generator
        Random number generator for permutation.
    n_permutations : int, optional
        Number of permutations to perform (default is 0).

    Returns
    -------
    res : np.ndarray, shape (n_permutations, b)
        MMD values for each tested bandwidth.
    """

    xy_idx = np.concatenate((x_idx, y_idx))
    m = len(x_idx)
    n = len(y_idx)
    u = K.shape[-1]

    S = np.repeat(
        xy_idx[np.newaxis, :], repeats=n_permutations, axis=0
    )  # [permutations, m + n]
    S = rng.permuted(S, axis=1)
    X = S[:, :m]  # [permutations, m]

    # Vectorising bincount requires an offset trick. Add a new u index for each permutation and then bincount that.
    # Reshape that back to the original dimensions and you get the counts of the unique indexes for each permutation.
    # U_y can be easily calculated per permutation since U_x + U_y must = U_xy.
    U_xy = np.bincount(xy_idx, minlength=u)  # [u, ]
    offsets = np.arange(n_permutations)[:, None] * u  # [permutations, 1]
    U_x = np.bincount(
        (X + offsets).ravel(),  # [permutations * m, ]
        minlength=n_permutations * u,
    ).reshape(n_permutations, u)  # [permutations, u]
    U_y = U_xy - U_x  # [permutations, u]

    U = np.divide(U_x, m) - np.divide(U_y, n)  # [permutations, u]

    perms = np.sum((U @ K) * U, 2)  # [bandwidths, permutations]
    return np.moveaxis(perms, 1, 0)  # [permutations, bandwidths]


def get_bandwidths(xy, n=10):
    """Generate bandwidths for the RBF kernel based on the pairwise distances of the pooled sample.

    Generate a geometric grid of n sigma length-scales spanning the range of pairwise Euclidean distances
    across all samples. See Schrab et al. (2023) MMD Aggregated Two-Sample Test for motivation of this formula.

    Parameters
    ----------
    xy : np.ndarray, shape (m + n, d)
        Pooled samples from both distributions.
    n : int, optional
        Number of bandwidths to generate (default is 10).


    Returns
    -------
    sigmas : np.ndarray, shape (n,)
        Sigma length-scales.
    """
    D = pdist(xy, "euclidean")
    lambda_min, lambda_max = D.min(), D.max()
    t = np.arange(n) / (n - 1)
    sigmas = (lambda_min / 2) * ((2 * lambda_max) / (lambda_min / 2)) ** t
    return sigmas


def cauchy_combination(p_vals, weight_distribution="uniform"):
    """Combine p-values across bandwidths using the Cauchy combination method.

    Follows the formula ``T = sum(w_i * tan((0.5 - p_i) * pi))`` where ``w_i`` are the weights for each p-value and ``p_i`` are the individual p-values.
    See Liu and Xie (2019) Cauchy Combination Test... for more details.

    Parameters
    ----------
    p_vals : np.ndarray, shape (b,)
        Array of p-values to combine, where ``b`` is the number of bandwidths.
    weight_distribution : str or None, optional
        Method for weighting p-values in the combination. Options are:
        - "uniform": Equal weights for all p-values (default).
        - "left": More weight on smaller p-values.
        - "right": More weight on larger p-values.
        - "centered": More weight on p-values near 0.5.
        - None: No combination, return NaN for the combined p-value.

    Returns
    -------
    cauchy_p : float
        Combined p-value from the Cauchy combination method.

    Raises
    ------
    ValueError
        If an invalid weight distribution is provided.
    """

    p_vals = np.clip(
        p_vals, 1e-30, 1 - 1e-30
    )  # Avoid extreme p-values that can cause numerical issues

    def norm(x):
        return x / np.sum(x)

    match weight_distribution:
        case "uniform":
            w = norm(np.ones(len(p_vals)))
        case "left":
            w = norm(1 / np.arange(1, len(p_vals) + 1))
        case "right":
            w = norm(1 / np.arange(len(p_vals), 0, -1))
        case "centered":
            mid = (len(p_vals) - 1) / 2
            w = norm(np.exp(-0.5 * ((np.arange(len(p_vals)) - mid) / (mid / 2)) ** 2))
        case None:
            return np.nan  # No combination, return NaN for the combined p-value
        case _:
            raise ValueError(
                "Invalid weight distribution. Must be one of ['uniform', 'left', 'right', 'centered', None]."
            )

    # Cauchy combination formula
    T = np.sum(w * np.tan((0.5 - p_vals) * np.pi))
    cauchy_p = 0.5 - (np.arctan(T) / np.pi)
    return cauchy_p


@timer
def generate_ummd_input(x, y):
    """Convert two distributions into the unique values and index vectors representing the values in each distribution.

    Parameters
    ----------
    x : np.ndarray, shape (m, d)
        First distribution with ``m`` samples and ``d`` dimensions.
    y : np.ndarray, shape (n, d)
        Second distribution with ``n`` samples and ``d`` dimensions.

    Returns
    -------
    unique_values : np.ndarray, shape (u, d)
        Unique values from the combined distributions.
    x_idx : np.ndarray, shape (m,)
        Index vector representing the positions of ``x`` values in the unique values array.
    y_idx : np.ndarray, shape (n,)
        Index vector representing the positions of ``y`` values in the unique values array.
    """
    unique_values, inverse = np.unique(
        np.concatenate((x, y), axis=0), axis=0, return_inverse=True
    )
    x_idx = inverse[: len(x)]
    y_idx = inverse[len(x) :]
    return unique_values, x_idx, y_idx


@timer
def MMD(
    x,
    y,
    unique=True,
    kernel_fn="gaussian",
    bandwidths="median",
    n_permutations=0,
    perm_batch_size=999,
    cauchy_weighting="uniform",
    seed=11,
):
    """Calculate the MMD of two distributions.

    Maximum Mean Discrepancy (MMD) is a kernel-based distance measure between distributions allowing identification in second moment differences.
    The backbone of the test is based on kernel distance matrices, namely following the formula ``MMD^2 = K_x + K_y - 2K_xy``
    where ``K_x`` and ``K_y`` are kernel distances between each entry of X and Y distributions respectively,
    and ``K_xy`` is the cross-kernel distance matrix between each value of X with each value of Y.
    The kernel matrix itself requires O(N^2) time and space complexity per bandwidth, which can be reduced to O(U^2)
    where U is the number of unique values across both distributions with the unique value optimisation.

    Parameters
    ----------
    x : np.ndarray, shape (m, d)
        First distribution with ``m`` samples and ``d`` dimensions.
    y : np.ndarray, shape (n, d)
        Second distribution with ``n`` samples and ``d`` dimensions.
    unique : bool
        Whether to use the unique value optimisation, which can be much faster for discrete data with many repeated values. Default: True.
    kernel_fn : str or callable
        Kernel used to build the (stacked) kernel matrix. One of:
        - "gaussian": the built-in RBF kernel (default).
        - callable: ``kernel_fn(x, y, bandwidths) -> np.ndarray`` of shape
          ``(len(bandwidths), m, n)``, i.e. one kernel matrix per bandwidth. The
          callable builds the entire stacked matrix so it can be vectorised/optimised
          freely. ``bandwidths`` is the 1-D array of sigma length-scales resolved by
          MMD; the kernel owns how it interprets them (and may ignore them, returning
          a single (1, m, n) matrix).
    bandwidths : str or int or np.ndarray, shape (b,)
        Kernel bandwidths as sigma length-scales (same units as the data). One of:
        - "median": median pairwise Euclidean distance of the pooled unique sample (default).
        - int: generate that many bandwidths spanning the pooled pairwise distances (see get_bandwidths).
        - 1-D np.array: the sigma values to test.
        Each sigma is converted internally to an RBF gamma via gamma = 1 / (2 * sigma**2).
    n_permutations : int
        number of permutations to approximate p-value. Default: 0.
    perm_batch_size : int
        number of permutations to calculate in each batch. Default: 999.
    cauchy_weighting: str or None
        Method for weighting p-values across bandwidths in the cauchy combination. If None,
        p-values per bandwidth are returned without aggregation. Weighting options:
            - "centered": Highest weight on bandwidths near the median, decreasing towards the extremes.
            - "uniform": Equal weight on p-values across all bandwidths (default).
            - "left": More weight on smaller bandwidths.
            - "right": More weight on larger bandwidths.
            - None: No Cauchy aggregation.
    seed : int
        Random seed for reproducibility. Default: 11.

    Returns
    -------
    res : dict
        Dictionary of MMD results with attributes:
            - bandwidths: bandwidths used in the RBF kernel.
            - n_permutations: number of permutations used to approximate p-value.
            - biased_MMD: MMD statistic per bandwidth.
            - p-values_per_bandwidth: permuation derived p-values for each bandwidth tested.
            - cauchy_method: method used for Cauchy combination.
            - p-value: Cauchy adjusted p-value across bandwidths if cauchy_weighting is not None, otherwise the same as p-values_per_bandwidth.

    Raises
    ------
    ValueError
        If bandwidths parameter is invalid.
        If cauchy_weighting parameter is invalid.

    Warns
    -----
    UserWarning
        If all values across both distributions are identical.
    """

    # Check for 2d array
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    m = len(x)
    n = len(y)

    xy = np.concatenate((x, y), axis=0)  # [(m + n), d]

    # Handle case when all values are identical
    if len(np.unique(xy, axis=0)) == 1:
        warnings.warn(
            "All values are identical across both distributions. MMD will be 0 and p-value 1. Skipping computation.",
            UserWarning,
        )
        return {
            "bandwidths": np.array([np.nan]),
            "n_permutations": np.nan,
            "biased_MMD": np.array([0.0]),
            "p-values_per_bandwidth": np.array([1.0]),
            "cauchy_method": None,
            "p-value": np.array([1.0]),
        }

    # Resolve kernel function
    kernel_matrix = resolve_kernel(kernel_fn)

    # Resolve bandwidths
    if isinstance(bandwidths, np.ndarray):
        if bandwidths.ndim != 1:
            raise ValueError("Bandwidths array must be 1D.")
    elif bandwidths is None or bandwidths == "median":
        bandwidths = np.array(
            [np.median(pdist(np.unique(xy, axis=0), metric="euclidean"))]
        )
    elif isinstance(bandwidths, (int, np.integer)):
        if bandwidths <= 1:
            bandwidths = np.array(
                [np.median(pdist(np.unique(xy, axis=0), metric="euclidean"))]
            )
        else:
            bandwidths = get_bandwidths(np.unique(xy, axis=0), n=bandwidths)
    else:
        raise ValueError("Bandwidths must be None, 'median', an int, or a 1D np.array.")

    # Calulate MMD
    if unique:
        unique_values, x_idx, y_idx = generate_ummd_input(x, y)  # [u, d], [m, ], [n, ]
        u = len(unique_values)
        K = kernel_matrix(
            unique_values, unique_values, bandwidths
        )  # [bandwidths, u, u]
        s_x = np.bincount(x_idx, minlength=u) / m  # [u, ]
        s_y = np.bincount(y_idx, minlength=u) / n  # [u, ]
        s = s_x - s_y  # [u, ]
    else:
        K = kernel_matrix(xy, xy, bandwidths)  # [bandwidths, (m + n), (m + n)]

        s_x = np.ones(m) / m  # [m, ]
        s_y = np.ones(n) / n * -1  # [n, ]
        s = np.concatenate((s_x, s_y))  # [(m + n), ]

    # Validate kernel output: one (m, n) kernel matrix per bandwidth
    K = np.asarray(K)
    if K.ndim != 3 or K.shape[0] != len(bandwidths):
        raise ValueError(
            "kernel_fn must return a stacked kernel matrix of shape "
            f"(len(bandwidths), m, n); expected {len(bandwidths)} matrices "
            f"but got an array with shape {K.shape}."
        )

    # Define results output dictionary
    res = {
        "bandwidths": bandwidths,
        "n_permutations": n_permutations,
        "biased_MMD": None,
        "p-values_per_bandwidth": None,
        "cauchy_method": cauchy_weighting,
        "p-value": None,
    }

    obs = calc_MMD(K, s)  # [bandwidths, ]
    res["biased_MMD"] = obs

    if n_permutations > 0:
        # NOTE: p-values will not be identical for a given seed and n_permutations between unique=True and unique=False.
        # The unique and brute-force paths sample the same permutation null but realize different draws at a given seed,
        # so p-values differ by O(1/√B) Monte-Carlo error (independent of repeats); they converge with increasing n_permutations.

        batches = np.arange(0, n_permutations, perm_batch_size)
        rng = np.random.default_rng(seed)
        perms = np.empty(
            (n_permutations, len(bandwidths))
        )  # [permutations, bandwidths]

        # Batch permutations
        for batch_start in batches:
            n_batch = min(perm_batch_size, n_permutations - batch_start)
            if unique:
                perms[batch_start : batch_start + n_batch] = perm_uMMD(
                    K, x_idx, y_idx, rng=rng, n_permutations=n_batch
                )  # [batch_size, bandwidths]
            else:
                perms[batch_start : batch_start + n_batch] = perm_MMD(
                    K, s, rng=rng, n_permutations=n_batch
                )  # [batch_size, bandwidths]

        p_values = (np.sum(perms.round(10) >= obs.round(10), axis=0) + 1) / (
            n_permutations + 1
        )  # [bandwidths, ]
        res["p-values_per_bandwidth"] = p_values.round(6)

        # cauchy combination of p-values across bandwidths
        if cauchy_weighting is not None and len(bandwidths) > 1:
            if isinstance(cauchy_weighting, str):
                if cauchy_weighting not in ["uniform", "left", "right", "centered"]:
                    raise ValueError(
                        "Invalid cauchy weighting method. Must be one of ['uniform', 'left', 'right', 'centered']."
                    )

            res["p-value"] = cauchy_combination(
                p_values, weight_distribution=cauchy_weighting
            )
        else:
            res["p-value"] = res["p-values_per_bandwidth"]

    return res
