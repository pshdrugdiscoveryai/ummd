from scipy.optimize import brentq
from scipy.special import ndtr
import numpy as np


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


def _lugannani_rice(weights, t):
    """Upper-tail probability ``P(Q >= t)`` for a non-negatively weighted sum of independent
    chi-square variables with one degree of freedom, ``Q = sum_i weights_i * Z_i**2``.

    Uses the Lugannani-Rice saddlepoint approximation as applied to quadratic forms by
    Kuonen (1999). The cumulant generating function of ``Q`` is
    ``L(z) = -0.5 * sum_i log(1 - 2 * weights_i * z)``; the saddlepoint ``z_hat`` solves
    ``L'(z_hat) = t`` and the tail probability follows in closed form. Near the mean the
    formula has a removable singularity, handled by the limiting skewness correction.

    Parameters
    ----------
    weights : np.ndarray, shape (r,)
        Mixture weights (kernel eigenvalues). Non-positive entries are treated as numerical
        zeros and dropped.
    t : float
        Point at which the upper tail is evaluated (the observed statistic).

    Returns
    -------
    float
        Approximate ``P(Q >= t)``, clipped to ``[0, 1]``.
    """
    lam = np.asarray(weights, dtype=float)
    if lam.size:
        lam = lam[
            lam > 1e-10 * np.max(lam)
        ]  # drop the (near-)zero centering eigenvalues
    if lam.size == 0:  # degenerate null: Q is identically 0
        return 1.0 if t <= 0 else 0.0
    if t <= 0:  # statistic below the support of Q
        return 1.0

    mean = lam.sum()  # E[Q] = sum_i weights_i (one dof each)

    def dcgf(z):  # L'(z)
        return np.sum(lam / (1.0 - 2.0 * lam * z))

    def d2cgf(z):  # L''(z)
        return np.sum(2.0 * (lam / (1.0 - 2.0 * lam * z)) ** 2)

    def cgf(z):  # L(z)
        return -0.5 * np.sum(np.log1p(-2.0 * lam * z))

    def skewness_tail():
        # Limiting tail at the removable singularity z_hat -> 0 (t == E[Q]).
        s2 = np.sum(lam**2)
        s3 = np.sum(lam**3)
        skew = 8.0 * s3 / (2.0 * s2) ** 1.5  # L'''(0) / L''(0)**1.5
        return float(0.5 - skew / (6.0 * np.sqrt(2.0 * np.pi)))

    if np.isclose(t, mean):
        return skewness_tail()

    # Bracket and solve the saddlepoint equation L'(z) = t. L' increases from 0 (at z -> -inf)
    # to +inf at the right edge z = 1 / (2 * max weight) of the convergence strip.
    zeta_hi = 0.5 / lam.max()
    if t > mean:
        eps = 1e-6
        lo, hi = 0.0, zeta_hi * (1.0 - eps)
        while (
            dcgf(hi) <= t and eps > 1e-15
        ):  # push toward the singularity for extreme tails
            eps *= 0.1
            hi = zeta_hi * (1.0 - eps)
    else:
        lo, hi = -1.0, 0.0
        while dcgf(lo) > t:  # expand until L'(lo) drops below t
            lo *= 2.0
    z_hat = brentq(lambda z: dcgf(z) - t, lo, hi, xtol=1e-12, rtol=1e-14)

    w = np.sign(z_hat) * np.sqrt(max(2.0 * (z_hat * t - cgf(z_hat)), 0.0))
    v = z_hat * np.sqrt(d2cgf(z_hat))
    if abs(w) < 1e-9 or abs(v) < 1e-9:  # too close to the mean for the stable form
        return skewness_tail()

    phi = np.exp(-0.5 * w * w) / np.sqrt(2.0 * np.pi)
    p = ndtr(-w) + phi * (1.0 / v - 1.0 / w)
    return float(min(max(p, 0.0), 1.0))


def saddlepoint_pvalue(K, counts, m, n, obs):
    """Analytical MMD p-values per bandwidth via the Kuonen (1999) saddlepoint approximation.

    Under the permutation null the biased MMD statistic ``T = s @ K @ s`` behaves like a
    non-negatively weighted sum of independent chi-square variables with one degree of freedom,
    ``T ~ sum_i lambda_i * Z_i**2``. The weights ``lambda_i`` are the eigenvalues of the
    count-weighted, doubly-centered kernel, scaled by ``N / (m * n * (N - 1))`` with ``N = m + n``;
    the Lugannani-Rice saddlepoint formula then gives the upper tail ``P(T >= obs)`` in closed
    form, replacing the permutation loop.

    The weights are obtained from ``K_b @ G`` where ``G = diag(counts) - counts counts^T / N`` is
    the count-weighted centering operator. Because ``G`` and the kernel are indexed by the ``u``
    unique values, the eigendecomposition stays O(u**3) rather than O(N**3), preserving the
    unique-value speed-up. This yields the same weights (and hence p-values) whether the caller
    ran the unique-value or brute-force path.

    Parameters
    ----------
    K : np.ndarray, shape (b, u, u)
        Stacked kernel matrix over the ``u`` unique values, one matrix per bandwidth.
    counts : np.ndarray, shape (u,)
        Total number of observations taking each unique value (sums to ``N = m + n``).
    m : int
        Number of samples in the first distribution.
    n : int
        Number of samples in the second distribution.
    obs : np.ndarray, shape (b,)
        Observed biased MMD statistic per bandwidth.

    Returns
    -------
    np.ndarray, shape (b,)
        Analytical p-values, one per bandwidth.
    """
    counts = np.asarray(counts, dtype=float)
    N = m + n

    # G = diag(counts) - counts counts^T / N is symmetric PSD with null vector 1. Factor it once
    # as G = L L^T so each per-bandwidth solve reduces to the symmetric eigenproblem L^T K_b L,
    # which shares the non-zero spectrum of K_b @ G.
    G = np.diag(counts) - np.outer(counts, counts) / N
    gvals, gvecs = np.linalg.eigh(G)
    gvals = np.clip(gvals, 0.0, None)
    L = gvecs * np.sqrt(gvals)  # scale eigenvector columns; G == L @ L.T
    scale = N / (m * n * (N - 1))

    pvals = np.empty(K.shape[0])
    for b in range(K.shape[0]):
        weights = scale * np.linalg.eigvalsh(L.T @ K[b] @ L)
        pvals[b] = _lugannani_rice(weights, float(obs[b]))
    return pvals


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
