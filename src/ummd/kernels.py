"""Kernel matrix builders for the MMD test, plus the registry of built-in kernels.

A kernel builds the *entire* stacked kernel matrix, so it can be vectorised/optimised
freely. The contract is::

    kernel_fn(x, y, bandwidths) -> np.ndarray of shape (len(bandwidths), m, n)

i.e. one kernel matrix per bandwidth. ``bandwidths`` is the 1-D array of sigma
length-scales resolved by ``MMD``; a kernel that does not need them can simply
ignore the argument.
"""

import numpy as np
from scipy.spatial.distance import cdist

from ._utils import timer


@timer
def gaussian_kernel_matrix(x, y, bandwidths):
    """Compute the RBF (Gaussian) kernel matrix between two distributions.

    One kernel matrix is produced per bandwidth, using squared Euclidean distance
    with gamma = 1/(2*sigma**2), i.e. k(a, b) = exp(-(1 / (2 * sigma**2)) * ||a - b||**2).

    Parameters
    ----------
    x : np.ndarray, shape (m, d)
        First distribution with ``m`` samples and ``d`` dimensions.
    y : np.ndarray, shape (n, d)
        Second distribution with ``n`` samples and ``d`` dimensions.
    bandwidths : np.ndarray, shape (b,)
        1-D array of RBF sigma length-scales, one per bandwidth. Each sigma is
        converted internally to an RBF precision via gamma = 1 / (2 * sigma**2).

    Returns
    -------
    np.ndarray, shape (b, m, n)
        Kernel matrices for each bandwidth, where ``b`` is the number of bandwidths,
        ``m`` is the number of samples in ``x``, and ``n`` is the number of samples in ``y``.

    Raises
    ------
    AssertionError
        If ``bandwidths`` is not a 1D array.
    """

    assert isinstance(bandwidths, np.ndarray) and bandwidths.ndim == 1, (
        "Bandwidths must be a 1D array of sigma length-scales."
    )

    gammas = 1.0 / (2.0 * bandwidths**2)  # [bandwidths, ]
    D = cdist(x, y, metric="sqeuclidean")  # [m, n]
    K = np.exp(-gammas[:, None, None] * D[None, :, :])  # [bandwidths, m, n]
    return K


# Registry of built-in named kernels. Custom kernels are passed directly as callables.
_KERNELS = {"gaussian": gaussian_kernel_matrix}


def resolve_kernel(kernel_fn):
    """Resolve ``kernel_fn`` (a built-in name or a callable) to a kernel matrix builder.

    Raises
    ------
    ValueError
        If ``kernel_fn`` is neither a callable nor a known built-in name.
    """
    if callable(kernel_fn):
        return kernel_fn
    if isinstance(kernel_fn, str) and kernel_fn in _KERNELS:
        return _KERNELS[kernel_fn]
    raise ValueError(
        f"kernel_fn must be a callable or one of {sorted(_KERNELS)}; got {kernel_fn!r}."
    )
