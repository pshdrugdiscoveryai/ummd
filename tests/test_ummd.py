import numpy as np
from ummd import MMD
from ummd.statistical_testing import _lugannani_rice
from scipy.spatial.distance import cdist
from scipy.stats import chi2
import pytest


@pytest.fixture
def discrete_distributions():
    rng = np.random.default_rng(seed=11)
    # Choosing integers guarentees repeats
    x = rng.integers(0, 5, size=200)
    y = rng.integers(0, 5, size=200)
    return x, y


@pytest.fixture
def discrete_joint_distributions():
    rng = np.random.default_rng(seed=11)
    # Choosing integers guarentees repeats

    x = rng.integers(0, 5, size=(200, 10))
    y = rng.integers(0, 5, size=(200, 10))
    return x, y


# Arrange, act, assert
def test_same_distribution_returns_zero(discrete_distributions):
    # Arrange
    x, _ = discrete_distributions

    # Act
    res = MMD(x, x, unique=True, n_permutations=0)

    # Assert
    np.testing.assert_allclose(res["biased_MMD"][0], 0)


def test_unique_and_non_unique_agree(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act
    res_unique = MMD(x, y, unique=True, n_permutations=0)
    res_non_unique = MMD(x, y, unique=False, n_permutations=0)

    # Assert
    np.testing.assert_allclose(
        res_unique["biased_MMD"][0], res_non_unique["biased_MMD"][0]
    )


def test_statistic_is_symmetric(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act
    res_xy = MMD(x, y, unique=True, n_permutations=0)
    res_yx = MMD(y, x, unique=True, n_permutations=0)

    # Assert
    np.testing.assert_allclose(res_xy["biased_MMD"][0], res_yx["biased_MMD"][0])


def test_statistic_is_non_negative(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act
    res = MMD(x, y, unique=True, n_permutations=0)

    # Assert
    assert np.all(res["biased_MMD"] >= 0)


def test_smoke_with_joint_distributions(discrete_joint_distributions):
    # Arrange
    x, y = discrete_joint_distributions

    # Act
    res = MMD(x, y, unique=True, n_permutations=0)

    # Assert
    assert np.all(np.isfinite(res["biased_MMD"]))


@pytest.mark.parametrize("bandwidths", [None, "median", 1, 5, np.array([1.0, 5.0])])
def test_smoke_bandwidth_selection(discrete_distributions, bandwidths):
    # Arrange
    x, y = discrete_distributions

    # Act
    res = MMD(x, y, unique=True, bandwidths=bandwidths, n_permutations=0)

    # Assert
    assert np.all(np.isfinite(res["biased_MMD"]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bandwidths": 1, "n_permutations": 0},
        {"bandwidths": 5, "n_permutations": 0},
        {"n_permutations": 9, "perm_batch_size": 9},
        {"n_permutations": 99, "perm_batch_size": 9},
        {"bandwidths": 5, "n_permutations": 99, "perm_batch_size": 9},
    ],
)
def test_smoke_across_parameters(discrete_distributions, kwargs):
    # Arrange
    x, y = discrete_distributions

    # Act, assert
    res = MMD(x, y, unique=True, **kwargs)
    assert np.all(np.isfinite(res["biased_MMD"]))

    # Act, assert
    res = MMD(x, y, unique=False, **kwargs)
    assert np.all(np.isfinite(res["biased_MMD"]))


def test_p_values_between_zero_and_one(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act
    res = MMD(x, y, unique=True, n_permutations=99)

    # Assert
    assert np.all(
        (res["p-values_per_bandwidth"] >= 0) & (res["p-values_per_bandwidth"] <= 1)
    )


def test_p_values_identify_same_distribution(discrete_distributions):
    # Arrange
    x, _ = discrete_distributions

    # Act
    res = MMD(x, x, unique=True, n_permutations=99)

    # Assert
    assert np.all(
        (res["p-values_per_bandwidth"] >= 0) & (res["p-values_per_bandwidth"] <= 1)
    )
    np.testing.assert_allclose(
        res["p-values_per_bandwidth"], 1.0
    )  # Same distribution should not be significant


def test_p_values_identify_different_distributions():
    # Arrange
    rng = np.random.default_rng(seed=11)
    x = rng.integers(-5, 2, size=200)
    y = rng.integers(0, 5, size=200)

    # Act
    res = MMD(x, y, unique=True, n_permutations=99)

    # Assert
    assert np.all(
        (res["p-values_per_bandwidth"] >= 0) & (res["p-values_per_bandwidth"] <= 1)
    )
    assert np.all(res["p-values_per_bandwidth"] < 0.05)


def test_cauchy_smoke(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act
    res = MMD(x, y, unique=True, bandwidths=5, n_permutations=99)

    # Assert
    assert np.all(np.isfinite(res["p-value"]))


def test_fails_with_non_2d_arrays(discrete_joint_distributions):
    # Arrange
    x, y = discrete_joint_distributions

    # Act, assert
    with pytest.raises(ValueError):
        MMD(x[:, np.newaxis], y[:, np.newaxis], unique=True, n_permutations=99)


def test_fails_with_invalid_bandwidths(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act, assert
    with pytest.raises(ValueError):
        MMD(x, y, unique=True, bandwidths="banana", n_permutations=99)

    with pytest.raises(ValueError):
        MMD(x, y, unique=True, bandwidths=np.array([[1.0], [5.0]]), n_permutations=99)


def test_fails_with_invalid_cauchy_weighting(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act, assert
    with pytest.raises(ValueError):
        MMD(
            x,
            y,
            unique=True,
            bandwidths=2,
            cauchy_weighting="banana",
            n_permutations=99,
        )


def test_runs_with_only_one_unique_value():
    # Arrange
    x = np.array([1] * 100)
    y = np.array([1] * 100)

    # Act
    with pytest.warns(UserWarning):
        res = MMD(x, y, unique=True, n_permutations=99)

    assert np.allclose(res["biased_MMD"], 0.0)
    assert np.allclose(res["p-value"], 1.0)


def test_one_bandwidth_returns_p_val(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act
    res = MMD(x, y, unique=True, bandwidths=1, n_permutations=99)

    # Assert
    assert np.all((res["p-value"] >= 0) & (res["p-value"] <= 1))


def test_kernel_function_smoke(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act
    res = MMD(x, y, unique=True, bandwidths=1, n_permutations=99, kernel_fn="gaussian")

    # Assert
    assert np.all(np.isfinite(res["biased_MMD"]))


def test_custom_kernel_matches_builtin_gaussian(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    def custom_rbf(x, y, bandwidths):
        gammas = 1.0 / (2.0 * bandwidths**2)
        D = cdist(x, y, metric="sqeuclidean")
        return np.exp(-gammas[:, None, None] * D[None, :, :])

    # Act
    res_custom = MMD(
        x, y, unique=True, bandwidths=5, n_permutations=0, kernel_fn=custom_rbf
    )
    res_builtin = MMD(
        x, y, unique=True, bandwidths=5, n_permutations=0, kernel_fn="gaussian"
    )

    # Assert
    np.testing.assert_allclose(res_custom["biased_MMD"], res_builtin["biased_MMD"])


def test_fails_with_invalid_kernel_fn(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act, assert
    with pytest.raises(ValueError):
        MMD(x, y, unique=True, bandwidths=1, n_permutations=0, kernel_fn="banana")

    with pytest.raises(ValueError):
        MMD(x, y, unique=True, bandwidths=1, n_permutations=0, kernel_fn=123)

    def bad_kernel(x, y, bandwidths):
        return np.ones((len(x), len(y)))  # missing the leading bandwidth axis

    # Act, assert
    with pytest.raises(ValueError):
        MMD(x, y, unique=True, bandwidths=1, n_permutations=0, kernel_fn=bad_kernel)


def test_custom_kernel_returns_expected(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    def custom_kernel(x, y, bandwidths):
        return np.zeros((1, len(x), len(y)))  # missing the leading bandwidth axis

    res = MMD(
        x, y, unique=True, bandwidths=1, n_permutations=0, kernel_fn=custom_kernel
    )

    # Act, assert
    assert np.isclose(res["biased_MMD"], 0.0)


# --- Saddlepoint (Kuonen, 1999) analytical p-values ---------------------------


def test_lugannani_rice_matches_chi2_single_weight():
    # A single-weight mixture is a scaled chi-square_1: P(lam * Z^2 >= t) = chi2_1.sf(t / lam).
    for lam in (0.5, 2.0):
        for t in (1.0, 3.0, 6.0):
            approx = _lugannani_rice(np.array([lam]), t)
            exact = chi2.sf(t / lam, df=1)
            assert abs(approx - exact) < 0.01


def test_lugannani_rice_edge_cases():
    # Statistic at/below the support floor, and a degenerate (all-zero-weight) null.
    assert _lugannani_rice(np.array([1.0, 2.0]), 0.0) == 1.0
    assert _lugannani_rice(np.array([0.0, 0.0]), 1.0) == 0.0
    p = _lugannani_rice(
        np.array([2.0, 1.0, 0.5]), 3.5
    )  # t == E[Q]: removable singularity
    assert 0.0 <= p <= 1.0


def test_saddlepoint_p_values_between_zero_and_one(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act
    res = MMD(x, y, unique=True, bandwidths=5, method="saddlepoint")

    # Assert
    p = np.atleast_1d(res["p-values_per_bandwidth"])
    assert np.all((p >= 0) & (p <= 1))


def test_saddlepoint_identifies_same_distribution(discrete_distributions):
    # Arrange
    x, _ = discrete_distributions

    # Act
    res = MMD(x, x, unique=True, bandwidths=5, method="saddlepoint")

    # Assert
    np.testing.assert_allclose(res["p-values_per_bandwidth"], 1.0)


def test_saddlepoint_identifies_different_distributions():
    # Arrange
    rng = np.random.default_rng(seed=11)
    x = rng.integers(-5, 2, size=200)
    y = rng.integers(0, 5, size=200)

    # Act
    res = MMD(
        x, y, unique=True, bandwidths=5, method="saddlepoint", cauchy_weighting=None
    )

    # Assert
    assert np.all(res["p-values_per_bandwidth"] < 0.05)


def test_saddlepoint_is_deterministic(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act: no sampling, so repeated calls must be identical
    a = MMD(
        x, y, unique=True, bandwidths=5, method="saddlepoint", cauchy_weighting=None
    )
    b = MMD(
        x, y, unique=True, bandwidths=5, method="saddlepoint", cauchy_weighting=None
    )

    # Assert
    np.testing.assert_array_equal(
        a["p-values_per_bandwidth"], b["p-values_per_bandwidth"]
    )


def test_saddlepoint_unique_and_non_unique_agree(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act: the unique-value and brute-force paths share the same non-zero null spectrum.
    ru = MMD(
        x, y, unique=True, bandwidths=5, method="saddlepoint", cauchy_weighting=None
    )
    rn = MMD(
        x, y, unique=False, bandwidths=5, method="saddlepoint", cauchy_weighting=None
    )

    # Assert
    np.testing.assert_allclose(
        ru["p-values_per_bandwidth"], rn["p-values_per_bandwidth"], atol=1e-6
    )


def test_saddlepoint_agrees_with_permutation(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act
    perm = MMD(
        x,
        y,
        unique=True,
        bandwidths=5,
        method="permutation",
        n_permutations=9999,
        cauchy_weighting=None,
    )
    saddle = MMD(
        x, y, unique=True, bandwidths=5, method="saddlepoint", cauchy_weighting=None
    )

    # Assert: analytical tail tracks the Monte-Carlo permutation p-values.
    np.testing.assert_allclose(
        saddle["p-values_per_bandwidth"],
        perm["p-values_per_bandwidth"],
        atol=0.05,
    )


def test_saddlepoint_cauchy_smoke(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act
    res = MMD(x, y, unique=True, bandwidths=5, method="saddlepoint")

    # Assert
    assert np.all(np.isfinite(res["p-value"]))
    assert (res["p-value"] >= 0) and (res["p-value"] <= 1)


def test_fails_with_invalid_method(discrete_distributions):
    # Arrange
    x, y = discrete_distributions

    # Act, assert
    with pytest.raises(ValueError):
        MMD(x, y, unique=True, method="banana")
