import numpy as np
from ummd import MMD
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
