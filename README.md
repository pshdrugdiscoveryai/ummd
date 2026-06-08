# Unique Maximum Mean Discrepancy (uMMD)

An efficient implementation of the Maximum Mean Discrepancy two-sample test for datasets with duplicate observations via count-weighting of unique values. This implementation scales with unique data values rather than sample size.

## Installation

```bash
pip install ummd
```

## Quick start

```python
import numpy as np
from ummd import MMD

rng = np.random.default_rng(0)
x = rng.integers(0, 10, size=500)   # sample from one distribution
y = rng.integers(2, 12, size=500)   # sample from a shifted distribution

result = MMD(x, y, unique=True, bandwidths=10, n_permutations=999)

print(result["biased_MMD"])   # MMD statistic per bandwidth
# [ 0.04408069 0.053788   0.06124013 0.06328209 0.06290089 0.0602459 0.04713144 0.02831863 0.01431563 0.0066321 ]

print(result["p-value"])    # combined p-value across bandwidths
# 0.001
```

## Interpreting the result

MMD returns a dictionary with:

- `biased_MMD`: the MMD statistic for each tested bandwidth
- `p-values_per_bandwidth`: permutation p-value for each bandwidth
- `p-value`: a single Cauchy-combined p-value across the bandwidths
- `bandwidths`: the kernel bandwidths actually used

## Why uMMD

A standard MMD test builds an `N x N` kernel matrix, so cost grows with sample size. When your data has many repeated values (counts, categories, discretised measurements), uMMD instead works over the `u` unique values, where `u << n`, giving the same test at a fraction of the cost.
