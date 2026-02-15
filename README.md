# ESN Toolbox

A Python toolbox for evaluating information-theoretic measures in Echo State Networks (ESN). Implements the Kraskov-Stoegbauer-Grassberger (KSG) algorithm for estimating Transfer Entropy and Active Information Storage, along with Memory Capacity, Lyapunov Exponent computation, Ragwitz-Kantz embedding criterion, and Kozachenko-Leonenko differential entropy estimation.

This toolbox accompanies the conference paper:

> M. Torda and I. Farkas, "Evaluation of Information-Theoretic Measures in Echo State Networks on the Edge of Stability," *2018 International Joint Conference on Neural Networks (IJCNN)*, Rio de Janeiro, Brazil, 2018, pp. 1-6, doi: [10.1109/IJCNN.2018.8489152](https://doi.org/10.1109/IJCNN.2018.8489152)

## Requirements

- Python 3.6
- numpy, scipy, scikit-learn
- numba (optional, for the optimized TE permutation test)

## Installation

```bash
git clone https://github.com/milotorda-net/esn-toolbox.git
cd esn-toolbox
pip install -r requirements.txt
```

## Algorithms

| Function | Module | Description |
|---|---|---|
| `AIS` | `ESNtoolbox` | Active Information Storage (KSG estimator) |
| `TE` | `ESNtoolbox` | Transfer Entropy (KSG estimator) |
| `MC` | `ESNtoolbox` | Memory Capacity of an ESN |
| `LE` | `ESNtoolbox` | Lyapunov Exponent of an ESN |
| `ragwitz` | `ESNtoolbox` | Ragwitz-Kantz embedding criterion grid search |
| `locally_const_predictor` | `ESNtoolbox` | Locally constant prediction error for the Ragwitz criterion |
| `entropy` | `ESNtoolbox` | Kozachenko-Leonenko differential entropy estimator |
| `significance_TE` | `TE_permutation_test` | Permutation test p-value for Transfer Entropy |
| `pairwise_TE` | `TE_permutation_test_numba012` | Pairwise Transfer Entropy matrix of a network |

## Usage

### Transfer Entropy between two time series

```python
from ESNtoolbox import TE

# TE(source, target, kHistory, kTau, lHistory, lTau, u, k)
te_value = TE(source, target, kHistory=1, kTau=1, lHistory=1, lTau=1, u=1, k=4)
```

### Active Information Storage

```python
from ESNtoolbox import AIS

# AIS(target, kHistory, kTau, k)
ais_value = AIS(target, kHistory=1, kTau=1, k=4)
```

### Transfer Entropy with significance testing

```python
from TE_permutation_test import significance_TE

# Returns (p_value, te_value)
p_value, te_value = significance_TE(source, target,
    kHistory=1, kTau=1, lHistory=1, lTau=1, u=1, k=4, n_permutations=100)
```

### Pairwise Transfer Entropy of a network

```python
from TE_permutation_test_numba012 import pairwise_TE

# X is an N x K matrix (N nodes, K time steps)
# Returns N x N x 2 array (TE values and p-values)
result = pairwise_TE(X, kHistory=1, kTau=1, lHistory=1, lTau=1, u=1, k=4,
    n_permutations=100, significance=True)
```

## Modules

| File | Description |
|---|---|
| `ESNtoolbox.py` | Core library: KSG estimators (AIS, TE), Memory Capacity, Lyapunov Exponent, Ragwitz-Kantz criterion, differential entropy |
| `TE_permutation_test.py` | Permutation test for Transfer Entropy significance |
| `TE_permutation_test_numba012.py` | Numba-JIT-optimized permutation test with pairwise TE computation for full networks |

## Research Blog

The development of this toolbox and the computational experiments performed with it are documented on the author's research blog:

- Blog pages [13](https://milotorda.net/blog/page/13/)--[16](https://milotorda.net/blog/page/16/) contain the full series of ESN experiments (Sep 2017 -- Jan 2018), including Python scripts for reproducing the results on Mackey-Glass, NARMA and Lorenz attractor systems.

## Citation

```bibtex
@inproceedings{torda2018evaluation,
  title     = {Evaluation of information-theoretic measures in echo state networks on the edge of stability},
  author    = {Torda, Miloslav and Farkas, Igor},
  booktitle = {2018 International Joint Conference on Neural Networks (IJCNN)},
  pages     = {1--6},
  year      = {2018},
  organization = {IEEE}
}
```

## License

[MIT](LICENSE)
