# softDOE

`softDOE` is a Python package for constructing space-filling
one-factor-at-a-time (SOFT) designs.

## What is included

`softDOE` currently includes:

* `soft()`: generate SOFT designs.
* `measure()`: compute screening measures.

Generated designs are returned as NumPy arrays with entries in `(0, 1)`.

## Installation

Install `softDOE` from PyPI:

```bash
pip install softDOE
```


`softDOE` requires Python 3.9 or later. The package depends on NumPy, SciPy,
and pyDOE, and uses a compiled C++ extension built with pybind11.

## Quick example

```python
import numpy as np
import softDOE

# SOFT design
design = softDOE.soft(p=10, l=4)

# Friedman function
def friedman(x):
    x = np.asarray(x)
    return (
        10 * np.sin(np.pi * x[0] * x[1])
        + 20 * (x[2] - 0.5) ** 2
        + 10 * x[3]
        + 5 * x[4]
    )

# Evaluate Friedman function at each design point.
y = np.array([friedman(row) for row in design])

# Sensitivity measures
result = softDOE.measure(design, y)

print(result["mustar"])
print(result["t"])
```

## Generating SOFT designs

Use `softDOE.soft()` to generate a SOFT design:

```python
design = softDOE.soft(p=5, l=4, structure="standard", m=500, seed=123)
```

Arguments:

* `p`: number of factors.
* `l`: number of base runs. This value must be even.
* `structure`: either `"standard"` or `"strict"`. The default is
  `"standard"`.
* `m`: number of Monte Carlo samples used when evaluating the design
  objective.
* `seed`: optional seed for reproducible design generation.

The returned design has shape:

```python
(l * (p + 1), p)
```

## Computing screening measures

Use `softDOE.measure()` with a design matrix and response vector:

```python
result = softDOE.measure(design, y)
```

The response vector `y` must have one value for each row of the design. The
function returns a dictionary with two entries:

* `mustar`: $\mu^*$ measure of Campolongo et al. (2007).
* `t`: total Sobol' index of Sobol' (1993).

## Testing

After installing the test dependencies, run the test suite with:

```bash
python -m pytest
```

## Citation

If you use `softDOE`, please cite:

```bibtex
@article{yu2026space,
  title={Space-Filling One-Factor-At-A-Time Designs},
  author={Yu, Wei-Yang and Joseph, V Roshan},
  journal={arXiv preprint arXiv:2606.02533},
  year={2026}
}
```

## References

* Sobol', I. M. (1993), "On sensitivity estimation for nonlinear mathematical
  models," Mathematical Modeling and Computational Experiments, 1, 407-414.
* Campolongo, F., Cariboni, J., and Saltelli, A. (2007), "An effective
  screening design for sensitivity analysis of large models," Environmental
  Modelling and Software, 22, 1509-1518.

## Authors

Wei-Yang Yu and V. Roshan Joseph.

## License

`softDOE` is licensed under the MIT License.
