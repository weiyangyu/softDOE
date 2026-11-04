# softDOE

**softDOE** creates space-filling one-factor-at-a-time (SOFT) designs for factor
screening and surrogate modeling. Use it to generate experimental designs,
identify influential factors, and add runs to an existing design.

The method is described in [Space-Filling One-Factor-At-A-Time Designs](https://arxiv.org/abs/2606.02533)
by Wei-Yang Yu and V. Roshan Joseph (2026).

## Installation

Requires **Python 3.9 or newer**.

```bash
python -m pip install "softDOE>=0.2.0"
```

NumPy, SciPy, and PyDOE are installed automatically. If a compatible binary
wheel is unavailable, installation builds the package from source and requires
a C++17 compiler: Xcode Command Line Tools on macOS, GCC or Clang with Python
development headers on Linux, or Visual Studio Build Tools on Windows.

CPU support is included on macOS, Linux, and Windows. Apple Metal acceleration
is included on supported Macs and does not require PyTorch. For NVIDIA CUDA
on Linux, install a CUDA-enabled build of PyTorch 2.8 or newer using the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Quick start

```python
import softDOE

result = softDOE.soft(p=10, l=4, structure="strict", seed=123)
design = result["design"]

print(design.shape)          # (44, 10): 44 runs for 10 factors
print(result["Q"])          # Design quality
print(result["backend_used"])
```

Each row is an experimental run and each column is a factor. Coordinates lie
in `(0, 1)`; scale each column to the corresponding factor's range before
running your experiment or simulation.

## Functions

| Function | Purpose |
| --- | --- |
| `soft()` | Generate a SOFT design. |
| `a_soft()` | Generate an augmented design or add runs to an existing design. |
| `measure()` | Estimate factor importance from a design and its responses. |
| `qmc_generate()` | Generate Sobol evaluation points with boundary points. |
| `Q()` | Evaluate the space-filling quality of a design. |
| `backend_info()` | Check CPU, Apple Metal, and NVIDIA CUDA availability. |

### Generate a design with `soft()`

`soft(p, l, structure="standard", m=500, seed=None, backend="auto")` generates
`l * (p + 1)` runs.

- `p`: number of factors, from 2 to 1024.
- `l`: number of base runs; an even integer of at least 2.
- `structure`: `"standard"` compares each factor block with the base block;
  `"strict"` changes one factor between consecutive blocks.
- `m`: number of Sobol evaluation points used in the search, before adding
  boundary points.
- `seed`: random seed for design initialization.
- `backend`: `"auto"`, `"cpu"`, `"metal"`, or `"cuda"`.

The result is a dictionary containing `design`, `Q`, `normalized_Q`,
`qmc_size`, `backend_used`, `device`, and `fallback_reason`.

### Screen factors with `measure()`

Evaluate your model at each design row, then pass the responses in the same
row order to `measure(design, y)`. For example:

```python
import numpy as np
import softDOE

D = softDOE.soft(p=10, l=4, seed=123)["design"]
y = (
    10 * np.sin(np.pi * D[:, 0] * D[:, 1])
    + 20 * (D[:, 2] - 0.5)**2
    + 10 * D[:, 3]
    + 5 * D[:, 4]
)

effects = softDOE.measure(D, y)
print(effects["mustar"])     # Mean absolute elementary effects
print(effects["t"])          # Total Sobol' index estimates
```

Both outputs contain one value per factor, in column order. `measure()`
supports standard and strict designs. Keep the original OFAT row ordering;
after augmentation, use only the original OFAT rows and their responses for
screening. Constant responses give `NaN` for `t`.

### Add runs with `a_soft()`

Augmentation adds runs for further exploration or surrogate modeling. Create
a SOFT design and augment it in one call:

```python
import softDOE

augmented = softDOE.a_soft(p=2, l=4, n_aug=8, seed=123)
print(augmented["design"].shape)  # (20, 2): 12 initial runs + 8 new runs
```

Or add runs to a design you already have:

```python
import softDOE

base = softDOE.soft(p=2, l=4, seed=123)["design"]
extended = softDOE.a_soft(n_aug=8, ofat_design=base)
new_runs = extended["design"][len(base):]
print(new_runs.shape)            # (8, 2)
```

The original rows are preserved exactly. A supplied design must lie in
`[0, 1]` and have `l * (p + 1)` rows for an even `l >= 2` and `p >= 2`.
Its shape determines `p` and `l`; initialization arguments are ignored.

`n_aug` is a positive integer no greater than `N`, the candidate-pool size
(default `10000`). When generating the initial design, `structure`, `m`, and
`seed` have the same roles as in `soft()`. Added runs need not follow an OFAT
structure. The returned dictionary contains `design`, `Q`, `normalized_Q`,
`backend_used`, `device`, and `fallback_reason`; scores describe the complete
augmented design.

### Evaluate quality with `Q()` and `qmc_generate()`

```python
import softDOE

D = softDOE.soft(p=2, l=4, seed=123)["design"]
points = softDOE.qmc_generate(n=1000, p=2)
scores = softDOE.Q(alpha=1, beta=1 / (2 * 4)**2, design=D, qmc=points)

print(points.shape)              # (1004, 2): Sobol points plus corners
print(scores["Q"], scores["normalized_Q"])
```

Larger `Q` indicates better space-filling coverage when comparing designs
with the **same kernel parameters and evaluation points**. `alpha` and `beta`
are positive kernel parameters; `soft()` uses `alpha=1` and
`beta=(2*l)**-2`. Augmentation uses `beta=(2*l+n_aug)**-2`.
`normalized_Q` is `Q**(1/p)`, computed in log space to preserve small values.
`Q()` also returns `backend_used`.

`qmc_generate(n, p, threshold=10)` returns a deterministic NumPy array of
Sobol points and boundary points. It adds all corners when `p <= threshold`
and representative opposite corners otherwise, removing duplicate rows.
The final number of rows can therefore differ from `n`. To reproduce a
`soft()` score, use the same `m`, `p`, and kernel parameters.

## CPU and GPU execution

The default `backend="auto"` selects an available Apple Metal backend on macOS
or NVIDIA CUDA backend on Linux, otherwise CPU. Pass `backend="cpu"` to
request CPU execution explicitly.

```python
import softDOE

print(softDOE.backend_info())
```

If a requested GPU is unavailable or execution fails, softDOE falls back to
CPU. Check `backend_used` in the result; `soft()` and `a_soft()` also provide
`fallback_reason`. CPU and CUDA use double-precision scoring; Metal uses
single-precision scoring. Numerical differences can lead to different designs
across backends. For reproducible runs, record the seed, package version, and
backend used.

See [more examples](https://github.com/weiyangyu/softDOE/blob/main/examples/mofat_examples.py)
for complete workflows.

## Citation

If you use **softDOE** in your research, please cite the package:

Yu, W.-Y. and Joseph, V. R. (2026). *softDOE* (version 0.2.0).
[Python package](https://github.com/weiyangyu/softDOE).

```bibtex
@software{yu2026softdoe,
  title   = {softDOE},
  author  = {Yu, Wei-Yang and Joseph, V. Roshan},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/weiyangyu/softDOE}
}
```

## Acknowledgments

This work is supported by U.S. NSF grant DMS-2310637.

## License

softDOE is distributed under the [MIT License](https://github.com/weiyangyu/softDOE/blob/main/LICENSE).
Redistributed Sobol data and the index permutation carry additional notices
in [THIRD_PARTY_NOTICES.txt](https://github.com/weiyangyu/softDOE/blob/main/src/softDOE/THIRD_PARTY_NOTICES.txt).
