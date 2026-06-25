# `monsab`: Fast symmetry-adapted basis transform for monomial groups

[![ci](https://github.com/lcwllmr/monsab/actions/workflows/ci.yml/badge.svg)](https://github.com/lcwllmr/monsab/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lcwllmr/monsab/graph/badge.svg?token=RI56K0HE76)](https://codecov.io/gh/lcwllmr/monsab)

> [!TIP]
> Preprint coming soon! Will be linked here, so check back later!

> [!WARNING]
> The package is in early development and, while the core functionality is in place, many of the advertised features in this readme are still in development.
> Moreover, there is a good chance that I will do some major design changes down the road.

This package implements a fast symmetry-adapted basis transform for signed permutation actions of monomial groups.
The "fast" comes from the special structure of such groups allowing for significant speed-ups when computing its irreducible representations and the fine symmetry-adapted basis transform, which block-diagonalizes commuting matrices.
A very good analogy is the dramatic `O(n log n)` speed-up of the fast Fourier transform over conventional matrix-vector multiplication in `O(n^2)`.
Basically, if a matrix `T` commutes with all generating signed permutations of the group, then this package computes the (representation-theoretically) finest possible block-diagonalization `T = ⨁  T_i`.
However, instead of explicitly computing the basis and performing matrix-vector products, it extracts blocks directly by exploiting the special monomial representation structure.

While it is built for general application, the main intended use cases are in semidefinite programming and polynomial optimization via Lasserre's hierarchy.
Here is a simple but complete example:

```python
# TODO
```

Check out the [examples directory](./examples/) for interesting use cases that demonstrate the more specialized features of the package:

- Many predefined families of monomial groups ready to go which all support fast automatic computation of all required data for SAB transforms
- Tools for generating fully reduced SDPs for Lasserre's hierarchy with support for sign symmetries, 0-1 programming and reduction of equality constraints (the latter WIP)
- Exploit larger groups in the computation of invariant moment matrices even if they are not monomial themselves
- Various handy tests e.g. for verifying that a given polynomial is really invariant, or to check whether a given group is supersolvable
- Automatic computation of purely real blocks in cases where that is possible
- Heuristic algorithms for finding symmetries in a given set of polynomials and for finding large monomial groups in arbitrary permutation groups

# Installation

**Install package for use**:
Currently, you can only install the package right from GitHub.
A PyPI version will be available once development reaches a more stable point.
In any case, Python version at least 3.14 is required to make use of all modern language features.

```bash
pip install git+https://github.com/lcwllmr/monsab.git
```

**Setup for local development and/or running examples/benchmarks**:
For local development of the package itself or to play with the examples or benchmarks you only need to make sure that the Python version manager [`uv`](https://docs.astral.sh/uv/) is installed (no global Python install required).
After cloning, you can run the following commands:

```bash
# install dev dependencies
uv sync

# run the test suite and generate a coverage report
uv run pytest --cov=monsab

# run an example
uv run examples/

# run a benchmarks on your machine e.g.
uv run benchmarks/sab_transform_sparse.py

# contributors: install pre-commit hooks for auto-formatting, linting etc.
uv run pre-commit install
```

# API overview

# Benchmark

The great thing about focusing on monomial groups is that the hard parts of the SAB transform are simplified at all levels.


# Changelog

Work in progress:

`v0.1.0` (2026-06-22): first version

- 

# Contributing

For now, while I'm still working on the core design of the package, I have disabled pull requests until that stabilizes.
If you have ideas for improvements or other comments open an issue or contact me privately.
Always happy to discuss!
