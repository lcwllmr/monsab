"""Symmetry reduction tools for polynomial optimization.

This package provides a high-performance framework to compute the Symmetry Adapted Basis (SAB) for large monomial spaces using the generalized Baum-Clausen algorithm.

## Example Usage

The following example demonstrates how to block-diagonalize a batch of invariant matrices (intertwiners) under the action of the affine group:

```python
import scipy.sparse
from monsab import MonomialSpace, Permutation, build_sab, zoo
from monsab.util import primitive_root

# 1. Define the physical space and symmetry group
n = 5
space = MonomialSpace(n=n, d=2)
group = zoo.affine_group_1d(n)

# 2. Generate explicit permutations for the physical space
g = primitive_root(n)
gen1 = Permutation(tuple((i + 1) % n for i in range(n)))
gen2 = Permutation(tuple((g * i) % n for i in range(n)))

# 3. Check generator consistency with abstract presentation
assert group.test_generators([gen1, gen2])

# 4. Build the Symmetry-Adapted Basis transform in one step!
transform = build_sab(space, group, [gen1, gen2])

# 5. Apply the transform to a batch of matrices
N = space.total_monomials()
T = scipy.sparse.eye(N, format="csr")  # Trivial intertwiner example
block_diagonal_matrices = transform([T, T])

print(f"Original Dimension: {N}")
print(f"Reduced Isotypic Blocks: {len(block_diagonal_matrices)}")
```
"""

from . import zoo
from .core import PcGroup, Permutation
from .pop import (
    LasserreHierarchy,
    MonomialSpace,
    SquarefreeMonomialSpace,
    build_sab,
)

__all__ = [
    "LasserreHierarchy",
    "MonomialSpace",
    "PcGroup",
    "Permutation",
    "SquarefreeMonomialSpace",
    "build_sab",
    "zoo",
]

__docformat__ = "google"
