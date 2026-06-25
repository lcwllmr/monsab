"""
Symmetry reduction tools for polynomial optimization.

This package provides a high-performance framework to compute the Symmetry Adapted Basis (SAB) for large monomial spaces using the generalized Baum-Clausen algorithm.

## Example Usage

The following example demonstrates how to block-diagonalize a batch of invariant matrices (intertwiners) under the action of the affine group:

```python
import scipy.sparse
import numpy as np

from monsab.zoo import affine_group_1d
from monsab.core import BaumClausenStage, BaumClausenPaths, Permutation
from monsab.pop import MonomialSpace, build_monomial_sab
from monsab.util import primitive_root

# 1. Define the group and its variables
n = 5
d = 2
group = affine_group_1d(n)
presentation = group.description

# 2. Run the Baum-Clausen algorithm over the polycyclic series
e_safe = np.prod(presentation.orders)
stages = [BaumClausenStage.trivial(e=e_safe, presentation=presentation)]
for k, order in enumerate(presentation.orders, start=1):
    stages.append(BaumClausenStage.next_level(stages[-1], g_i=k, p=order))
abstract_paths = BaumClausenPaths.from_baum_clausen(tuple(stages))

# 3. Generate explicit permutations for the physical space
g = primitive_root(n)
gen1 = Permutation(tuple((i + 1) % n for i in range(n)))
gen2 = Permutation(tuple((g * i) % n for i in range(n)))
concrete_generators = {1: gen1, 2: gen2}

# 4. Build the Monomial Space and extract orbits
space = MonomialSpace(n, d)
orbits = space.get_full_orbits(concrete_generators, num_threads=1)

# 5. Initialize the batched SAB Transform
transform = build_monomial_sab(
    abstract_paths,
    concrete_generators,
    orbits,
    space,
    num_threads=1
)

# 6. Apply the transform to a batch of matrices
N = space.total_monomials
T = scipy.sparse.eye(N, format="csr")  # Trivial intertwiner example
batch_matrices = [T, T]

# Returns a list of batches, where each entry corresponds to a distinct Isotypic Component block
block_diagonal_matrices = transform(batch_matrices)

print(f"Original Dimension: {N}")
print(f"Reduced Isotypic Blocks: {len(block_diagonal_matrices)}")
```
"""

__docformat__ = "google"
