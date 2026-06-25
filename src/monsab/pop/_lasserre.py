"""
Lasserre hierarchy integration utilities.
"""

import itertools
from collections.abc import Mapping
import numpy as np
from scipy.sparse import csr_matrix

from monsab.core import Permutation
from ._monomials import MonomialSpace


class LasserreHierarchy:
    """
    Generates sparse moment matrices for invariant Lasserre hierarchy SDPs.
    """

    def __init__(
        self, space: MonomialSpace, t: int, generators: Mapping[int, Permutation]
    ):
        if space.d < 2 * t:
            raise ValueError(
                f"Monomial space degree {space.d} must be at least 2*t ({2 * t})"
            )

        self.space2t = space
        self.t = t
        self.generators = generators

        # Subspace for polynomials up to degree t
        self.space_t = MonomialSpace(space.n, t)
        self.N_t = self.space_t.total_monomials

    def moment_matrix(self, orbit: tuple[int, ...]) -> csr_matrix:
        """
        Produces a scipy sparse matrix of size N_t x N_t which has ones at index (i, j)
        if and only if the associated monomials multiplied are part of the given orbit.
        """
        rows = []
        cols = []

        # Iterate over each monomial in the orbit
        for m_rank in orbit:
            tup = self.space2t.unrank_tuple(m_rank)
            k = len(tup)

            # Factor the tuple (multiset) into two subsets of size <= t
            seen = set()
            min_k1 = max(0, k - self.t)
            max_k1 = min(k, self.t)

            for k1 in range(min_k1, max_k1 + 1):
                # Iterate over all combinations of indices to form tup1
                for indices in itertools.combinations(range(k), k1):
                    # We can extract elements efficiently
                    indices_set = set(indices)
                    tup1_list = []
                    tup2_list = []
                    for idx, val in enumerate(tup):
                        if idx in indices_set:
                            tup1_list.append(val)
                        else:
                            tup2_list.append(val)

                    tup1 = tuple(tup1_list)
                    tup2 = tuple(tup2_list)

                    pair = (tup1, tup2)
                    if pair not in seen:
                        seen.add(pair)
                        r1 = self.space_t.rank_tuple(tup1)
                        r2 = self.space_t.rank_tuple(tup2)
                        rows.append(r1)
                        cols.append(r2)

        data = np.ones(len(rows), dtype=float)
        return csr_matrix((data, (rows, cols)), shape=(self.N_t, self.N_t))
