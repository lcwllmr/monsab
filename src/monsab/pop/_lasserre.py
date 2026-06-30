"""
Lasserre hierarchy integration utilities.
"""

import itertools
from collections.abc import Mapping
import numpy as np
from scipy.sparse import csr_matrix

from monsab.core import Permutation
from ._monomials import MonomialSpace, SquarefreeMonomialSpace


class LasserreHierarchy:
    """
    Generates sparse moment matrices for invariant Lasserre hierarchy SDPs.
    """

    def __init__(
        self,
        space: MonomialSpace | SquarefreeMonomialSpace,
        t: int,
        generators: Mapping[int, Permutation],
    ):
        if space.d < 2 * t:
            raise ValueError(
                f"Monomial space degree {space.d} must be at least 2*t ({2 * t})"
            )

        self.space2t = space
        self.t = t
        self.generators = generators

        # Subspace for polynomials up to degree t
        self.space_t = space.__class__(space.n, t)
        self.N_t = self.space_t.total_monomials

    def moment_matrix(
        self, orbit: tuple[int, ...], reynolds: bool = False
    ) -> csr_matrix:
        """
        Produces a scipy sparse matrix of size N_t x N_t which has ones at index (i, j)
        if and only if the associated monomials multiplied are part of the given orbit.
        If reynolds=True, only the moment matrix for the first representative of the orbit
        is constructed, suitable for passing to SAB transform with reynolds=True.
        """
        rows = []
        cols = []

        # Iterate over each monomial in the orbit (or just the representative if reynolds=True)
        orbit_to_iterate = [orbit[0]] if reynolds else orbit
        for m_rank in orbit_to_iterate:
            tup = self.space2t.unrank_tuple(m_rank)
            k = len(tup)

            if isinstance(self.space2t, SquarefreeMonomialSpace):
                # For squarefree, we need all pairs of subsets (I, J) such that I union J == tup
                # Each element in tup can belong to I only, J only, or both I and J.
                # So there are 3^k pairs.
                for p in itertools.product([1, 2, 3], repeat=k):
                    tup1_list = []
                    tup2_list = []
                    for idx, val in enumerate(tup):
                        if p[idx] == 1:
                            tup1_list.append(val)
                        elif p[idx] == 2:
                            tup2_list.append(val)
                        else:
                            tup1_list.append(val)
                            tup2_list.append(val)
                    if len(tup1_list) <= self.t and len(tup2_list) <= self.t:
                        r1 = self.space_t.rank_tuple(tuple(tup1_list))
                        r2 = self.space_t.rank_tuple(tuple(tup2_list))
                        rows.append(r1)
                        cols.append(r2)
            else:
                seen = set()
                min_k1 = max(0, k - self.t)
                max_k1 = min(k, self.t)

                for k1 in range(min_k1, max_k1 + 1):
                    for indices in itertools.combinations(range(k), k1):
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
