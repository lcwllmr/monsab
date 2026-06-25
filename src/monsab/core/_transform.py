"""
Transform components.
"""

from dataclasses import dataclass
import typing
import math
import cmath

import numpy as np
import numpy.typing as npt
import scipy.sparse


@dataclass(frozen=True, slots=True)
class SABBlock:
    rep_id: int
    dim: int
    e: int
    orbit_reps: tuple[int, ...]
    orbit_reps_flat: tuple[int, ...]
    orbit_sizes: tuple[int, ...]
    orbit: tuple[int, ...]
    col_to_j: tuple[int, ...]
    col_to_l: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SABTransform:
    blocks: tuple[SABBlock, ...]
    N: int

    def __call__(
        self, matrices: typing.Sequence[scipy.sparse.csr_matrix]
    ) -> list[list[scipy.sparse.csr_matrix]]:
        if not self.blocks:
            return []

        result = []
        for block in self.blocks:
            m_k = block.dim
            e = block.e
            col_to_j = np.array(block.col_to_j, dtype=np.int32)
            col_to_l = np.array(block.col_to_l, dtype=np.int32)
            orbit_sizes = np.array(block.orbit_sizes, dtype=np.float64)
            orbit_reps_flat = list(block.orbit_reps_flat)

            phase_map = np.exp(-2j * np.pi * np.arange(e) / e)

            block_batch = []
            for T in matrices:
                T_reps = T[orbit_reps_flat, :]
                T_reps_coo = T_reps.tocoo()

                rows = T_reps_coo.row
                cols = T_reps_coo.col
                data = T_reps_coo.data

                j = col_to_j[cols]
                valid = j != -1

                if not np.any(valid):
                    block_batch.append(
                        scipy.sparse.coo_matrix((m_k, m_k), dtype=np.complex128).tocsr()
                    )
                    continue

                rows_valid = rows[valid]
                j_valid = j[valid]
                cols_valid = cols[valid]
                data_valid = data[valid]
                l_valid = col_to_l[cols_valid]

                phases = phase_map[l_valid]
                norms = np.sqrt(orbit_sizes[rows_valid] / orbit_sizes[j_valid])
                vals = data_valid * norms * phases

                T_k = scipy.sparse.coo_matrix(
                    (vals, (rows_valid, j_valid)),
                    shape=(m_k, m_k),
                    dtype=np.complex128,
                ).tocsr()

                # Symmetrize numerical noise if input was real symmetric
                # Note: keeping it general, so we just append
                block_batch.append(T_k)

            result.append(block_batch)

        return result

    @typing.overload
    def explicit_basis(
        self, sparse: typing.Literal[True] = True
    ) -> list[scipy.sparse.csr_matrix]: ...

    @typing.overload
    def explicit_basis(
        self, sparse: typing.Literal[False]
    ) -> list[npt.NDArray[np.complex128]]: ...

    def explicit_basis(self, sparse: bool = True) -> typing.Any:
        matrices = []
        if not self.blocks:
            return []

        N = self.N

        for block in self.blocks:
            m_k = block.dim
            e = block.e
            col_to_j = block.col_to_j
            col_to_l = block.col_to_l
            orbit_sizes = block.orbit_sizes

            if sparse:
                rows = []
                cols = []
                data = []

                for v, j in enumerate(col_to_j):
                    if j == -1:
                        continue

                    ell = col_to_l[v]
                    val = cmath.exp(-2j * math.pi * ell / e) / math.sqrt(orbit_sizes[j])
                    rows.append(v)
                    cols.append(j)
                    data.append(val)

                U_k = scipy.sparse.csr_matrix(
                    (data, (rows, cols)), shape=(N, m_k), dtype=np.complex128
                )
                matrices.append(U_k)
            else:
                U_k = np.zeros((N, m_k), dtype=np.complex128)
                for v, j in enumerate(col_to_j):
                    if j == -1:
                        continue

                    ell = col_to_l[v]
                    val = cmath.exp(-2j * math.pi * ell / e) / math.sqrt(orbit_sizes[j])
                    U_k[v, j] = val
                matrices.append(U_k)

        return matrices
