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
from monsab import _backend


@dataclass(frozen=True, slots=True)
class SABBlock:
    rep_id: int
    dim: int
    e: int
    orbit_reps_flat: tuple[int, ...]
    orbit_sizes: tuple[int, ...]
    col_to_j: np.ndarray | tuple[int, ...] | list[int]
    col_to_l: np.ndarray | tuple[int, ...] | list[int]
    orbit_reps: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class SABTransform:
    _rust_transform: _backend.RustSABTransform | None = None
    _blocks: tuple[SABBlock, ...] | None = None
    _N: int | None = None

    def __init__(
        self,
        blocks: tuple[SABBlock, ...] | None = None,
        N: int | None = None,
        _rust_transform: _backend.RustSABTransform | None = None,
    ):
        object.__setattr__(self, "_rust_transform", _rust_transform)
        object.__setattr__(self, "_blocks", blocks)
        object.__setattr__(self, "_N", N)

    @property
    def blocks(self) -> tuple[SABBlock, ...]:
        if self._blocks is not None:
            return self._blocks

        if self._rust_transform is None:
            return ()

        result = []
        for b in self._rust_transform.blocks:
            result.append(
                SABBlock(
                    rep_id=b.rep_id,
                    dim=b.dim,
                    e=b.e,
                    orbit_reps_flat=tuple(b.orbit_reps_flat),
                    orbit_sizes=tuple(b.orbit_sizes),
                    col_to_j=b.col_to_j,
                    col_to_l=b.col_to_l,
                )
            )
        return tuple(result)

    @property
    def N(self) -> int:
        if self._N is not None:
            return self._N
        if self._rust_transform is not None:
            return self._rust_transform.n_monomials
        return 0

    def __call__(
        self, matrices: list[scipy.sparse.csr_matrix] | scipy.sparse.csr_matrix
    ) -> list[list[scipy.sparse.csr_matrix]]:
        if not isinstance(matrices, list):
            matrices = [matrices]

        if self._rust_transform is None:
            return []

        # Use Rust to extract blocks natively
        results = self._rust_transform.extract_batch(
            [m.data for m in matrices],
            [m.indices for m in matrices],
            [m.indptr for m in matrices],
        )

        final_blocks = []
        for block_res in results:
            b_list = []
            for data, rows, cols, m_k in block_res:
                if len(data) == 0:
                    b_list.append(
                        scipy.sparse.csr_matrix((m_k, m_k), dtype=np.complex128)
                    )
                else:
                    mat = scipy.sparse.coo_matrix(
                        (data, (rows, cols)), shape=(m_k, m_k)
                    )
                    b_list.append(mat.tocsr())
            final_blocks.append(b_list)

        return final_blocks

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
        blocks = self.blocks
        if not blocks:
            return []

        N = self.N

        for block in self.blocks:
            m_k = block.dim
            e = block.e
            orbit_sizes = block.orbit_sizes

            col_to_j = np.asarray(block.col_to_j, dtype=np.uint32)
            col_to_l = np.asarray(block.col_to_l, dtype=np.uint32)

            valid_cols = np.where(col_to_j != 4294967295)[0]
            j_values = col_to_j[valid_cols]
            l_values = col_to_l[valid_cols]

            if sparse:
                vals = np.exp(-2j * np.pi * l_values / e) / np.sqrt(
                    np.array(orbit_sizes)[j_values]
                )
                U_k = scipy.sparse.coo_matrix(
                    (vals, (valid_cols, j_values)),
                    shape=(N, m_k),
                    dtype=np.complex128,
                )
                matrices.append(U_k.tocsr())
            else:
                U_k = np.zeros((N, m_k), dtype=np.complex128)
                for v, j, ell in zip(valid_cols, j_values, l_values):
                    val = cmath.exp(-2j * math.pi * ell / e) / math.sqrt(orbit_sizes[j])
                    U_k[v, j] = val
                matrices.append(U_k)

        return matrices
