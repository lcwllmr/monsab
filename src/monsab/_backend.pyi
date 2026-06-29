import numpy as np
import numpy.typing
import numpy

class RustSABBlock:
    rep_id: int
    dim: int
    e: int
    orbit_reps_flat: list[int]
    orbit_sizes: list[int]
    col_to_j: np.ndarray
    col_to_l: np.ndarray

class RustSABTransform:
    blocks: list[RustSABBlock]
    n_monomials: int

    def extract_batch(
        self,
        batch_data: list[np.ndarray],
        batch_indices: list[np.ndarray],
        batch_indptr: list[np.ndarray],
    ) -> list[list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]]: ...

def build_sab_blocks(
    orbits: list[tuple[int, ...]],
    paths: dict[int, list[tuple[int, list[int], int]]],
    g_inv_data: dict[int, list[int]],
    n: int,
    d: int,
    e: int,
    is_squarefree: bool,
    total_monomials: int,
    coset_actions: dict[int, numpy.typing.NDArray[numpy.uint32]] | None = None,
    coset_actions_inv: dict[int, numpy.typing.NDArray[numpy.uint32]] | None = None,
) -> RustSABTransform: ...
