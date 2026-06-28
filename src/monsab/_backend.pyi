import numpy as np

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
    orbits_dict: dict[int, list[int]],
    abstract_paths_dict: dict[int, list[tuple[int, list[tuple[int, int]], int]]],
    g_gens_dict: dict[int, list[int]],
    n: int,
    d: int,
    e: int,
    is_squarefree: bool,
    n_monomials: int,
) -> RustSABTransform: ...
