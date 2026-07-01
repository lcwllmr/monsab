import numpy as np
import numpy.typing
import numpy
import typing

class SABBlock:
    rep_id: int
    dim: int
    e: int
    orbit_reps_flat: list[int]
    orbit_sizes: list[int]
    valid_cols: np.ndarray
    j_values: np.ndarray
    l_values: np.ndarray
    fs_indicator: int | None
    orbit_reps: list[int] | None
    d_k: int

    def __init__(
        self,
        rep_id: int,
        dim: int,
        e: int,
        orbit_reps_flat: list[int],
        orbit_sizes: list[int],
        valid_cols: list[int] | np.ndarray,
        j_values: list[int] | np.ndarray,
        l_values: list[int] | np.ndarray,
        fs_indicator: int | None = None,
        orbit_reps: typing.Any | None = None,
        d_k: int = 0,
    ) -> None: ...

class SABTransform:
    blocks: list[SABBlock]
    n_monomials: int
    N: int
    realize_skip_reps: set[int]

    def __init__(
        self,
        blocks: list[SABBlock] | tuple[SABBlock, ...] | None = None,
        N: int | None = None,
        _rust_transform: typing.Any | None = None,
        _realize_skip_reps: set[int] | None = None,
    ) -> None: ...
    def __call__(
        self,
        matrices: typing.Any,
        reynolds: bool = False,
        realize: bool = False,
    ) -> list[list[typing.Any]]: ...
    def explicit_basis(
        self, sparse: bool = True, realize: bool = False
    ) -> list[typing.Any]: ...
    def extract_batch(
        self,
        batch_data: list[np.ndarray],
        batch_indices: list[np.ndarray],
        batch_indptr: list[np.ndarray],
        reynolds: bool = False,
        realize: bool = False,
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
) -> SABTransform: ...
def compute_fs_and_t_data(
    paths: dict[int, list[tuple[int, list[int], int]]],
    orbits: list[tuple[int, ...]],
    is_squarefree: bool,
    n: int,
    d: int,
    coset_actions: dict[int, numpy.typing.NDArray[numpy.uint32]],
    rep_types: dict[int, int],
) -> tuple[
    dict[int, int], dict[int, dict[tuple[int, ...], tuple[float, list[int]]]]
]: ...
def evaluate_word(
    word: list[tuple[int, int]], generators: list[Permutation]
) -> Permutation: ...

class PcGroup:
    number_of_generators: int
    orders: list[int]
    conjugation_exponents: dict[tuple[int, int], int]
    power_tails: dict[int, list[tuple[int, int]]]
    conjugation_tails: dict[tuple[int, int], list[tuple[int, int]]]
    generators: list[list[int]]

    def __init__(
        self,
        number_of_generators: int,
        orders: list[int],
        conjugation_exponents: dict[tuple[int, int], int],
        power_tails: dict[int, list[tuple[int, int]]],
        conjugation_tails: dict[tuple[int, int], list[tuple[int, int]]],
        generators: list[list[int]],
    ) -> None: ...
    def order(self) -> int: ...
    def has_only_prime_factors(
        self, primes: list[int] | set[int] | tuple[int, ...]
    ) -> bool: ...
    def is_normal_series(self) -> bool: ...
    def verify(self) -> bool: ...

class OrbitLifter:
    def __init__(self, group: PcGroup, d: int, is_squarefree: bool) -> None: ...
    def clear_cache(self) -> None: ...
    def canonicalize(self, monomial: list[int]) -> list[int]: ...

class Permutation:
    def __init__(self, data: list[int] | tuple[int, ...]) -> None: ...
    @classmethod
    def identity(cls, n: int) -> Permutation: ...
    @property
    def data(self) -> tuple[int, ...]: ...
    @property
    def size(self) -> int: ...
    def __getitem__(self, index: int) -> int: ...
    def __mul__(self, other: Permutation) -> Permutation: ...
    def __pow__(self, p: int) -> Permutation: ...
    def __invert__(self) -> Permutation: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __getnewargs__(self) -> tuple[list[int]]: ...

class MonomialMatrix:
    def __init__(
        self,
        perm: list[int] | tuple[int, ...],
        vals: list[int] | tuple[int, ...],
        e: int,
    ) -> None: ...
    @classmethod
    def identity(cls, dim: int, e: int) -> MonomialMatrix: ...
    @property
    def perm(self) -> tuple[int, ...]: ...
    @property
    def vals(self) -> tuple[int, ...]: ...
    @property
    def e(self) -> int: ...
    @property
    def dim(self) -> int: ...
    def __add__(self, other: MonomialMatrix) -> MonomialMatrix: ...
    def __matmul__(self, other: MonomialMatrix) -> MonomialMatrix: ...
    def __pow__(self, p: int) -> MonomialMatrix: ...
    def inverse(self) -> MonomialMatrix: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __getnewargs__(self) -> tuple[list[int], list[int], int]: ...
