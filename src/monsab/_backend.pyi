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
    def apply_forward(
        self,
        matrices: typing.Any,
        reynolds: bool = False,
        realize: bool = False,
        reduced: bool = True,
        sparse: bool = True,
    ) -> list[list[typing.Any]]: ...
    def apply_inverse(
        self,
        blocks: typing.Any,
        realize: bool = False,
        sparse: bool = True,
    ) -> typing.Any: ...
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
    bundle: MonomialRepresentationBundle,
    g_inv_data: dict[int, list[int]],
    n: int,
    d: int,
    is_squarefree: bool,
    total_monomials: int,
    coset_actions: dict[int, numpy.typing.NDArray[numpy.uint32]] | None = None,
    coset_actions_inv: dict[int, numpy.typing.NDArray[numpy.uint32]] | None = None,
) -> SABTransform: ...
def compute_fs_and_t_data(
    g_gens: list[list[int]],
    n: int,
    e: int,
    h_visited_all: dict[int, dict[tuple[int, ...], int]],
    h_gens_fwd_all: dict[int, dict[int, list[int]]],
    char_phases_rep_all: dict[int, dict[int, int]],
) -> tuple[dict[int, int], dict[int, list[int] | None]]: ...
def evaluate_word(
    word: list[tuple[int, int]], generators: list[Permutation]
) -> Permutation: ...

class PcGroup:
    number_of_generators: int
    orders: list[int]
    conjugation_exponents: dict[tuple[int, int], int]
    power_tails: dict[int, list[tuple[int, int]]]
    conjugation_tails: dict[tuple[int, int], list[tuple[int, int]]]

    def __init__(
        self,
        number_of_generators: int,
        orders: list[int],
        conjugation_exponents: dict[tuple[int, int], int],
        power_tails: dict[int, list[tuple[int, int]]],
        conjugation_tails: dict[tuple[int, int], list[tuple[int, int]]],
    ) -> None: ...
    @property
    def order(self) -> int: ...
    @property
    def is_normal_series(self) -> bool: ...
    def test_generators(self, generators: list[typing.Any]) -> bool: ...
    def test_consistency(self) -> bool: ...
    @property
    def is_abelian(self) -> bool: ...
    def test_abelian(self) -> bool: ...
    @property
    def is_nilpotent(self) -> bool: ...
    def test_nilpotent(self) -> bool: ...
    @property
    def is_supersolvable(self) -> bool: ...
    def test_supersolvable(self) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class OrbitLifter:
    def __init__(
        self,
        group: PcGroup,
        generators: list[typing.Any],
        d: int,
        is_squarefree: bool,
    ) -> None: ...
    def clear_cache(self) -> None: ...
    def canonicalize(self, monomial: list[int]) -> list[int]: ...

class MonomialRepresentation:
    id: int
    dim: int
    e: int
    conjugate_id: int
    fs_indicator: int | None
    v_matrix: list[int] | None

    def __init__(
        self,
        id: int,
        dim: int,
        e: int,
        conjugate_id: int,
        fs_indicator: int | None = None,
        v_matrix: list[int] | None = None,
    ) -> None: ...

class MonomialRepresentationBundle:
    representations: list[MonomialRepresentation]
    paths_dict: dict[int, list[tuple[int, list[tuple[int, int]], int]]]
    fs_indicators: dict[int, int]
    v_matrices: dict[int, list[int]]
    realize_skip_reps: set[int]
    e: int

    def __init__(
        self,
        representations: list[MonomialRepresentation],
        paths_dict: dict[int, list[tuple[int, list[tuple[int, int]], int]]],
        fs_indicators: dict[int, int],
        v_matrices: dict[int, list[int]],
        realize_skip_reps: set[int],
        e: int,
    ) -> None: ...

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
