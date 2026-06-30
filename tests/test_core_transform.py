import numpy as np
import scipy.sparse
import scipy.linalg
from monsab.core import (
    BaumClausenStage,
    PolycyclicPresentation,
    Permutation,
    BaumClausenPaths,
)
from monsab.pop import MonomialSpace, build_monomial_sab


def test_sab_transform_s3():
    # S3 acting on {0, 1, 2}
    g1 = Permutation((1, 2, 0))  # (0 1 2)  order 3
    g2 = Permutation((1, 0, 2))  # (0 1)    order 2

    # 1-based indexing for generators
    concrete_generators = {
        1: g1,
        2: g2,
    }

    presentation = PolycyclicPresentation(
        number_of_generators=2,
        orders=(3, 2),
        conjugation_exponents={(0, 1): 2},
        power_tails={},
        conjugation_tails={},
    )

    assert presentation.verify((g1, g2))

    # Run Baum-Clausen
    e = 6  # max order of elements is 3, but 6 is a multiple
    stages = [BaumClausenStage.trivial(e, presentation)]
    stages.append(BaumClausenStage.next_level(stages[-1], 1, 3))  # g_1, p=3
    stages.append(BaumClausenStage.next_level(stages[-1], 2, 2))  # g_2, p=2

    paths = BaumClausenPaths.from_baum_clausen(tuple(stages))
    space = MonomialSpace(3, 1)
    orbits = [tuple(range(1, 4))]
    transform = build_monomial_sab(paths, concrete_generators, orbits, space)

    assert len(transform.blocks) >= 2

    rep_map = {r.id: r for r in stages[-1].representations}
    total_dim = 0
    for block in transform.blocks:
        total_dim += block.dim * rep_map[block.rep_id].dim
    assert total_dim == 3


def test_transform_edge_cases():
    from monsab.core._transform import SABTransform, SABBlock

    empty_transform = SABTransform(blocks=(), N=0)
    assert empty_transform([]) == []
    assert empty_transform.explicit_basis() == []

    # Test sparse explicit basis
    b = SABBlock(
        rep_id=0,
        dim=1,
        e=2,
        orbit_reps=(0,),
        orbit_reps_flat=(0,),
        orbit_sizes=(1,),
        valid_cols=[0],
        j_values=[0],
        l_values=[0],
    )
    t = SABTransform(blocks=(b,), N=1)

    basis = t.explicit_basis(sparse=True)
    assert len(basis) == 1
    assert basis[0].shape == (1, 1)

    # Test valid filter
    # Provide a matrix that maps to no valid j (using j=4294967295 which is u32::MAX)
    b2 = SABBlock(
        rep_id=0,
        dim=1,
        e=2,
        orbit_reps=(0,),
        orbit_reps_flat=(0,),
        orbit_sizes=(1,),
        valid_cols=[1],
        j_values=[0],
        l_values=[0],
    )
    t2 = SABTransform(blocks=(b2,), N=2)
    basis = t2.explicit_basis(sparse=True)
    assert len(basis) == 1
    t2.explicit_basis(sparse=False)


def test_explicit_basis_orthogonality():
    # S3 acting on 3 elements
    g1 = Permutation((1, 2, 0))  # (0 1 2) order 3
    g2 = Permutation((1, 0, 2))  # (0 1)   order 2

    concrete_generators = {
        1: g1,
        2: g2,
    }

    presentation = PolycyclicPresentation(
        number_of_generators=2,
        orders=(3, 2),
        conjugation_exponents={(0, 1): 2},
        power_tails={},
        conjugation_tails={},
    )
    e = 6
    stages = [BaumClausenStage.trivial(e, presentation)]
    stages.append(BaumClausenStage.next_level(stages[-1], 1, 3))
    stages.append(BaumClausenStage.next_level(stages[-1], 2, 2))

    from monsab.core import BaumClausenPaths

    paths = BaumClausenPaths.from_baum_clausen(tuple(stages))

    space = MonomialSpace(3, 1)
    orbits = [tuple(range(1, 4))]

    transform = build_monomial_sab(paths, concrete_generators, orbits, space)

    # We test the explicit basis
    matrices = transform.explicit_basis(sparse=False)

    # The sum of m_k * d_k should be the dimension of the space, which is 3 here.
    rep_map = {r.id: r for r in stages[-1].representations}
    total_dim = 0
    for block, U_k in zip(transform.blocks, matrices):
        total_dim += U_k.shape[1] * rep_map[block.rep_id].dim
    assert total_dim == 3

    # Check orthogonality within each block
    for U_k in matrices:
        if U_k.shape[1] == 0:
            continue
        prod = U_k.conj().T @ U_k
        assert np.allclose(prod, np.eye(U_k.shape[1]), atol=1e-10)

    # Check orthogonality between different blocks
    for i, U_k in enumerate(matrices):
        for j, U_l in enumerate(matrices):
            if i != j and U_k.shape[1] > 0 and U_l.shape[1] > 0:
                prod = U_k.conj().T @ U_l
                assert np.allclose(
                    prod, np.zeros((U_k.shape[1], U_l.shape[1])), atol=1e-10
                )


def test_explicit_basis_dft_equivalence():
    # Cyclic group C3
    g = Permutation((1, 2, 0))
    concrete_generators = {1: g}

    presentation = PolycyclicPresentation(
        number_of_generators=1,
        orders=(3,),
        conjugation_exponents={},
        power_tails={},
        conjugation_tails={},
    )
    e = 3
    stages = [BaumClausenStage.trivial(e, presentation)]
    stages.append(BaumClausenStage.next_level(stages[-1], 1, 3))

    from monsab.core import BaumClausenPaths

    paths = BaumClausenPaths.from_baum_clausen(tuple(stages))

    space = MonomialSpace(3, 1)
    orbits = [tuple(range(1, 4))]

    transform = build_monomial_sab(paths, concrete_generators, orbits, space)
    matrices = transform.explicit_basis(sparse=False)

    # The transform should give 3 blocks of dimension 1x1
    assert len(matrices) == 3
    for U_k in matrices:
        assert U_k.shape == (4, 1)

    # Stack them into a 3x3 matrix
    U = np.hstack(matrices)

    # The standard scipy DFT matrix
    dft = scipy.linalg.dft(3) / np.sqrt(3)

    # Due to phase conventions and irrep order, the columns of U
    # should be a permutation of the columns of the DFT matrix,
    # possibly conjugated.

    for i in range(3):
        col_matched = False
        for j in range(3):
            # Check if U[1:, i] matches any column in dft (or its conjugate)
            if np.allclose(U[1:, i], dft[:, j], atol=1e-10) or np.allclose(
                U[1:, i], dft[:, j].conj(), atol=1e-10
            ):
                col_matched = True
                break
        assert col_matched
    assert np.allclose(U.conj().T @ U, np.eye(3), atol=1e-10)

    # We check that applying U to a circulant matrix block-diagonalizes it.
    circulant_sub = np.array(
        [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.complex128
    )
    circulant = np.zeros((4, 4), dtype=np.complex128)
    circulant[1:, 1:] = circulant_sub

    diagonalized = U.conj().T @ circulant @ U

    # It must be diagonal
    off_diag = diagonalized - np.diag(np.diag(diagonalized))
    assert np.allclose(off_diag, 0, atol=1e-10)

    # The eigenvalues should be 0, 3, 3
    eigenvalues = np.sort(np.real(np.diag(diagonalized)))
    assert np.allclose(eigenvalues, [0, 3, 3], atol=1e-10)


def test_fast_transform_equivalence():
    # We will build the S3 group representation and transform a generic symmetric matrix
    # to see if the batched __call__ output matches U^dag T U.
    g1 = Permutation((1, 2, 0))  # (0 1 2) order 3
    g2 = Permutation((1, 0, 2))  # (0 1)   order 2

    concrete_generators = {
        1: g1,
        2: g2,
    }

    presentation = PolycyclicPresentation(
        number_of_generators=2,
        orders=(3, 2),
        conjugation_exponents={(0, 1): 2},
        power_tails={},
        conjugation_tails={},
    )
    e = 6
    stages = [BaumClausenStage.trivial(e, presentation)]
    stages.append(BaumClausenStage.next_level(stages[-1], 1, 3))
    stages.append(BaumClausenStage.next_level(stages[-1], 2, 2))

    paths = BaumClausenPaths.from_baum_clausen(tuple(stages))

    space = MonomialSpace(3, 1)
    orbits = [tuple(range(1, 4))]

    transform = build_monomial_sab(paths, concrete_generators, orbits, space)

    # We generate a generic real symmetric commutant matrix T
    # S3 acting on 3 elements has a 2-dimensional commutant algebra:
    # 1. the identity matrix
    # 2. the all-ones matrix without diagonal (J - I)

    T1 = np.eye(3)
    T2 = np.ones((3, 3)) - np.eye(3)
    T_dense_sub = 1.5 * T1 + 0.5 * T2

    # Pad to full 4x4 space
    T_dense = np.zeros((4, 4))
    T_dense[1:, 1:] = T_dense_sub
    T_sparse = scipy.sparse.csr_matrix(T_dense)

    # Explicit basis
    explicit_matrices = transform.explicit_basis(sparse=False)

    # Fast transform
    fast_blocks_batch = transform([T_sparse])

    for i, block in enumerate(transform.blocks):
        U_k = explicit_matrices[i]
        if U_k.shape[1] == 0:
            continue

        expected_T_k = U_k.conj().T @ T_dense @ U_k
        fast_T_k = fast_blocks_batch[i][0].toarray()

        assert np.allclose(expected_T_k, fast_T_k, atol=1e-10)
