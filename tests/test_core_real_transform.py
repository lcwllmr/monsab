import numpy as np
import scipy.sparse
from monsab.core import (
    BaumClausenStage,
    BaumClausenPaths,
    Permutation,
    PcGroup,
)
from monsab.pop import MonomialSpace, build_monomial_sab


def test_real_transform_cyclic_group():
    conjugation_exponents = {}
    conjugation_tails = {}
    presentation = PcGroup(
        number_of_generators=1,
        orders=(3,),
        conjugation_exponents=conjugation_exponents,
        power_tails={0: ()},
        conjugation_tails=conjugation_tails,
    )
    g = Permutation((1, 2, 0))
    G_gens = {1: g}

    stages = [BaumClausenStage.trivial(e=3, presentation=presentation)]
    for k, order in enumerate(presentation.orders, start=1):
        stages.append(BaumClausenStage.next_level(stages[-1], g_i=k, p=order))

    paths = BaumClausenPaths.from_baum_clausen(tuple(stages))
    space = MonomialSpace(3)
    orbits_space = [(0,), tuple(range(1, 4))]
    transform = build_monomial_sab(paths, G_gens, orbits_space, space, 1)

    # Construct a real symmetric matrix invariant under C3
    T_dense = np.array(
        [
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, -1.0, -1.0],
            [0.0, -1.0, 2.0, -1.0],
            [0.0, -1.0, -1.0, 2.0],
        ]
    )
    T_sparse = scipy.sparse.csr_matrix(T_dense)

    # 1. Complex transform
    blocks_cplx = [
        b_list[0] for b_list in transform.apply_forward(T_sparse, realize=False)
    ]

    # There are 3 blocks in the complex transform because realize=False
    assert len(blocks_cplx) == 3
    assert blocks_cplx[0].shape == (2, 2)
    assert blocks_cplx[1].shape == (1, 1)
    assert blocks_cplx[2].shape == (1, 1)

    # 2. Real transform
    blocks_real = [
        b_list[0] for b_list in transform.apply_forward(T_sparse, realize=True)
    ]

    # Expect: One 2x2 real block (fs=1), one 2x2 real block (fs=0)
    assert len(blocks_real) == 2
    b1, b2 = blocks_real
    assert b1.shape == (2, 2)
    assert b2.shape == (2, 2)

    assert np.isrealobj(b1.data)
    assert np.isrealobj(b2.data)

    # Check eigenvalues are preserved
    eig_orig = np.linalg.eigvalsh(T_dense)

    eig_blocks = []
    eig_blocks.extend(np.linalg.eigvalsh(b1.toarray()))
    eig_blocks.extend(np.linalg.eigvalsh(b2.toarray()))

    np.testing.assert_allclose(np.sort(eig_orig), np.sort(eig_blocks), atol=1e-10)


def test_real_transform_quaternionic():
    # Construct Q8
    def idx_Q(i, j):
        return i + 4 * j

    L_Q_x = [0] * 8
    L_Q_y = [0] * 8
    for i in range(4):
        for j in range(2):
            v = idx_Q(i, j)
            L_Q_x[v] = idx_Q((i + 1) % 4, j)
            new_i = (4 - i) % 4 if j == 0 else (6 - i) % 4
            L_Q_y[v] = idx_Q(new_i, (j + 1) % 2)

    g1 = Permutation(tuple(L_Q_x))
    g2 = Permutation(tuple(L_Q_y))

    conjugation_exponents = {(0, 1): 3}
    conjugation_tails = {(0, 1): ()}

    presentation = PcGroup(
        number_of_generators=2,
        orders=(4, 2),
        conjugation_exponents=conjugation_exponents,
        power_tails={0: (), 1: ((0, 2),)},
        conjugation_tails=conjugation_tails,
    )

    stages = [BaumClausenStage.trivial(e=4, presentation=presentation)]
    for k, order in enumerate(presentation.orders, start=1):
        stages.append(BaumClausenStage.next_level(stages[-1], g_i=k, p=order))

    paths = BaumClausenPaths.from_baum_clausen(tuple(stages))
    G_gens = {1: g1, 2: g2}
    space = MonomialSpace(8)
    orbits_space = [(0,), tuple(range(1, 9))]
    transform = build_monomial_sab(paths, G_gens, orbits_space, space, 1)

    # Build a real symmetric invariant matrix
    # Averaging a random matrix
    np.random.seed(42)
    M = np.random.randn(9, 9)
    M = M + M.T
    T_dense = np.zeros((9, 9))

    def apply_word(w):
        if w == 0:
            return Permutation.identity(8)
        if w == 1:
            return g1
        if w == 2:
            return g1 * g1
        if w == 3:
            return g1 * g1 * g1
        if w == 4:
            return g2
        if w == 5:
            return g1 * g2
        if w == 6:
            return g1 * g1 * g2
        if w == 7:
            return g1 * g1 * g1 * g2

    for w in range(8):
        p = apply_word(w)
        P_mat = np.zeros((9, 9))
        P_mat[0, 0] = 1.0  # scalar invariant
        for i, val in enumerate(p.data):
            P_mat[val + 1, i + 1] = 1.0
        T_dense += P_mat @ M @ P_mat.T

    T_dense /= 8.0
    T_sparse = scipy.sparse.csr_matrix(T_dense)

    blocks_real = [
        b_list[0] for b_list in transform.apply_forward(T_sparse, realize=True)
    ]

    # The SAB transform might merge some reps depending on abstract paths.
    # Q8 has four 1D irreps (fs=1) and one 2D irrep (fs=-1)
    # The 2D irrep should become a 4x4 real block
    shapes = [b.shape for b in blocks_real]
    print(shapes)
    # Check that there is one 4x4 block
    assert shapes.count((4, 4)) == 1

    for b in blocks_real:
        assert np.isrealobj(b.data)

    eig_orig = np.linalg.eigvalsh(T_dense)
    eig_blocks = []
    for b in blocks_real:
        eig_blocks.extend(np.linalg.eigvalsh(b.toarray()))

    np.testing.assert_allclose(np.sort(eig_orig), np.sort(eig_blocks), atol=1e-10)
