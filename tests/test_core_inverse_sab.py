import numpy as np
import scipy.sparse

from monsab import build_sab
from monsab.core import Permutation
from monsab.pop import MonomialSpace
from monsab.zoo import cyclic, dihedral


def get_monomial_permutation_matrix(g: Permutation, space: MonomialSpace) -> np.ndarray:
    """Helper to generate an N x N permutation matrix on the monomial space from a domain Permutation."""
    N = space.total_monomials()
    P = np.zeros((N, N), dtype=np.float64)
    g_inv_data = (~g).data
    for j in range(N):
        i = space.apply_gen(j, g_inv_data)
        P[i, j] = 1.0
    return P


def project_to_intertwiner(
    M: np.ndarray, g_list: list[Permutation], space: MonomialSpace
) -> np.ndarray:
    """Project matrix M onto the commutant algebra by averaging over group elements."""
    n = space.n
    visited = {tuple(range(n)): Permutation(tuple(range(n)))}
    queue = [tuple(range(n))]
    while queue:
        curr = queue.pop(0)
        curr_perm = visited[curr]
        for g in g_list:
            next_perm = curr_perm * g
            if next_perm.data not in visited:
                visited[next_perm.data] = next_perm
                queue.append(next_perm.data)

    T = np.zeros_like(M)
    for g_perm in visited.values():
        P_g = get_monomial_permutation_matrix(g_perm, space)
        T += P_g @ M @ P_g.T
    return T / len(visited)


def test_inverse_sab_roundtrip_reduced():
    """Verify that forward(reduced=True) -> inverse is exact for intertwiner matrices."""
    n = 3
    space = MonomialSpace(n=n, d=2)  # N = 10 monomials
    group = dihedral(n)

    # Generators for D_3 ~ S_3 acting on 3 elements
    g1 = Permutation((1, 2, 0))
    g2 = Permutation((0, 2, 1))
    id_perm = Permutation((0, 1, 2))
    coset_reps = {4: [id_perm], 6: [id_perm, g2]}

    sab = build_sab(space, group, [g1, g2], coset_reps=coset_reps)
    N = space.total_monomials()

    # Create a random symmetric intertwiner matrix
    rng = np.random.default_rng(42)
    M_dense = rng.standard_normal((N, N))
    M_dense = 0.5 * (M_dense + M_dense.T)
    T_dense_init = project_to_intertwiner(M_dense, [g1, g2], space)
    T_sparse_init = scipy.sparse.csr_matrix(T_dense_init)

    # 1. Get reduced blocks
    blocks_red = sab.apply_forward([T_sparse_init], reynolds=False, reduced=True)

    # 2. Reconstruct intertwiner T via apply_inverse
    T_sparse = sab.apply_inverse(blocks_red, sparse=True)[0]
    T_dense = T_sparse.toarray()

    # 3. Verify T is an intertwiner: commutes with group generators
    P1 = get_monomial_permutation_matrix(g1, space)
    P2 = get_monomial_permutation_matrix(g2, space)
    assert np.allclose(P1 @ T_dense @ P1.T, T_dense, atol=1e-10)
    assert np.allclose(P2 @ T_dense @ P2.T, T_dense, atol=1e-10)
    assert np.allclose(T_dense, T_dense_init, atol=1e-10)

    # 4. Apply forward(reduced=True) without reynolds to T_sparse and check exact recovery of blocks
    blocks_red_2 = sab.apply_forward([T_sparse], reynolds=False, reduced=True)
    assert len(blocks_red) == len(blocks_red_2)
    for b_list1, b_list2 in zip(blocks_red, blocks_red_2):
        b1, b2 = b_list1[0], b_list2[0]
        if b1 is None:
            assert b2 is None
        else:
            np.testing.assert_allclose(
                b1.toarray() if scipy.sparse.issparse(b1) else b1,
                b2.toarray() if scipy.sparse.issparse(b2) else b2,
                atol=1e-10,
            )

    # 5. Inverse again should give back T exactly
    T_sparse_2 = sab.apply_inverse(blocks_red_2, sparse=True)[0]
    np.testing.assert_allclose(T_dense, T_sparse_2.toarray(), atol=1e-10)


def test_inverse_sab_roundtrip_full():
    """Verify that forward(reduced=False) -> inverse gives back the matrix EXACTLY."""
    n = 3
    space = MonomialSpace(n=n, d=1)
    group = cyclic(n)
    g1 = Permutation((1, 2, 0))

    sab = build_sab(space, group, [g1])
    N = space.total_monomials()

    rng = np.random.default_rng(123)
    M_sparse = scipy.sparse.csr_matrix(rng.standard_normal((N, N)))

    # Generate an intertwiner by applying forward with reynolds and then inverse
    blocks_init = sab.apply_forward([M_sparse], reynolds=True, reduced=True)
    T_sparse = sab.apply_inverse(blocks_init, sparse=True)[0]

    # Now test full forward and inverse
    blocks_full = sab.apply_forward([T_sparse], reynolds=False, reduced=False)

    # In full mode, number of blocks equals total number of irrep copies in decomposition
    assert len(blocks_full) >= len(blocks_init)

    # Reconstruct from full blocks
    T_rec_sparse = sab.apply_inverse(blocks_full, sparse=True)[0]
    np.testing.assert_allclose(T_sparse.toarray(), T_rec_sparse.toarray(), atol=1e-10)


def test_dense_and_sparse_parity():
    """Verify that sparse=False and sparse=True produce identical numerical results."""
    n = 3
    space = MonomialSpace(n=n, d=1)
    group = cyclic(n)
    g1 = Permutation((1, 2, 0))

    sab = build_sab(space, group, [g1])
    N = space.total_monomials()

    rng = np.random.default_rng(999)
    M_dense = rng.standard_normal((N, N))
    M_sparse = scipy.sparse.csr_matrix(M_dense)

    # Forward reduced
    blocks_sparse = sab.apply_forward(
        [M_sparse], reynolds=True, reduced=True, sparse=True
    )
    blocks_dense = sab.apply_forward(
        [M_dense], reynolds=True, reduced=True, sparse=False
    )

    assert len(blocks_sparse) == len(blocks_dense)
    for b_s_list, b_d_list in zip(blocks_sparse, blocks_dense):
        b_s, b_d = b_s_list[0], b_d_list[0]
        if b_s is None:
            assert b_d is None
        else:
            assert scipy.sparse.issparse(b_s)
            assert isinstance(b_d, np.ndarray)
            np.testing.assert_allclose(b_s.toarray(), b_d, atol=1e-10)

    # Inverse sparse vs dense
    T_rec_sparse = sab.apply_inverse(blocks_sparse, sparse=True)[0]
    T_rec_dense = sab.apply_inverse(blocks_dense, sparse=False)[0]

    assert isinstance(T_rec_dense, np.ndarray)
    np.testing.assert_allclose(T_rec_sparse.toarray(), T_rec_dense, atol=1e-10)


def test_realize_roundtrip():
    """Verify forward and inverse with real realization (realize=True)."""
    n = 3
    space = MonomialSpace(n=n, d=2)
    group = dihedral(n)
    g1 = Permutation((1, 2, 0))
    g2 = Permutation((0, 2, 1))
    id_perm = Permutation((0, 1, 2))
    coset_reps = {4: [id_perm], 6: [id_perm, g2]}

    sab = build_sab(space, group, [g1, g2], coset_reps=coset_reps)
    N = space.total_monomials()

    rng = np.random.default_rng(777)
    M_dense = rng.standard_normal((N, N))
    M_dense = 0.5 * (M_dense + M_dense.T)  # Real symmetric
    T_dense_init = project_to_intertwiner(M_dense, [g1, g2], space)

    # Forward with realize=True
    blocks_real = sab.apply_forward(
        [T_dense_init], reynolds=False, realize=True, reduced=True, sparse=False
    )

    # Verify blocks are real numpy arrays
    for b_list in blocks_real:
        b = b_list[0]
        if b is not None:
            assert np.isrealobj(b)

    # Inverse with realize=True
    T_rec = sab.apply_inverse(blocks_real, realize=True, sparse=False)[0]
    assert np.isrealobj(T_rec)

    # Check intertwiner invariance and exact reconstruction
    P1 = get_monomial_permutation_matrix(g1, space)
    assert np.allclose(P1 @ T_rec @ P1.T, T_rec, atol=1e-10)
    assert np.allclose(T_rec, T_dense_init, atol=1e-10)

    # Re-apply forward to check roundtrip exactness
    blocks_real_2 = sab.apply_forward(
        [T_rec], reynolds=False, realize=True, reduced=True, sparse=False
    )
    for b_list1, b_list2 in zip(blocks_real, blocks_real_2):
        b1, b2 = b_list1[0], b_list2[0]
        if b1 is None:
            assert b2 is None
        else:
            np.testing.assert_allclose(b1, b2, atol=1e-10)


def test_batch_processing():
    """Verify that batched input to apply_forward and apply_inverse works correctly."""
    n = 3
    space = MonomialSpace(n=n, d=1)
    group = cyclic(n)
    g1 = Permutation((1, 2, 0))
    sab = build_sab(space, group, [g1])
    N = space.total_monomials()

    rng = np.random.default_rng(42)
    batch_in = [rng.standard_normal((N, N)) for _ in range(3)]

    # Forward batch
    blocks_batch = sab.apply_forward(
        batch_in, reynolds=True, reduced=True, sparse=False
    )
    assert len(blocks_batch) > 0
    assert len(blocks_batch[0]) == 3

    # Inverse batch
    rec_batch = sab.apply_inverse(blocks_batch, sparse=False)
    assert len(rec_batch) == 3

    # Check individual match
    for idx in range(3):
        b_ind = sab.apply_forward(
            [batch_in[idx]], reynolds=True, reduced=True, sparse=False
        )
        rec_ind = sab.apply_inverse(b_ind, sparse=False)[0]
        np.testing.assert_allclose(rec_batch[idx], rec_ind, atol=1e-10)
