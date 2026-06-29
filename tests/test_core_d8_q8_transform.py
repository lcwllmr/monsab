import numpy as np

from monsab.core import (
    BaumClausenStage,
    BaumClausenPaths,
    Permutation,
    PolycyclicPresentation,
)
from monsab.pop import MonomialSpace, build_monomial_sab


def test_d8_q8_transform_quaternionic_block():
    # 1. Build D8 x Q8 regular representation on 64 elements
    def idx_D(i, j):
        return i + 4 * j

    def idx_Q(i, j):
        return i + 4 * j

    def idx_DQ(u, v):
        return u + 8 * v

    L_D_r = [0] * 8
    L_D_s = [0] * 8
    for i in range(4):
        for j in range(2):
            u = idx_D(i, j)
            L_D_r[u] = idx_D((i + 1) % 4, j)
            L_D_s[u] = idx_D((4 - i) % 4, (j + 1) % 2)

    L_Q_x = [0] * 8
    L_Q_y = [0] * 8
    for i in range(4):
        for j in range(2):
            v = idx_Q(i, j)
            L_Q_x[v] = idx_Q((i + 1) % 4, j)
            new_i = (4 - i) % 4 if j == 0 else (6 - i) % 4
            L_Q_y[v] = idx_Q(new_i, (j + 1) % 2)

    def apply_gen(u, v, D_perm, Q_perm):
        return D_perm[u] if D_perm else u, Q_perm[v] if Q_perm else v

    Z_perm = [0] * 64
    for u in range(8):
        for v in range(8):
            u2 = L_D_r[L_D_r[u]]
            v2 = L_Q_x[L_Q_x[v]]
            Z_perm[idx_DQ(u, v)] = idx_DQ(u2, v2)

    orbit_id = [-1] * 64
    orbits = []
    for w in range(64):
        if orbit_id[w] == -1:
            w2 = Z_perm[w]
            curr_id = len(orbits)
            orbit_id[w] = curr_id
            orbit_id[w2] = curr_id
            orbits.append((w, w2))

    assert len(orbits) == 32

    def project_perm(D_perm, Q_perm):
        p = [0] * 32
        for oid, (w, _) in enumerate(orbits):
            u = w % 8
            v = w // 8
            u_next, v_next = apply_gen(u, v, D_perm, Q_perm)
            w_next = idx_DQ(u_next, v_next)
            p[oid] = orbit_id[w_next]
        return Permutation(tuple(p))

    g0 = project_perm(L_D_r, None)
    g0 = g0 * g0  # z
    g1 = project_perm(None, L_Q_x)  # x
    g2 = project_perm(None, L_Q_y)  # y
    g3 = project_perm(L_D_r, None)  # r
    g4 = project_perm(L_D_s, None)  # s

    conjugation_exponents = {}
    conjugation_tails = {}
    for k in range(5):
        for j in range(k):
            conjugation_exponents[(j, k)] = 1
            conjugation_tails[(j, k)] = ()

    conjugation_tails[(3, 4)] = ((0, 1),)
    conjugation_tails[(1, 2)] = ((0, 1),)

    presentation = PolycyclicPresentation(
        number_of_generators=5,
        orders=(2, 2, 2, 2, 2),
        conjugation_exponents=conjugation_exponents,
        power_tails={0: (), 1: ((0, 1),), 2: ((0, 1),), 3: ((0, 1),), 4: ()},
        conjugation_tails=conjugation_tails,
    )

    stages = [BaumClausenStage.trivial(e=8, presentation=presentation)]
    for k, order in enumerate(presentation.orders, start=1):
        stages.append(BaumClausenStage.next_level(stages[-1], g_i=k, p=order))

    reps = stages[-1].representations

    R_D_r = [0] * 8
    R_D_s = [0] * 8
    for i in range(4):
        for j in range(2):
            u = idx_D(i, j)
            R_D_r[u] = idx_D((i + 1) % 4 if j == 0 else (i + 3) % 4, j)
            R_D_s[u] = idx_D(i, (j + 1) % 2)

    R_Q_x = [0] * 8
    R_Q_y = [0] * 8
    for i in range(4):
        for j in range(2):
            v = idx_Q(i, j)
            R_Q_x[v] = idx_Q((i + 1) % 4 if j == 0 else (i + 3) % 4, j)
            new_i = i if j == 0 else (i + 2) % 4
            R_Q_y[v] = idx_Q(new_i, (j + 1) % 2)

    def project_right_perm(D_perm, Q_perm):
        p = [0] * 32
        for oid, (w, _) in enumerate(orbits):
            u = w % 8
            v = w // 8
            u_next, v_next = apply_gen(u, v, D_perm, Q_perm)
            w_next = idx_DQ(u_next, v_next)
            p[oid] = orbit_id[w_next]
        return p

    def perm_to_matrix(p):
        M = np.zeros((32, 32))
        for i, val in enumerate(p):
            M[val, i] = 1.0
        return M

    rR_mat = perm_to_matrix(project_right_perm(R_D_r, None))
    rS_mat = perm_to_matrix(project_right_perm(R_D_s, None))
    rX_mat = perm_to_matrix(project_right_perm(None, R_Q_x))
    rY_mat = perm_to_matrix(project_right_perm(None, R_Q_y))

    # Commutant
    np.random.seed(42)
    mats = [np.eye(32), rR_mat, rS_mat, rX_mat, rY_mat]
    commutant_mats = []
    for m1 in mats:
        for m2 in mats:
            for m3 in mats:
                commutant_mats.append(m1 @ m2 @ m3)

    paths = BaumClausenPaths.from_baum_clausen(tuple(stages))
    concrete_generators = {1: g0, 2: g1, 3: g2, 4: g3, 5: g4}
    space = MonomialSpace(32, 1)
    orbits_space = [tuple(range(1, 33))]
    transform = build_monomial_sab(paths, concrete_generators, orbits_space, space)
    explicit_matrices = transform.explicit_basis(sparse=False)

    rand_coeffs = np.random.randn(len(commutant_mats))
    T_sub = sum(c * M for c, M in zip(rand_coeffs, commutant_mats))
    T_dense = np.zeros((33, 33))
    T_dense[1:, 1:] = T_sub

    for i, block_info in enumerate(transform.blocks):
        rep = next(r for r in reps if r.id == block_info.rep_id)
        if rep.dim == 4:
            U_k = explicit_matrices[i]
            block_matrix = U_k.conj().T @ T_dense @ U_k

            # Check quaternionic symmetry:
            # [[ A, B ]]
            # The structure is [[A, B], [-conj(B)[::-1, ::-1], conj(A)[::-1, ::-1]]]
            # This is equivalent to standard quaternion structure up to a basis permutation
            A = block_matrix[0:2, 0:2]
            B = block_matrix[0:2, 2:4]
            C = block_matrix[2:4, 0:2]
            D = block_matrix[2:4, 2:4]

            np.testing.assert_allclose(C, -np.conj(B[::-1, ::-1]), atol=1e-10)
            np.testing.assert_allclose(D, np.conj(A[::-1, ::-1]), atol=1e-10)
