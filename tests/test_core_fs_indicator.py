from monsab.core import (
    BaumClausenStage,
    BaumClausenPaths,
    Permutation,
    PcGroup,
)
from monsab.pop import MonomialSpace, build_monomial_sab


def test_d8_q8_fs_indicators():
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

    presentation = PcGroup(
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
    paths = BaumClausenPaths.from_baum_clausen(tuple(stages))
    concrete_generators = {1: g0, 2: g1, 3: g2, 4: g3, 5: g4}
    space = MonomialSpace(32)
    orbits_space = [tuple(range(1, 33))]

    transform = build_monomial_sab(paths, concrete_generators, orbits_space, space, 1)

    fs = {}
    for b in transform.blocks:
        rep = next(r for r in reps if r.id == b.rep_id)
        if rep.id not in fs:
            fs[rep.id] = (rep.dim, getattr(b, "fs_indicator", None))

    counts = {
        (1, 1): 0,
        (1, -1): 0,
        (1, 0): 0,
        (2, 1): 0,
        (2, -1): 0,
        (2, 0): 0,
        (4, 1): 0,
        (4, -1): 0,
        (4, 0): 0,
    }

    for dim, ind in fs.values():
        counts[(dim, ind)] += 1

    # Central product of D8 and Q8 has 16 1D irreps and one 4D irrep.
    # The 4D irrep is the tensor product of the 2D irreps of D8 (real) and Q8 (quaternionic),
    # so its FS indicator is 1 * -1 = -1.
    assert counts[(1, 1)] == 16
    assert counts[(4, -1)] == 1


def test_cyclic_group_complex_fs():
    conjugation_exponents = {}
    conjugation_tails = {}
    presentation = PcGroup(
        number_of_generators=1,
        orders=(3,),
        conjugation_exponents=conjugation_exponents,
        power_tails={0: ()},
        conjugation_tails=conjugation_tails,
    )
    # G = C3
    g = Permutation((1, 2, 0))
    G_gens = {1: g}

    stages = [BaumClausenStage.trivial(e=3, presentation=presentation)]
    for k, order in enumerate(presentation.orders, start=1):
        stages.append(BaumClausenStage.next_level(stages[-1], g_i=k, p=order))

    paths = BaumClausenPaths.from_baum_clausen(tuple(stages))
    space = MonomialSpace(3)
    orbits_space = [tuple(range(1, 4))]
    transform = build_monomial_sab(paths, G_gens, orbits_space, space, 1)

    fs = {}
    for b in transform.blocks:
        fs[b.rep_id] = getattr(b, "fs_indicator", None)

    # C3 has one real irrep (trivial, fs=1) and two complex conjugate irreps (fs=0)
    fs_values = list(fs.values())
    assert fs_values.count(1) == 1
    assert fs_values.count(0) == 2
