import numpy as np
import scipy.sparse
from monsab.core import (
    Permutation,
    PcGroup,
    BaumClausenStage,
    BaumClausenPaths,
)
from monsab.pop import MonomialSpace, build_monomial_sab


def test_reynolds_extraction_equivalence():
    """
    Tests that evaluating the SAB Transform on a fully G-averaged matrix R(X)
    yields the exact same blocks as pre-averaging X over the subgroup H_k
    and applying the Method 1 (Coset Averaging) evaluation via `reynolds=True`.
    """
    g1 = Permutation((1, 2, 0, 5, 3, 4))
    g2 = Permutation((3, 4, 5, 0, 1, 2))
    concrete_generators = {1: g1, 2: g2}

    presentation = PcGroup(2, (3, 2), {(0, 1): 2}, {}, {})

    stages = [BaumClausenStage.trivial(6, presentation)]
    stages.append(BaumClausenStage.next_level(stages[-1], 1, 3))
    stages.append(BaumClausenStage.next_level(stages[-1], 2, 2))

    paths = BaumClausenPaths.from_baum_clausen(tuple(stages))

    space = MonomialSpace(6)
    orbits = [tuple(range(7))]

    s1 = Permutation((0, 1, 2, 3, 4, 5))
    s2 = g2
    all_g = [
        s1,
        g1,
        Permutation(tuple(g1.data[g1.data[i]] for i in g1.data)),  # g1^2
        s2,
        Permutation(tuple(s2.data[g1.data[i]] for i in g1.data)),  # s2 g1
        Permutation(tuple(s2.data[g1.data[g1.data[i]]] for i in g1.data)),  # s2 g1^2
    ]
    coset_reps = {rep_id: all_g for rep_id in paths.paths.keys()}

    build_monomial_sab(
        paths, concrete_generators, orbits, space, 1, coset_reps=coset_reps
    )
    transform_no_reps = build_monomial_sab(paths, concrete_generators, orbits, space, 1)

    X1_dense = np.zeros((7, 7))
    X1_dense[1, 2] = 1.0

    # Compute the full group average R(X)
    X1_avg = np.zeros((7, 7))
    for g in all_g:
        g_inv = ~g
        for i in range(7):
            for j in range(7):
                if X1_dense[i, j] != 0:
                    X1_avg[
                        space.apply_gen(i, g_inv.data), space.apply_gen(j, g_inv.data)
                    ] += X1_dense[i, j] / 6.0

    X1_avg_sparse = scipy.sparse.csr_matrix(X1_avg)

    # Note: Method 1 on X1_hk will evaluate Tk(R(X)) exactly!
    X1_hk = np.zeros((7, 7))

    # Average X over H_k (here H_k for standard representation is C_3)
    # Note: H_k is trivial for the standard representation because the splitting subgroup is trivial,
    # but the representation is derived from C_3. Wait, H_k is just the subgroup assigned at the leaf node.
    # At level 2 in the test, H_k = {id}, but we provided all 6 coset reps, so we effectively pre-averaged over {id}
    # For H_k = C_3, the reps would be [s1, s2] and we would pre-average over C_3.
    # Here we simulate pre-averaging over C_3 and applying coset reps for C_3.

    # Re-build for H_k = C_3
    coset_reps_c3 = {rep_id: [s1, s2] for rep_id in paths.paths.keys()}
    transform_c3 = build_monomial_sab(
        paths, concrete_generators, orbits, space, 1, coset_reps=coset_reps_c3
    )

    for h in [s1, g1, Permutation(tuple(g1.data[g1.data[i]] for i in g1.data))]:
        h_inv = ~h
        for i in range(7):
            for j in range(7):
                if X1_dense[i, j] != 0:
                    X1_hk[
                        space.apply_gen(i, h_inv.data), space.apply_gen(j, h_inv.data)
                    ] += X1_dense[i, j] / 3.0
    X1_hk_sparse = scipy.sparse.csr_matrix(X1_hk)

    fast_blocks = transform_no_reps.apply_forward([X1_avg_sparse])
    method1_blocks = transform_c3.apply_forward([X1_hk_sparse], reynolds=True)

    for k in range(len(fast_blocks)):
        if fast_blocks[k]:
            np.testing.assert_allclose(
                fast_blocks[k][0].toarray(), method1_blocks[k][0].toarray(), atol=1e-10
            )
