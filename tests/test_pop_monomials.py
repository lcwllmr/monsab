from monsab.pop import MonomialSpace
from monsab.core import Permutation


def test_monomial_space():
    space = MonomialSpace(2)
    assert space.total_monomials(2) == 6
    g = {1: Permutation((1, 0))}
    orbits = space.get_orbits(g, 2)
    assert len(orbits) == 4
    full_orbits = space.get_full_orbits(g, 2)
    assert len(full_orbits) == 4


def test_monomial_space_degree_4():
    from monsab.pop import MonomialSpace
    from monsab.core import Permutation

    space = MonomialSpace(3)
    g = {1: Permutation((1, 2, 0)), 2: Permutation((1, 0, 2))}

    orbits = space.get_orbits(g, 4)
    full_orbits = space.get_full_orbits(g, 4, num_threads=2)
    assert len(orbits) == len(full_orbits)

    # Also test with num_threads=1 to get coverage of _compute_orbit_chunk
    full_orbits_seq = space.get_full_orbits(g, 4, num_threads=1)
    assert len(full_orbits_seq) == len(full_orbits)

    # Trigger unrank/rank on large degree
    tup = space.unrank_tuple(space.total_monomials(4) - 1)
    assert space.rank_tuple(tup) == space.total_monomials(4) - 1

    from monsab.core import SABTransform, BaumClausenPaths
    from monsab.pop._monomials import build_monomial_sab

    # Provide a dummy abstract BaumClausenPaths
    abstract = BaumClausenPaths({0: ()}, 2, {0: 0})
    sab = build_monomial_sab(abstract, g, full_orbits[:1], space, 4, num_threads=2)
    assert isinstance(sab, SABTransform)

    # Also test sequential for coverage
    sab_seq = build_monomial_sab(abstract, g, full_orbits[:1], space, 4, num_threads=1)
    assert isinstance(sab_seq, SABTransform)


def test_squarefree_monomial_space():
    from monsab.pop import SquarefreeMonomialSpace
    from monsab.core import Permutation

    space = SquarefreeMonomialSpace(4)
    g = {1: Permutation((1, 2, 0, 3)), 2: Permutation((1, 0, 2, 3))}
    # k=0: 1))}

    orbits = space.get_orbits(g, 3)
    full_orbits = space.get_full_orbits(g, 3, num_threads=2)
    assert len(orbits) == len(full_orbits)

    full_orbits_seq = space.get_full_orbits(g, 3, num_threads=1)
    assert len(full_orbits_seq) == len(full_orbits)

    # Test rank/unrank for all elements
    for i in range(space.total_monomials(3)):
        tup = space.unrank_tuple(i)
        assert space.rank_tuple(tup) == i

        # verify apply_gen matches generic tup sorting
        inv_data = (~g[1]).data
        nxt_i = space.apply_gen(i, inv_data)
        nxt_tup = tuple(sorted(inv_data[v] for v in tup))
        assert space.rank_tuple(nxt_tup) == nxt_i

    from monsab.core import SABTransform, BaumClausenPaths
    from monsab.pop._monomials import build_monomial_sab

    # Provide a dummy abstract BaumClausenPaths
    abstract = BaumClausenPaths({0: ()}, 2, {0: 0})
    sab = build_monomial_sab(abstract, g, full_orbits[:1], space, 3, num_threads=2)
    assert isinstance(sab, SABTransform)

    # Also test sequential for coverage
    sab_seq = build_monomial_sab(abstract, g, full_orbits[:1], space, 3, num_threads=1)
    assert isinstance(sab_seq, SABTransform)


def test_sparse_sab_block_structure():
    """
    Tests that SABBlock holds sparse mappings (valid_cols, j_values, l_values)
    and that parallel Rust threads behave identically to sequential ones.
    """
    from monsab.pop import SquarefreeMonomialSpace
    from monsab.core import Permutation
    from monsab.core import BaumClausenPaths
    from monsab.pop._monomials import build_monomial_sab
    import numpy as np

    space = SquarefreeMonomialSpace(4)
    g = {1: Permutation((1, 2, 0, 3)), 2: Permutation((1, 0, 2, 3))}
    orbits = space.get_full_orbits(g, 2, num_threads=1)
    abstract = BaumClausenPaths({0: ()}, 4, {0: 0})

    sab_seq = build_monomial_sab(abstract, g, orbits, space, 2, num_threads=1)
    sab_par = build_monomial_sab(abstract, g, orbits, space, 2, num_threads=4)

    # Verify structural equivalence between seq and par
    assert len(sab_seq.blocks) == len(sab_par.blocks)

    for b_seq, b_par in zip(sab_seq.blocks, sab_par.blocks):
        assert b_seq.rep_id == b_par.rep_id
        assert b_seq.dim == b_par.dim
        np.testing.assert_array_equal(b_seq.valid_cols, b_par.valid_cols)
        np.testing.assert_array_equal(b_seq.j_values, b_par.j_values)
        np.testing.assert_array_equal(b_seq.l_values, b_par.l_values)

    # Check explicit basis to ensure sparse values are correct
    basis_seq = sab_seq.explicit_basis(sparse=False)
    basis_par = sab_par.explicit_basis(sparse=False)
    assert len(basis_seq) == len(basis_par)
    for m1, m2 in zip(basis_seq, basis_par):
        np.testing.assert_allclose(m1, m2)
