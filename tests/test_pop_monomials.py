from monsab.pop import MonomialSpace
from monsab.core import Permutation


def test_monomial_space():
    space = MonomialSpace(2, 2)
    assert space.total_monomials == 6
    g = {1: Permutation((1, 0))}
    orbits = space.get_orbits(g)
    assert len(orbits) == 4
    full_orbits = space.get_full_orbits(g)
    assert len(full_orbits) == 4


def test_monomial_space_degree_4():
    from monsab.pop import MonomialSpace
    from monsab.core import Permutation

    space = MonomialSpace(3, 4)
    g = {1: Permutation((1, 2, 0)), 2: Permutation((1, 0, 2))}

    orbits = space.get_orbits(g)
    full_orbits = space.get_full_orbits(g, num_threads=2)
    assert len(orbits) == len(full_orbits)

    # Also test with num_threads=1 to get coverage of _compute_orbit_chunk
    full_orbits_seq = space.get_full_orbits(g, num_threads=1)
    assert len(full_orbits_seq) == len(full_orbits)

    # Trigger unrank/rank on large degree
    tup = space.unrank_tuple(space.total_monomials - 1)
    assert space.rank_tuple(tup) == space.total_monomials - 1

    from monsab.core._transform import SABTransform
    from monsab.core import BaumClausenPaths
    from monsab.pop._monomials import build_monomial_sab

    # Provide a dummy abstract BaumClausenPaths
    abstract = BaumClausenPaths({0: ()}, 2)
    sab = build_monomial_sab(abstract, g, full_orbits[:1], space, num_threads=2)
    assert isinstance(sab, SABTransform)

    # Also test sequential for coverage
    sab_seq = build_monomial_sab(abstract, g, full_orbits[:1], space, num_threads=1)
    assert isinstance(sab_seq, SABTransform)


def test_squarefree_monomial_space():
    from monsab.pop import SquarefreeMonomialSpace
    from monsab.core import Permutation

    space = SquarefreeMonomialSpace(4, 3)
    # k=0: 1, k=1: 4, k=2: 6, k=3: 4 => total 15
    assert space.total_monomials == 15
    g = {1: Permutation((1, 2, 3, 0))}

    orbits = space.get_orbits(g)
    full_orbits = space.get_full_orbits(g, num_threads=2)
    assert len(orbits) == len(full_orbits)

    full_orbits_seq = space.get_full_orbits(g, num_threads=1)
    assert len(full_orbits_seq) == len(full_orbits)

    # Test rank/unrank for all elements
    for i in range(space.total_monomials):
        tup = space.unrank_tuple(i)
        assert space.rank_tuple(tup) == i

        # verify apply_gen matches generic tup sorting
        inv_data = (~g[1]).data
        nxt_i = space.apply_gen(i, inv_data)
        nxt_tup = tuple(sorted(inv_data[v] for v in tup))
        assert space.rank_tuple(nxt_tup) == nxt_i
