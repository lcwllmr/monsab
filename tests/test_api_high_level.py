import pytest
import scipy.sparse

from monsab import (
    MonomialSpace,
    SquarefreeMonomialSpace,
    build_sab,
    LasserreHierarchy,
    Permutation,
    PcGroup,
    zoo,
)
from monsab.util import primitive_root


def test_top_level_exports():
    assert MonomialSpace is not None
    assert SquarefreeMonomialSpace is not None
    assert build_sab is not None
    assert LasserreHierarchy is not None
    assert Permutation is not None
    assert PcGroup is not None
    assert zoo is not None


def test_build_sab_with_sequences_and_mappings():
    n = 3  # AGL(1,3) ≅ S_3; p=3 is a safe prime (p-1=2 is prime)
    space = MonomialSpace(n=n, d=2)
    group = zoo.affine_group_1d(n)

    g = primitive_root(n)
    gen1_tup = tuple((i + 1) % n for i in range(n))
    gen2_tup = tuple((g * i) % n for i in range(n))

    gen1 = Permutation(gen1_tup)
    gen2 = Permutation(gen2_tup)

    # 1. Using a list of Permutation objects
    assert group.test_generators([gen1, gen2])
    t1 = build_sab(space, group, [gen1, gen2])

    # 2. Using a dictionary mapping
    t2 = build_sab(space, group, {1: gen1, 2: gen2})

    # 3. Using 0-indexed dictionary mapping (should auto-adjust)
    t3 = build_sab(space, group, {0: gen1, 1: gen2})

    # 4. Using a list of raw integer tuples
    t4 = build_sab(space, group, [gen1_tup, gen2_tup])

    N = space.total_monomials()
    mat = scipy.sparse.eye(N, format="csr")
    blocks1 = t1([mat])
    blocks2 = t2([mat])
    blocks3 = t3([mat])
    blocks4 = t4([mat])

    assert len(blocks1) == len(blocks2) == len(blocks3) == len(blocks4)
    assert len(blocks1) > 0


def test_optional_degree_d():
    # 1. MonomialSpace initialized with d
    space = MonomialSpace(n=4, d=2)
    assert space.total_monomials() == space.total_monomials(2)

    # 2. MonomialSpace without d initialized must raise if d not passed
    space_no_d = MonomialSpace(n=4)
    with pytest.raises(ValueError):
        space_no_d.total_monomials()
    assert space_no_d.total_monomials(2) == space.total_monomials()

    # 3. SquarefreeMonomialSpace defaults to n if d not specified
    sf_space = SquarefreeMonomialSpace(n=3)
    assert sf_space.total_monomials() == sf_space.total_monomials(3)


def test_pcgroup_representation():
    group = zoo.dihedral(5)
    rep = repr(group)
    assert "<" in rep and "|" in rep and ">" in rep
    assert "g0^5 = 1" in rep and "g1^2 = 1" in rep
    assert str(group) == rep
