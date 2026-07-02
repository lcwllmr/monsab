from monsab.core import Permutation
from monsab.zoo import dihedral, metacyclic, affine_group_1d


def test_dihedral():
    n = 5
    grp = dihedral(n)
    r = Permutation(tuple((i + 1) % n for i in range(n)))
    s = Permutation(tuple((n - i) % n for i in range(n)))
    assert grp.test_generators([r, s])
    assert grp.test_consistency()


def test_metacyclic():
    grp = metacyclic(3, 2, 2)
    r = Permutation(tuple((i + 1) % 3 for i in range(3)))
    s = Permutation(tuple((3 - i) % 3 for i in range(3)))
    assert grp.test_generators([r, s])
    assert grp.test_consistency()


def test_affine_group_1d():
    p = 3

    grp = affine_group_1d(p)
    t = Permutation(tuple((i + 1) % p for i in range(p)))
    s = Permutation(tuple((2 * i) % p for i in range(p)))
    assert grp.test_generators([t, s])
    assert grp.test_consistency()


def test_metacyclic_edge_cases():
    import pytest
    from monsab.zoo import metacyclic, dihedral, affine_group_1d

    with pytest.raises(ValueError):
        metacyclic(-1, 2, 3)

    with pytest.raises(ValueError):
        dihedral(0)

    with pytest.raises(ValueError):
        affine_group_1d(4)
