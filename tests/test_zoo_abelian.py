from monsab.core import Permutation
from monsab.zoo import cyclic, abelian


def test_cyclic():
    n = 5
    grp = cyclic(n)
    gen0 = Permutation(tuple((i + 1) % n for i in range(n)))
    assert grp.test_generators([gen0])
    assert grp.test_consistency()
    assert grp.test_supersolvable()


def test_abelian():
    grp = abelian(2, 3)
    assert grp.test_consistency()
    assert grp.test_supersolvable()
    assert len(grp.orders) == 2


def test_abelian_edge_cases():
    import pytest
    from monsab.zoo import cyclic, abelian

    with pytest.raises(ValueError):
        cyclic(0)

    assert cyclic(1).number_of_generators == 0

    assert abelian().number_of_generators == 0

    with pytest.raises(ValueError):
        abelian(2, -1)
