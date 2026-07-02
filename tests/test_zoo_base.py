from monsab.zoo._base import trivial


def test_trivial():
    grp = trivial()
    assert grp.number_of_generators == 0
    assert grp.test_consistency()
    assert grp.is_abelian
    assert grp.is_nilpotent
    assert grp.test_supersolvable()
