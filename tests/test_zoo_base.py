from monsab.zoo._base import trivial


def test_trivial():
    grp = trivial()
    assert grp.description.number_of_generators == 0
    assert grp.is_abelian
