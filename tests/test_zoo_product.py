from monsab.zoo import direct_product, cyclic


def test_direct_product():
    grp = direct_product(cyclic(2), cyclic(3))
    assert grp.number_of_generators == 2
    assert list(grp.orders) == [2, 3]
    assert grp.is_abelian
    assert grp.is_nilpotent
    assert grp.is_supersolvable


def test_direct_product_edge_cases():
    from monsab.zoo import direct_product, dihedral

    assert direct_product().number_of_generators == 0

    grp = direct_product(dihedral(3), dihedral(5))
    assert grp.number_of_generators == 4
    assert list(grp.orders) == [3, 2, 5, 2]
    assert len(grp.conjugation_exponents) == 2
    assert not grp.is_abelian
    assert not grp.is_nilpotent
    assert grp.is_supersolvable
