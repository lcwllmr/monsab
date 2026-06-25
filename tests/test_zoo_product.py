from monsab.zoo import direct_product, cyclic


def test_direct_product():
    grp = direct_product(cyclic(2), cyclic(3))
    assert grp.description.number_of_generators == 2
    assert grp.description.orders == (2, 3)


def test_direct_product_edge_cases():
    from monsab.zoo import direct_product, dihedral

    assert direct_product().description.number_of_generators == 0

    grp = direct_product(dihedral(3), dihedral(4))
    assert grp.description.number_of_generators == 4
    assert grp.description.orders == (3, 2, 4, 2)
    assert len(grp.description.conjugation_exponents) == 2
