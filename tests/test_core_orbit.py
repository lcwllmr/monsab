from monsab.core import PcGroup, OrbitLifter


def test_orbit_lifter_standard():
    # Z_3 action on 3 elements: generator (0 1 2)
    group = PcGroup(
        number_of_generators=1,
        orders=[3],
        conjugation_exponents={},
        power_tails={},
        conjugation_tails={},
        generators=[[1, 2, 0]],
    )

    lifter = OrbitLifter(group, 2, False)  # D=2, standard monomial

    # Monomials represented as indices
    m1 = [0, 1]
    m2 = [1, 2]
    m3 = [0, 2]

    c1 = lifter.canonicalize(m1)
    c2 = lifter.canonicalize(m2)
    c3 = lifter.canonicalize(m3)

    assert c1 == c2 == c3
    assert c1 == [0, 1]


def test_orbit_lifter_squarefree():
    # Z_2 x Z_2 on 4 elements
    group = PcGroup(
        number_of_generators=2,
        orders=[2, 2],
        conjugation_exponents={},
        power_tails={},
        conjugation_tails={},
        generators=[
            [1, 0, 3, 2],  # (0 1)(2 3)
            [2, 3, 0, 1],  # (0 2)(1 3)
        ],
    )

    lifter = OrbitLifter(group, 3, True)  # D=3, squarefree

    m1 = [0, 1, 2]
    m2 = [0, 1, 3]  # Under first gen: [1, 0, 2] which is [0, 1, 2]
    m3 = [0, 2, 3]  # Under second gen: [2, 3, 1] which is [1, 2, 3]
    m4 = [1, 2, 3]

    c1 = lifter.canonicalize(m1)
    c2 = lifter.canonicalize(m2)
    c3 = lifter.canonicalize(m3)
    c4 = lifter.canonicalize(m4)

    # All of these are in the same orbit, canonical rep should be the smallest sorted array
    assert c1 == c2 == c3 == c4
    assert c1 == [0, 1, 2]
