from monsab.core import PcGroup


def test_polycyclic_presentation():
    # Simple check for prime factors and normal series
    p = PcGroup(
        number_of_generators=2,
        orders=(2, 3),
        conjugation_exponents={(0, 1): 1},
        power_tails={0: (), 1: ()},
        conjugation_tails={},
    )
    assert p.test_consistency()
    assert p.is_normal_series
    assert p.is_abelian
    assert p.is_nilpotent
    assert p.is_supersolvable

    non_prime = PcGroup(
        number_of_generators=1,
        orders=(4,),
        conjugation_exponents={},
        power_tails={0: ()},
        conjugation_tails={},
    )
    assert not non_prime.test_consistency()


def test_polycyclic_edge_cases():
    from monsab.core import evaluate_word, Permutation

    gens = (Permutation((1, 0)),)
    res = evaluate_word(((0, 1),), gens)
    assert res == gens[0]

    desc = PcGroup(1, (2,), {}, {0: ((0, 1),)}, {})
    assert desc.order == 2

    assert not desc.test_generators(())
    assert desc.test_generators((Permutation((0,)),))
    assert not desc.test_generators((Permutation((1, 0)),))

    bad_conj = PcGroup(2, (2, 2), {(0, 1): 0}, {}, {(0, 1): ()})
    assert not bad_conj.test_generators((Permutation((1, 0)), Permutation((1, 0))))

    bad_normal = PcGroup(2, (2, 2), {}, {}, {(1, 0): ((0, 1),)})
    assert not bad_normal.is_normal_series

    bad_power = PcGroup(1, (2,), {}, {0: ((0, 1),)}, {})
    assert not bad_power.is_normal_series
