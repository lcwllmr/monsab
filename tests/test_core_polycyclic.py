from monsab.core import PolycyclicPresentation


def test_polycyclic_presentation():
    # Simple check for prime factors and normal series
    p = PolycyclicPresentation(
        number_of_generators=2,
        orders=(2, 3),
        conjugation_exponents={(0, 1): 2},
        power_tails={0: (), 1: ()},
        conjugation_tails={},
    )
    assert p.has_only_prime_factors
    assert p.is_normal_series


def test_polycyclic_edge_cases():
    from monsab.core._polycyclic import _evaluate_word
    from monsab.core import Permutation

    gens = (Permutation((1, 0)),)
    res = _evaluate_word(((0, 1),), gens)
    assert res == gens[0]

    desc = PolycyclicPresentation(1, (2,), {}, {0: ((0, 1),)}, {})
    assert desc.order == 2

    assert not desc.verify(())
    assert desc.verify((Permutation((0,)),))
    assert not desc.verify((Permutation((1, 0)),))

    bad_conj = PolycyclicPresentation(2, (2, 2), {(0, 1): 0}, {}, {(0, 1): ()})
    assert not bad_conj.verify((Permutation((1, 0)), Permutation((1, 0))))

    bad_normal = PolycyclicPresentation(2, (2, 2), {}, {}, {(1, 0): ((0, 1),)})
    assert not bad_normal.is_normal_series

    bad_power = PolycyclicPresentation(1, (2,), {}, {0: ((0, 1),)}, {})
    assert not bad_power.is_normal_series
