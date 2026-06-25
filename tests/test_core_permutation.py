from monsab.core import Permutation


def test_multiplication_is_right_to_left_and_non_commutative() -> None:
    p1 = Permutation((1, 2, 0))
    p2 = Permutation((1, 0, 2))
    assert p1 * p2 == Permutation((2, 1, 0))
    assert p2 * p1 == Permutation((0, 2, 1))
    assert p1 * p2 != p2 * p1


def test_multiplication_pads_shorter_permutation_with_fixed_points() -> None:
    shorter = Permutation((1, 0))
    longer = Permutation((0, 2, 1, 3))
    assert shorter * longer == Permutation((1, 2, 0, 3))


def test_inversion_composes_to_identity() -> None:
    p = Permutation((2, 0, 1, 3))
    identity = Permutation.identity(4)
    assert p * ~p == identity
    assert ~p * p == identity


def test_power_matches_repeated_multiplication() -> None:
    p = Permutation((1, 2, 0))
    expected = Permutation.identity(3)
    for _ in range(5):
        expected = expected * p
    assert p**5 == expected
    assert p**0 == Permutation.identity(3)


def test_permutation_edge_cases() -> None:
    import pytest

    p = Permutation((1, 2, 0))
    assert p != "not a permutation"
    with pytest.raises(ValueError):
        p**1.5

    assert p**-2 == (~p) ** 2
