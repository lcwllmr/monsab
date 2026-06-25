from monsab.core import MonomialMatrix


def test_identity_matrix_constructor() -> None:
    identity = MonomialMatrix.identity(dim=4, e=7)
    assert identity.perm == (0, 1, 2, 3)
    assert identity.vals == (0, 0, 0, 0)
    assert identity.e == 7


def test_multiply_composes_permutation_and_adds_values_mod_e() -> None:
    a = MonomialMatrix(perm=(1, 0, 2), vals=(3, 4, 5), e=7)
    b = MonomialMatrix(perm=(2, 1, 0), vals=(6, 1, 2), e=7)
    product = a @ b
    assert product.perm == (1, 2, 0)
    assert product.vals == (4, 3, 0)
    assert product.e == 7


def test_inverse_is_two_sided_inverse() -> None:
    matrix = MonomialMatrix(perm=(2, 0, 1, 3), vals=(9, 4, 12, 5), e=7)
    inv = matrix.inverse()
    identity = MonomialMatrix.identity(dim=4, e=7)
    assert matrix @ inv == identity
    assert inv @ matrix == identity


def test_power_matches_repeated_multiplication() -> None:
    matrix = MonomialMatrix(perm=(1, 2, 0), vals=(1, 4, 6), e=9)
    expected = MonomialMatrix.identity(dim=3, e=9)
    for _ in range(5):
        expected = expected @ matrix
    assert matrix**5 == expected
    assert matrix**0 == MonomialMatrix.identity(dim=3, e=9)


def test_power_accepts_negative_exponents() -> None:
    matrix = MonomialMatrix(perm=(1, 0), vals=(2, 5), e=8)
    assert matrix ** (-1) == matrix.inverse()
    assert matrix ** (-3) == matrix.inverse() ** 3


def test_direct_sum_adds_permutations_and_values() -> None:
    a = MonomialMatrix(perm=(1, 0), vals=(3, 4), e=5)
    b = MonomialMatrix(perm=(0, 1), vals=(2, 1), e=5)
    direct_sum = a + b
    assert direct_sum.perm == (1, 0, 2, 3)
    assert direct_sum.vals == (3, 4, 2, 1)
    assert direct_sum.e == 5


def test_direct_sum_with_empty_matrix() -> None:
    a = MonomialMatrix(perm=(0, 1), vals=(1, 2), e=3)
    empty = MonomialMatrix.identity(dim=0, e=3)
    direct_sum = a + empty
    assert direct_sum.perm == (0, 1)
    assert direct_sum.vals == (1, 2)
    assert direct_sum.e == 3


def test_matrix_edge_cases() -> None:
    import pytest

    with pytest.raises(ValueError):
        MonomialMatrix(perm=(0,), vals=(0,), e=-1)
    with pytest.raises(ValueError):
        MonomialMatrix(perm=(0,), vals=(0, 1), e=2)
    with pytest.raises(ValueError):
        MonomialMatrix(perm=(1,), vals=(0,), e=2)
    with pytest.raises(ValueError):
        MonomialMatrix.identity(dim=-1, e=2)

    a = MonomialMatrix(perm=(0,), vals=(0,), e=2)
    b = MonomialMatrix(perm=(0,), vals=(0,), e=3)
    c = MonomialMatrix(perm=(0, 1), vals=(0, 0), e=2)

    with pytest.raises(ValueError):
        a + b
    with pytest.raises(ValueError):
        a @ b
    with pytest.raises(ValueError):
        a @ c
    with pytest.raises(ValueError):
        a**1.5
