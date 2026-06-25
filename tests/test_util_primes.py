import pytest
from monsab.util import is_prime, prime_factorization, primitive_root, extract_roots


def test_is_prime():
    assert is_prime(5)
    assert not is_prime(4)
    assert not is_prime(1)
    assert not is_prime(0)
    assert not is_prime(-5)


def test_prime_factorization():
    assert prime_factorization(12) == [2, 2, 3]


def test_primitive_root():
    assert primitive_root(5) in (2, 3)
    with pytest.raises(ValueError):
        primitive_root(4)
    assert primitive_root(2) == 1


def test_extract_roots():
    assert extract_roots(0, 2, 4) == (0, 2)
