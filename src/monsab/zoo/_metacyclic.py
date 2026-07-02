"""Metacyclic group constructions."""

from monsab.core import PcGroup
from monsab.util import is_prime, primitive_root


def metacyclic(p: int, q: int, r: int) -> PcGroup:
    """Constructs a metacyclic group defined by the parameters p, q, r."""
    if p <= 0 or q <= 0 or r <= 0:
        raise ValueError("Parameters p, q, and r must be positive integers.")

    grp = PcGroup(
        number_of_generators=2,
        orders=[p, q],
        conjugation_exponents={(0, 1): r},
        power_tails={0: [], 1: []},
        conjugation_tails={},
    )
    assert grp.test_consistency()
    return grp


def dihedral(n: int) -> PcGroup:
    """Constructs a dihedral group of order 2n."""
    if n <= 0:
        raise ValueError("Parameter n must be a positive integer.")

    if n & (n - 1) == 0:
        k = n.bit_length() - 1
        orders = [2] * (k + 1)
        power_tails = {i: [(i + 1, 1)] if i < k - 1 else [] for i in range(k + 1)}
        conjugation_exponents = {}
        conjugation_tails = {}
        for j in range(k):
            for i in range(j + 1, k):
                conjugation_exponents[(j, i)] = 1
                conjugation_tails[(j, i)] = []
        for j in range(k):
            conjugation_exponents[(j, k)] = 1
            conjugation_tails[(j, k)] = [(m, 1) for m in range(j + 1, k)]
        grp = PcGroup(
            number_of_generators=k + 1,
            orders=orders,
            conjugation_exponents=conjugation_exponents,
            power_tails=power_tails,
            conjugation_tails=conjugation_tails,
        )
        assert grp.test_consistency()
        return grp

    grp = PcGroup(
        number_of_generators=2,
        orders=[n, 2],
        conjugation_exponents={(0, 1): n - 1},
        power_tails={0: [], 1: []},
        conjugation_tails={},
    )
    assert grp.test_consistency()
    return grp


def affine_group_1d(p: int) -> PcGroup:
    """Constructs the affine group AGL(1,p) of dimension 1 over the finite prime field of order p.

    Requires p to be a safe prime, i.e., both p and p-1 must be prime. This ensures the
    2-generator polycyclic presentation has prime relative orders throughout. The smallest
    valid values are p=3 (giving AGL(1,3) ≅ S_3) and p=7 (giving AGL(1,7) of order 42).
    """
    if not is_prime(p):
        raise ValueError("Parameter p must be a prime number.")
    if not is_prime(p - 1):
        raise ValueError(
            f"Parameter p must be a safe prime (both p and p-1 prime), but p-1={p - 1} is not prime."
        )

    g = primitive_root(p)
    g_inv = pow(g, -1, p)
    grp = PcGroup(
        number_of_generators=2,
        orders=[p, p - 1],
        conjugation_exponents={(0, 1): g_inv},
        power_tails={0: [], 1: []},
        conjugation_tails={},
    )
    assert grp.test_consistency()
    return grp
