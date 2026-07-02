"""Metacyclic group constructions."""

from monsab.core import PcGroup
from monsab.util import is_prime, primitive_root


def metacyclic(p: int, q: int, r: int) -> PcGroup:
    """Constructs a metacyclic group defined by the parameters p, q, r."""
    if p <= 0 or q <= 0 or r <= 0:
        raise ValueError("Parameters p, q, and r must be positive integers.")

    return PcGroup(
        number_of_generators=2,
        orders=[p, q],
        conjugation_exponents={(0, 1): r},
        power_tails={0: [], 1: []},
        conjugation_tails={},
    )


def dihedral(n: int) -> PcGroup:
    """Constructs a dihedral group of order 2n."""
    if n <= 0:
        raise ValueError("Parameter n must be a positive integer.")

    return PcGroup(
        number_of_generators=2,
        orders=[n, 2],
        conjugation_exponents={(0, 1): n - 1},
        power_tails={0: [], 1: []},
        conjugation_tails={},
    )


def affine_group_1d(p: int) -> PcGroup:
    """Constructs the affine group AGL(1,p) of dimension 1 over the finite prime field of order p."""
    if not is_prime(p):
        raise ValueError("Parameter p must be a prime number.")

    g = primitive_root(p)
    g_inv = pow(g, -1, p)
    return PcGroup(
        number_of_generators=2,
        orders=[p, p - 1],
        conjugation_exponents={(0, 1): g_inv},
        power_tails={0: [], 1: []},
        conjugation_tails={},
    )
