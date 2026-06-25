"""
Metacyclic group constructions.
"""

from monsab.core import PolycyclicPresentation
from monsab.util import is_prime, primitive_root

from ._base import MonomialGroup


def metacyclic(p: int, q: int, r: int) -> MonomialGroup:
    """Constructs a metacyclic group defined by the parameters p, q, r."""
    if p <= 0 or q <= 0 or r <= 0:
        raise ValueError("Parameters p, q, and r must be positive integers.")

    # The metacyclic group can be represented by two generators with specific relations
    description = PolycyclicPresentation(
        number_of_generators=2,
        orders=(p, q),
        conjugation_exponents={(0, 1): r},  # Conjugation relation: g1^-1 g0 g1 = g0^r
        power_tails={0: (), 1: ()},  # No additional power tails
        conjugation_tails={},  # No additional conjugation tails
    )

    return MonomialGroup(
        description=description,
        is_supersolvable=True,  # Metacyclic groups are supersolvable
    )


def dihedral(n: int) -> MonomialGroup:
    """Constructs a dihedral group of order 2n."""
    if n <= 0:
        raise ValueError("Parameter n must be a positive integer.")

    # The dihedral group can be represented by two generators with specific relations
    description = PolycyclicPresentation(
        number_of_generators=2,
        orders=(n, 2),
        conjugation_exponents={
            (0, 1): n - 1
        },  # Conjugation relation: g1^-1 g0 g1 = g0^(n-1)
        power_tails={0: (), 1: ()},  # No additional power tails
        conjugation_tails={},  # No additional conjugation tails
    )

    return MonomialGroup(
        description=description,
        is_abelian=False,  # Dihedral groups are generally non-abelian
        is_nilpotent=False,  # Dihedral groups are not necessarily nilpotent
        is_supersolvable=True,  # Dihedral groups are supersolvable
    )


def affine_group_1d(p: int) -> MonomialGroup:
    """Constructs the affine group AGL(1,p) of dimension 1 over the finite prime field of order p."""
    if not is_prime(p):
        raise ValueError("Parameter p must be a prime number.")

    # g0 must be the normal subgroup (translations, order p)
    # g1 must be the quotient (scaling, order p-1)
    g = primitive_root(p)
    g_inv = pow(g, -1, p)
    description = PolycyclicPresentation(
        number_of_generators=2,
        orders=(p, p - 1),
        conjugation_exponents={
            (0, 1): g_inv
        },  # Conjugation relation: g1^-1 g0 g1 = g0^(g^-1)
        power_tails={0: (), 1: ()},
        conjugation_tails={},
    )

    return MonomialGroup(
        description=description,
        is_abelian=False,  # Affine groups are generally non-abelian
        is_nilpotent=False,  # Affine groups are not necessarily nilpotent
        is_supersolvable=True,  # Affine groups are supersolvable
    )
