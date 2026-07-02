"""Abelian group constructions."""

from monsab.core import PcGroup

from ._base import trivial
from ._product import direct_product


def cyclic(order: int) -> PcGroup:
    """Constructs a cyclic group of a given order."""
    if order < 1:
        raise ValueError("Order must be a positive integer.")
    if order == 1:
        return trivial()
    grp = PcGroup(
        number_of_generators=1,
        orders=[order],
        conjugation_exponents={},
        power_tails={0: []},
        conjugation_tails={},
    )
    assert grp.test_consistency()
    return grp


def abelian(*orders: int) -> PcGroup:
    """Constructs a finite abelian group as a direct product of cyclic groups."""
    if not orders:
        return trivial()
    for order in orders:
        if order < 1:
            raise ValueError("All orders must be positive integers.")

    return direct_product(*[cyclic(order) for order in orders])
