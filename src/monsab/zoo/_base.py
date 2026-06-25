"""
Base classes for the group zoo.
"""

from dataclasses import dataclass

from monsab.core import PolycyclicPresentation


@dataclass(frozen=True, slots=True)
class MonomialGroup:
    """Represents a monomial group with a polycyclic presentation and optional properties."""

    description: PolycyclicPresentation
    is_abelian: bool | None = None
    is_nilpotent: bool | None = None
    is_supersolvable: bool | None = None


def trivial() -> MonomialGroup:
    """Constructs the trivial group."""
    return MonomialGroup(
        description=PolycyclicPresentation(
            number_of_generators=0,
            orders=(),
            conjugation_exponents={},
            power_tails={},
            conjugation_tails={},
        ),
        is_abelian=True,
        is_nilpotent=True,
        is_supersolvable=True,
    )
