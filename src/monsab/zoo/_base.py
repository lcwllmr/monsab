"""Base classes and trivial group construction for the group zoo."""

from monsab.core import PcGroup


def trivial() -> PcGroup:
    """Constructs the trivial group."""
    return PcGroup(
        number_of_generators=0,
        orders=[],
        conjugation_exponents={},
        power_tails={},
        conjugation_tails={},
    )
