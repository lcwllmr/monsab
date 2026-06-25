"""
Direct product constructions.
"""

from monsab.core import PolycyclicPresentation

from ._base import MonomialGroup, trivial


def direct_product(*factors: MonomialGroup) -> MonomialGroup:
    """Constructs the direct product of a sequence of monomial groups."""
    if not factors:
        return trivial()

    # Combine the descriptions of the factors into a new PolycyclicPresentation
    combined_number_of_generators = 0
    combined_orders = []
    combined_conjugation_exponents = {}
    combined_power_tails = {}
    combined_conjugation_tails = {}

    offset = 0
    for factor in factors:
        # Adjust indices for generators and orders
        combined_number_of_generators += factor.description.number_of_generators
        combined_orders.extend(factor.description.orders)

        # Adjust conjugation exponents and tails with offset
        for (i, j), exp in factor.description.conjugation_exponents.items():
            combined_conjugation_exponents[(i + offset, j + offset)] = exp

        for i, tail in factor.description.power_tails.items():
            combined_power_tails[i + offset] = tail

        for (i, j), tail in factor.description.conjugation_tails.items():
            combined_conjugation_tails[(i + offset, j + offset)] = tail

        offset += factor.description.number_of_generators

    combined_description = PolycyclicPresentation(
        combined_number_of_generators,
        orders=tuple(combined_orders),
        conjugation_exponents=combined_conjugation_exponents,
        power_tails=combined_power_tails,
        conjugation_tails=combined_conjugation_tails,
    )

    is_abelian = all(factor.is_abelian for factor in factors)
    is_nilpotent = all(factor.is_nilpotent for factor in factors)
    is_supersolvable = all(factor.is_supersolvable for factor in factors)

    return MonomialGroup(
        description=combined_description,
        is_abelian=is_abelian,
        is_nilpotent=is_nilpotent,
        is_supersolvable=is_supersolvable,
    )
