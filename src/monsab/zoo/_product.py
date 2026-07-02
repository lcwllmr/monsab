"""Direct product constructions."""

from monsab.core import PcGroup

from ._base import trivial


def direct_product(*factors: PcGroup) -> PcGroup:
    """Constructs the direct product of a sequence of polycyclic groups."""
    if not factors:
        return trivial()

    combined_number_of_generators = 0
    combined_orders = []
    combined_conjugation_exponents = {}
    combined_power_tails = {}
    combined_conjugation_tails = {}

    offset = 0
    for factor in factors:
        combined_number_of_generators += factor.number_of_generators
        combined_orders.extend(factor.orders)

        for (i, j), exp in factor.conjugation_exponents.items():
            combined_conjugation_exponents[(i + offset, j + offset)] = exp

        for i, tail in factor.power_tails.items():
            combined_power_tails[i + offset] = [
                (idx + offset, exp) for idx, exp in tail
            ]

        for (i, j), tail in factor.conjugation_tails.items():
            combined_conjugation_tails[(i + offset, j + offset)] = [
                (idx + offset, exp) for idx, exp in tail
            ]

        offset += factor.number_of_generators

    grp = PcGroup(
        combined_number_of_generators,
        orders=combined_orders,
        conjugation_exponents=combined_conjugation_exponents,
        power_tails=combined_power_tails,
        conjugation_tails=combined_conjugation_tails,
    )
    assert grp.test_consistency()
    return grp
