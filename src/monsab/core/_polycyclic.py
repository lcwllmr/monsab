"""
Polycyclic presentations.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from monsab.util import is_prime

from ._permutation import Permutation

type Word = tuple[tuple[int, int], ...]


def _evaluate_word(word: Word, generators: tuple[Permutation, ...]) -> Permutation:
    """Evaluate a word in the generators to produce a permutation."""
    result = Permutation.identity(len(generators))
    for gen_index, exponent in word:
        result = result * (generators[gen_index] ** exponent)
    return result


@dataclass(frozen=True, slots=True)
class PolycyclicPresentation:
    number_of_generators: int
    orders: tuple[int, ...]
    conjugation_exponents: Mapping[tuple[int, int], int]
    power_tails: Mapping[int, Word]
    conjugation_tails: Mapping[tuple[int, int], Word]

    def verify(self, generators: tuple[Permutation, ...]) -> bool:
        """
        Check whether the given generator permutations verify all relations of the presentation.
        """
        if self.number_of_generators != len(generators):
            return False

        if not generators:
            return True

        for k in range(self.number_of_generators):
            order = self.orders[k]
            lhs = generators[k] ** order
            tail = self.power_tails.get(k, ())
            rhs = _evaluate_word(tail, generators)
            if lhs != rhs:
                return False

        all_pairs = set(self.conjugation_exponents.keys()) | set(
            self.conjugation_tails.keys()
        )
        for j, k in all_pairs:
            c = self.conjugation_exponents.get((j, k), 0)
            tail = self.conjugation_tails.get((j, k), ())

            # g_k^-1 g_j g_k = g_j^c * tail
            lhs = (~generators[k]) * generators[j] * generators[k]
            rhs = (generators[j] ** c) * _evaluate_word(tail, generators)
            if lhs != rhs:
                return False

        return True

    @property
    def order(self) -> int:
        """
        Computes the order of the group defined by the presentation.
        """
        order = 1
        for r in self.orders:
            order *= r
        return order

    @property
    def has_only_prime_factors(self) -> bool:
        """
        Checks if all relative orders in the presentation are prime.
        """
        return all(is_prime(r) for r in self.orders)

    @property
    def is_normal_series(self) -> bool:
        """
        Checks global normality: verifies if G_j is normal in G for all j.
        Requires that for any j and conjugator k < j, the evaluated word
        g_k^{-1} g_j g_k only contains generators >= j.
        """
        # 1. Check conjugation relations: g_j^{g_k} = g_j^c * tail
        for (j, k), tail_word in self.conjugation_tails.items():
            # If the tail contains any generator < j, the conjugate
            # falls outside of G_j, meaning G_j is not globally normal.
            for gen_idx, _ in tail_word:
                if gen_idx < j:
                    return False

        # 2. Check power relations: g_k^{r_k} = tail
        # The power tail of g_k must strictly reside in G_{k+1}.
        for k, tail_word in self.power_tails.items():
            for gen_idx, _ in tail_word:
                if gen_idx <= k:
                    return False

        return True
