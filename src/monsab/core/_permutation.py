"""
Permutations.
"""

from dataclasses import dataclass


@dataclass
class Permutation:
    """A permutation represented by its zero-indexed image tuple."""

    data: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> int:
        return self.data[index] if index < self.size else index

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Permutation):
            return False
        n = max(self.size, other.size)
        return all(self[i] == other[i] for i in range(n))

    def __mul__(self, other: Permutation) -> Permutation:
        n = max(self.size, other.size)
        return Permutation(
            tuple(
                self[other.data[i]] if i < other.size else self.data[i]
                for i in range(n)
            )
        )

    def __pow__(self, p: int) -> Permutation:
        if type(p) is not int:
            raise ValueError("Power must be an integer.")
        if p == 0:
            return Permutation.identity(self.size)
        if p < 0:
            return (~self) ** (-p)
        result = Permutation.identity(self.size)
        factor = self
        exp = p
        while exp > 0:
            if exp & 1:
                result = result * factor
            factor = factor * factor
            exp >>= 1
        return result

    def __invert__(self) -> Permutation:
        return Permutation(tuple(self.data.index(i) for i in range(self.size)))

    @classmethod
    def identity(cls, n: int) -> Permutation:
        """Returns the identity permutation of size n."""
        return cls(tuple(range(n)))
