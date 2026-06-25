"""
Monomial matrices.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MonomialMatrix:
    """Represents a monomial matrix with a permutation and phase values."""

    perm: tuple[int, ...]
    vals: tuple[int, ...]
    e: int

    @property
    def dim(self) -> int:
        return len(self.perm)

    def __post_init__(self) -> None:
        if self.e <= 0:
            raise ValueError("Group exponent e must be a positive integer.")
        if len(self.perm) != len(self.vals):
            raise ValueError("perm and vals must have equal length.")
        if set(self.perm) != set(range(self.dim)):
            raise ValueError("perm must be a permutation of 0..d-1.")

    @classmethod
    def identity(cls, dim: int, e: int) -> MonomialMatrix:
        """Returns the identity monomial matrix of given dimension and exponent e."""
        if type(dim) is not int or dim < 0:
            raise ValueError("Matrix dimension must be a non-negative integer.")
        return cls(perm=tuple(range(dim)), vals=tuple(0 for _ in range(dim)), e=e)

    def __add__(self, other: MonomialMatrix) -> MonomialMatrix:
        """Builds the direct sum of two monomial matrices."""
        if self.e != other.e:
            msg = "Cannot add monomial matrices with different exponents e."
            raise ValueError(msg)
        return MonomialMatrix(
            perm=self.perm + tuple(i + self.dim for i in other.perm),
            vals=self.vals + other.vals,
            e=self.e,
        )

    def __matmul__(self, other: MonomialMatrix) -> MonomialMatrix:
        if self.e != other.e:
            msg = "Cannot multiply monomial matrices with different exponents e."
            raise ValueError(msg)
        if len(self.perm) != len(other.perm):
            msg = "Cannot multiply monomial matrices with different dimensions."
            raise ValueError(msg)
        return MonomialMatrix(
            perm=tuple(other.perm[j] for j in self.perm),
            vals=tuple(
                (self.vals[i] + other.vals[j]) % self.e for i, j in enumerate(self.perm)
            ),
            e=self.e,
        )

    def __pow__(self, p: int) -> MonomialMatrix:
        if type(p) is not int:
            raise ValueError("Power must be an integer.")
        if p == 0:
            return MonomialMatrix.identity(dim=len(self.perm), e=self.e)
        if p < 0:
            return self.inverse() ** (-p)
        result = MonomialMatrix.identity(dim=len(self.perm), e=self.e)
        factor = self
        exp = p
        while exp > 0:
            if exp & 1:
                result = result @ factor
            factor = factor @ factor
            exp >>= 1
        return result

    def inverse(self) -> MonomialMatrix:
        return MonomialMatrix(
            perm=tuple(self.perm.index(i) for i in range(len(self.perm))),
            vals=tuple(
                (-self.vals[self.perm.index(i)]) % self.e for i in range(len(self.perm))
            ),
            e=self.e,
        )
