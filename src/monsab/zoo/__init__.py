"""Zoo of polycyclic groups."""

from ._abelian import abelian, cyclic
from ._base import trivial
from ._metacyclic import affine_group_1d, dihedral, metacyclic
from ._product import direct_product

__all__ = [
    # abelian
    "abelian",
    "cyclic",
    # base
    "trivial",
    # metacyclic
    "affine_group_1d",
    "dihedral",
    "metacyclic",
    # product
    "direct_product",
]
