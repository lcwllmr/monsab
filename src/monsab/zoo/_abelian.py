from ._base import MonomialGroup
from ._product import DirectProduct
from monsab.core import Permutation

class Cyclic(MonomialGroup):
    def __init__(generator: Permutation):
        pass

class Abelian(DirectProduct):
    def __init__(*factors: iterable[Cyclic])
