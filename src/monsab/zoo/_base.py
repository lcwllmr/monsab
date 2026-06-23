from monsab.core import MonomialRepresentation

class MonomialGroup:
    @abstract
    def compute_irreps(self) -> list[MonomialRepresentation]:
        raise NotImplementedError
    
