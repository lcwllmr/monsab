"""
Core algorithms and data structures.
"""

from monsab._backend import (
    MonomialMatrix,
    Permutation,
    PcGroup,
    evaluate_word,
    SABBlock,
    SABTransform,
    OrbitLifter,
)
from ._baum_clausen import BaumClausenStage, Representation, BaumClausenPaths, Word
from ._representation import (
    MonomialRepresentation,
    MonomialRepresentationBundle,
    analyze_monomial_representations,
)

__all__ = [
    "MonomialMatrix",
    "Permutation",
    "PcGroup",
    "evaluate_word",
    "Word",
    "BaumClausenStage",
    "Representation",
    "SABTransform",
    "BaumClausenPaths",
    "SABBlock",
    "OrbitLifter",
    "MonomialRepresentation",
    "MonomialRepresentationBundle",
    "analyze_monomial_representations",
]
