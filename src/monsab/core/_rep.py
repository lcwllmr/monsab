from dataclasses import dataclass
import numpy as np

@dataclass
class MonomialRepresentation:
    # dimensions
    m_k # block dimension (= number of admissible orbits)
    M: int # number of bins (= order of the cyclic character image)

    # row extraction data
    orbit_reps: np.ndarray[int] # list of representative basis vector indices (length m_k)
    orbit_sizes: np.ndarray[int] # and their orbit sizes

    # column binning data: the master routing array of size equal to dimension.
    # - if column w admissible: col_to_bin[w] = (j, L),
    #   where j is the block index (0...m_k) and L the integer phase bin (0...M-1)
    col_to_bin: np.ndarray[tuple[int,int]]

