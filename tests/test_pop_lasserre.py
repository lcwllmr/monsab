import numpy as np
from scipy.sparse import csr_matrix

from monsab.pop import MonomialSpace, LasserreHierarchy


def test_lasserre_hierarchy_basic():
    # Variables: 3 (indices 0, 1, 2)
    # Hierarchy level: t=1
    # 2*t = 2. So space2t needs to be up to degree 2.
    n = 3
    t = 1
    space2t = MonomialSpace(n, 2 * t)

    import pytest

    with pytest.raises(ValueError):
        LasserreHierarchy(MonomialSpace(n, 2 * t - 1), t, {})

    # We will just test with an empty generator set for now,
    # because LasserreHierarchy factorizations do not depend on the generators,
    # only the orbit itself.
    generators = {}

    hierarchy = LasserreHierarchy(space2t, t, generators)

    # Let's manually construct an orbit: just the monomial x_0 * x_1
    # For d=2, the tuples are up to length 2.
    # The monomial x_0 x_1 is represented by tuple (0, 1).
    m_rank = space2t.rank_tuple((0, 1))

    orbit = (m_rank,)

    mat = hierarchy.moment_matrix(orbit)

    assert isinstance(mat, csr_matrix)
    assert mat.shape == (hierarchy.N_t, hierarchy.N_t)

    # For t=1, monomials are:
    # rank 0: ()  (degree 0)
    # rank 1: (0,) (x_0)
    # rank 2: (1,) (x_1)
    # rank 3: (2,) (x_2)
    # Total monomials N_t = 4
    assert hierarchy.N_t == 4

    # The pairs that multiply to x_0 * x_1 are:
    # 1. x_0 (rank 1) and x_1 (rank 2)
    # 2. x_1 (rank 2) and x_0 (rank 1)
    # 3. () (rank 0) and x_0 x_1 (degree 2 > t, so not valid!)

    # So we should only have ones at (1, 2) and (2, 1).
    dense_mat = mat.toarray()

    expected = np.zeros((4, 4))
    expected[1, 2] = 1.0
    expected[2, 1] = 1.0

    np.testing.assert_array_equal(dense_mat, expected)


def test_lasserre_hierarchy_degree_zero():
    space2t = MonomialSpace(3, 2)
    hierarchy = LasserreHierarchy(space2t, 1, {})

    # The trivial monomial is ()
    m_rank = space2t.rank_tuple(())
    orbit = (m_rank,)

    mat = hierarchy.moment_matrix(orbit)
    dense_mat = mat.toarray()

    # The only pair that multiplies to () is () * ()
    expected = np.zeros((4, 4))
    expected[0, 0] = 1.0

    np.testing.assert_array_equal(dense_mat, expected)


def test_lasserre_hierarchy_degree_two_squared():
    space2t = MonomialSpace(3, 2)
    hierarchy = LasserreHierarchy(space2t, 1, {})

    # The monomial x_0^2 is (0, 0)
    m_rank = space2t.rank_tuple((0, 0))
    orbit = (m_rank,)

    mat = hierarchy.moment_matrix(orbit)
    dense_mat = mat.toarray()

    # The only pair is x_0 * x_0
    expected = np.zeros((4, 4))
    expected[1, 1] = 1.0

    np.testing.assert_array_equal(dense_mat, expected)
