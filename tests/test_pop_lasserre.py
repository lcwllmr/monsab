import numpy as np

from monsab.pop import MonomialSpace, LasserreHierarchy


def test_lasserre_hierarchy_basic():
    # Variables: 3 (indices 0, 1, 2)
    # Hierarchy level: t=1
    # 2*t = 2. So space2t needs to be up to degree 2.
    n = 3
    t = 1
    space2t = MonomialSpace(n)
    LasserreHierarchy(space2t, t, {})


def test_lasserre_hierarchy_degree_zero():
    space2t = MonomialSpace(3)
    hierarchy = LasserreHierarchy(space2t, 1, {})

    # The trivial monomial is ()
    m_rank = space2t.rank_tuple(())
    rep = m_rank

    mat = hierarchy.moment_matrix(rep)
    dense_mat = mat.toarray()

    # The only pair that multiplies to () is () * ()
    expected = np.zeros((4, 4))
    expected[0, 0] = 1.0

    np.testing.assert_array_equal(dense_mat, expected)


def test_lasserre_hierarchy_degree_two_squared():
    space2t = MonomialSpace(3)
    hierarchy = LasserreHierarchy(space2t, 1, {})

    # The monomial x_0^2 is (0, 0)
    m_rank = space2t.rank_tuple((0, 0))
    rep = m_rank

    mat = hierarchy.moment_matrix(rep)
    dense_mat = mat.toarray()

    # The only pair is x_0 * x_0
    expected = np.zeros((4, 4))
    expected[1, 1] = 1.0

    np.testing.assert_array_equal(dense_mat, expected)
