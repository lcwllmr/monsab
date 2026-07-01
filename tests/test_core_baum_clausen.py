import pytest
from monsab.core import (
    BaumClausenStage,
    PcGroup,
)
from monsab.zoo import cyclic, abelian, dihedral, metacyclic


def run_baum_clausen(grp, e):
    presentation = grp.description
    stage = BaumClausenStage.trivial(e=e, presentation=presentation)
    for k, order in enumerate(presentation.orders, start=1):
        stage = BaumClausenStage.next_level(stage, g_i=k, p=order)
    return stage.representations


def test_dihedral_group():
    # D_{2p} for p=5. Group of order 10. Exponent e=10.
    # Chief series: G_2 = D_{10} > G_1 = Z_5 > {1}
    e = 10

    # We create a dummy presentation for D_10
    # Generators: g_1 (order 5), g_2 (order 2)
    # 1 is represented by an empty word ()
    presentation = PcGroup(
        number_of_generators=2,
        orders=(5, 2),
        power_tails={0: (), 1: ()},
        conjugation_tails={(0, 1): ((0, 4),)},
        conjugation_exponents={(0, 1): 4},
    )

    # Level 0
    stage0 = BaumClausenStage.trivial(
        e=e,
        presentation=presentation,
    )

    # Level 1 (G_1 = Z_5)
    # p_1 = 5, g_1 = 1
    stage1 = BaumClausenStage.next_level(stage0, g_i=1, p=5)

    assert len(stage1.representations) == 5
    for rep in stage1.representations:
        assert rep.dim == 1

    # Level 2 (G_2 = D_{10})
    stage2 = BaumClausenStage.next_level(stage1, g_i=2, p=2)

    # D_10 has 2 1-dim representations and 2 2-dim representations.
    assert len(stage2.representations) == 4
    dims = sorted([rep.dim for rep in stage2.representations])
    assert dims == [1, 1, 2, 2]


@pytest.mark.parametrize("p", [2, 3, 5, 7, 11])
def test_cyclic_baum_clausen(p):
    grp = cyclic(p)
    reps = run_baum_clausen(grp, e=p)
    assert len(reps) == p
    for rep in reps:
        assert rep.dim == 1


@pytest.mark.parametrize("p1, p2", [(2, 2), (2, 3), (3, 5)])
def test_abelian_baum_clausen(p1, p2):
    grp = abelian(p1, p2)
    reps = run_baum_clausen(grp, e=p1 * p2)
    expected_len = p1 * p2
    assert len(reps) == expected_len
    for rep in reps:
        assert rep.dim == 1


@pytest.mark.parametrize("p", [3, 5, 7, 11])
def test_dihedral_baum_clausen(p):
    grp = dihedral(p)
    reps = run_baum_clausen(grp, e=p * 2)
    dims = sorted([rep.dim for rep in reps])
    expected_dims = [1, 1] + [2] * ((p - 1) // 2)
    assert dims == expected_dims


@pytest.mark.parametrize(
    "p, q, r",
    [
        (5, 2, 4),  # D_10: Z_5 x Z_2
        (7, 3, 2),  # Z_7 x Z_3
        (11, 5, 3),  # Z_11 x Z_5
    ],
)
def test_metacyclic_baum_clausen(p, q, r):
    grp = metacyclic(p, q, r)
    reps = run_baum_clausen(grp, e=p * q)
    dims = sorted([rep.dim for rep in reps])
    expected_dims = [1] * q + [q] * ((p - 1) // q)
    assert dims == expected_dims


def test_baum_clausen_s3_z2():
    from monsab.zoo import direct_product, dihedral, cyclic
    from monsab.core import BaumClausenPaths
    import pytest

    grp = direct_product(dihedral(3), cyclic(2))
    e = 6
    stages = [BaumClausenStage.trivial(e, grp.description)]
    stages.append(BaumClausenStage.next_level(stages[-1], 1, 3))
    stages.append(BaumClausenStage.next_level(stages[-1], 2, 2))
    stages.append(BaumClausenStage.next_level(stages[-1], 3, 2))

    paths = BaumClausenPaths.from_baum_clausen(tuple(stages))
    assert paths.e == 6

    with pytest.raises(ValueError):
        BaumClausenPaths.from_baum_clausen(())
