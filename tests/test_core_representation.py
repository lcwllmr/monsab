from monsab.core import (
    BaumClausenStage,
    BaumClausenPaths,
    Permutation,
    PcGroup,
    MonomialRepresentation,
    MonomialRepresentationBundle,
    analyze_monomial_representations,
)
from monsab.pop import MonomialSpace, build_monomial_sab


def test_analyze_monomial_representations_c3():
    """
    Test direct analysis of C3 representations without constructing any monomial space.
    """
    presentation = PcGroup(
        number_of_generators=1,
        orders=(3,),
        conjugation_exponents={},
        power_tails={0: ()},
        conjugation_tails={},
    )
    g = Permutation((1, 2, 0))
    G_gens = {1: g}

    stages = [BaumClausenStage.trivial(e=3, presentation=presentation)]
    for k, order in enumerate(presentation.orders, start=1):
        stages.append(BaumClausenStage.next_level(stages[-1], g_i=k, p=order))

    paths = BaumClausenPaths.from_baum_clausen(tuple(stages))

    bundle = analyze_monomial_representations(paths, G_gens, n=3)

    assert isinstance(bundle, MonomialRepresentationBundle)
    assert len(bundle.representations) == 3
    assert bundle.e == 3

    fs_values = [rep.fs_indicator for rep in bundle.representations]
    assert fs_values.count(1) == 1
    assert fs_values.count(0) == 2

    for rep in bundle.representations:
        assert isinstance(rep, MonomialRepresentation)
        assert rep.dim == 1
        assert rep.e == 3
        if rep.fs_indicator == 0:
            assert (
                rep.id in bundle.realize_skip_reps
                or rep.conjugate_id in bundle.realize_skip_reps
            )


def test_analyze_monomial_representations_c4():
    """
    Test representation analysis on cyclic group C4 (order 4).
    Has two real 1D irreps (fs=1) and two complex conjugate irreps (fs=0).
    """
    presentation = PcGroup(
        number_of_generators=1,
        orders=(4,),
        conjugation_exponents={},
        power_tails={0: ()},
        conjugation_tails={},
    )
    g = Permutation((1, 2, 3, 0))
    G_gens = {1: g}

    stages = [BaumClausenStage.trivial(e=4, presentation=presentation)]
    stages.append(BaumClausenStage.next_level(stages[-1], g_i=1, p=4))

    paths = BaumClausenPaths.from_baum_clausen(tuple(stages))
    bundle = analyze_monomial_representations(paths, G_gens, n=4)

    assert len(bundle.representations) == 4
    fs_values = [rep.fs_indicator for rep in bundle.representations]
    assert fs_values.count(1) == 2
    assert fs_values.count(0) == 2


def test_build_monomial_sab_with_bundle():
    """
    Test that build_monomial_sab accepts a precomputed MonomialRepresentationBundle directly.
    """
    presentation = PcGroup(
        number_of_generators=1,
        orders=(3,),
        conjugation_exponents={},
        power_tails={0: ()},
        conjugation_tails={},
    )
    g = Permutation((1, 2, 0))
    G_gens = {1: g}

    stages = [BaumClausenStage.trivial(e=3, presentation=presentation)]
    stages.append(BaumClausenStage.next_level(stages[-1], g_i=1, p=3))

    paths = BaumClausenPaths.from_baum_clausen(tuple(stages))
    bundle = analyze_monomial_representations(paths, G_gens, n=3)

    space = MonomialSpace(3)
    orbits_space = [tuple(range(1, 4))]

    transform_from_bundle = build_monomial_sab(bundle, G_gens, orbits_space, space, 1)
    transform_from_paths = build_monomial_sab(paths, G_gens, orbits_space, space, 1)

    assert len(transform_from_bundle.blocks) == len(transform_from_paths.blocks)
    for b1, b2 in zip(transform_from_bundle.blocks, transform_from_paths.blocks):
        assert b1.rep_id == b2.rep_id
        assert getattr(b1, "fs_indicator", None) == getattr(b2, "fs_indicator", None)
