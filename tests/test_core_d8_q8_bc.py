from monsab.core import BaumClausenStage, PolycyclicPresentation


def test_baum_clausen_d8_q8():
    # Construct D8 * Q8 central product
    # Generators (bottom-up): z, x, y, r, s
    conjugation_exponents = {}
    conjugation_tails = {}
    for k in range(5):
        for j in range(k):
            conjugation_exponents[(j, k)] = 1
            conjugation_tails[(j, k)] = ()

    # s^-1 r s = r z -> j=3(r), k=4(s)
    conjugation_tails[(3, 4)] = ((0, 1),)
    # y^-1 x y = x z -> j=1(x), k=2(y)
    conjugation_tails[(1, 2)] = ((0, 1),)

    presentation = PolycyclicPresentation(
        number_of_generators=5,
        orders=(2, 2, 2, 2, 2),
        conjugation_exponents=conjugation_exponents,
        power_tails={
            0: (),  # z^2 = 1
            1: ((0, 1),),  # x^2 = z
            2: ((0, 1),),  # y^2 = z
            3: ((0, 1),),  # r^2 = z
            4: (),  # s^2 = 1
        },
        conjugation_tails=conjugation_tails,
    )

    e_safe = 8
    stages = [BaumClausenStage.trivial(e=e_safe, presentation=presentation)]
    for k, order in enumerate(presentation.orders, start=1):
        stages.append(BaumClausenStage.next_level(stages[-1], g_i=k, p=order))

    reps = stages[-1].representations
    dims = sorted([rep.dim for rep in reps])

    # D8 * Q8 has sixteen 1D irreps and one 4D irrep
    assert dims == ([1] * 16 + [4])
