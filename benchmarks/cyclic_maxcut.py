import argparse
import time

import cvxpy as cp

from monsab.core import BaumClausenPaths, BaumClausenStage, Permutation
from monsab.pop import SquarefreeMonomialSpace, build_monomial_sab
from monsab.pop._lasserre import LasserreHierarchy
from monsab.zoo import cyclic


def format_time(elapsed: float) -> str:
    if elapsed < 1e-3:
        return f"{elapsed * 1e6:.1f}us"
    elif elapsed < 1:
        return f"{elapsed * 1000:.1f}ms"
    elif elapsed < 60:
        return f"{elapsed:.2f}s"
    else:
        m = int(elapsed // 60)
        s = elapsed % 60
        return f"{m}m {s:.1f}s"


def timed_step(name, func):
    print(f"[*] {name:<50} ... ", end="", flush=True)
    start = time.perf_counter()
    res = func()
    elapsed = time.perf_counter() - start
    print(f"done in {format_time(elapsed)}")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--vertices", type=int, required=True)
    parser.add_argument("-t", "--degree", type=int, required=True)
    args = parser.parse_args()

    n = args.vertices
    t = args.degree

    print("=== Cyclic MaxCut (0-1) via Lasserre hierarcy ===")
    print(f"Vertices (n): {n}")
    print(f"Hierarchy level (t): {t}")

    # 1. Group
    grp = cyclic(n)
    perms = [Permutation(tuple((i + 1) % n for i in range(n)))]
    concrete_generators = {1: perms[0]}

    stages = [BaumClausenStage.trivial(e=n, presentation=grp.description)]
    nxt = BaumClausenStage.next_level(stages[-1], g_i=1, p=n)
    stages.append(nxt)
    paths = BaumClausenPaths.from_baum_clausen(tuple(stages))

    # 2. Spaces
    space2t = SquarefreeMonomialSpace(n, 2 * t)
    space_t = SquarefreeMonomialSpace(n, t)
    print(f"Monomial space 2t size: {space2t.total_monomials}")
    print(f"Monomial space t size: {space_t.total_monomials}")

    orbits_2t = timed_step(
        "Build orbits for 2t",
        lambda: space2t.get_full_orbits(concrete_generators, num_threads=1),
    )
    orbits_t = timed_step(
        "Build orbits for t",
        lambda: space_t.get_full_orbits(concrete_generators, num_threads=1),
    )

    # 3. SAB Transform
    # To use implicit Reynolds, we must provide coset_reps for the fast Coset Averaging.
    # For cyclic(n), we can just provide all group elements for every representation.
    all_g = [Permutation(tuple((j + i) % n for j in range(n))) for i in range(n)]
    coset_reps = {rep_id: all_g for rep_id in paths.paths.keys()}

    transform = timed_step(
        "Initialize SAB transform",
        lambda: build_monomial_sab(
            paths, concrete_generators, orbits_t, space_t, coset_reps=coset_reps
        ),
    )

    # 4. Generate Lasserre Matrices
    lh = LasserreHierarchy(space2t, t, concrete_generators)

    rep_matrices = []

    def build_matrices():
        for orb in orbits_2t:
            mat = lh.moment_matrix(orb, reynolds=True)
            rep_matrices.append(mat)

    timed_step("Build representative matrices", build_matrices)

    # 5. Extract blocks
    def extract_blocks():
        # Unscaled implicit Reynolds moment matrix
        # blocks_by_rep = transform(rep_matrices, reynolds=True, realize=True)

        # Scaled to exactly match explicit orbit sum
        blocks_by_rep = [
            [len(orbits_2t[i]) * b for i, b in enumerate(block_mats)]
            for block_mats in transform(rep_matrices, reynolds=True, realize=True)
        ]

        return blocks_by_rep

    blocks_by_rep = timed_step("Extract SAB blocks (reynolds=True)", extract_blocks)

    num_orbits = len(orbits_2t)

    # 6. Formulate SDP
    def solve_sdp():
        y = cp.Variable(num_orbits)
        constraints = []

        empty_rank = space2t.rank_tuple(())
        empty_idx = next(i for i, orb in enumerate(orbits_2t) if orb[0] == empty_rank)
        constraints.append(y[empty_idx] == 1)

        x0_rank = space2t.rank_tuple((0,))
        x0_idx = next(i for i, orb in enumerate(orbits_2t) if orb[0] == x0_rank)

        # Find edge {0, 1} orbit
        edge_rank = space2t.rank_tuple((0, 1))
        edge_idx = next(i for i, orb in enumerate(orbits_2t) if edge_rank in orb)

        obj = cp.Maximize(n * (2 * y[x0_idx] - 2 * y[edge_idx]))

        for k in range(len(blocks_by_rep)):
            dim = blocks_by_rep[k][0].shape[0]
            if dim == 0:
                continue

            block_k_expr = sum(
                y[i] * blocks_by_rep[k][i].toarray()
                for i in range(num_orbits)
                if blocks_by_rep[k][i].nnz > 0
            )

            if isinstance(block_k_expr, int):
                continue

            # Enforce symmetry explicitly since cvxpy can be picky with numerical noise
            sym_expr = 0.5 * (block_k_expr + block_k_expr.T)
            constraints.append(sym_expr >> 0)

        prob = cp.Problem(obj, constraints)

        # Try CLARABEL first, which is more robust for SDPs
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)

        return prob.value, prob.solver_stats

    val, stats = timed_step("Solve SDP", solve_sdp)
    print(f"\nSDP Value: {val:.6f}")

    # Exact cyclic maxcut solutions for small n are known:
    # if n is even, maxcut is n. If n is odd, maxcut is n - 1.
    print(f"Known exact maxcut for n={n}: {n if n % 2 == 0 else n - 1}")


if __name__ == "__main__":
    main()
