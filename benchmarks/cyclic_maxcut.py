import argparse
import time

import cvxpy as cp

from monsab import (
    SquarefreeMonomialSpace,
    build_sab,
    LasserreHierarchy,
    Permutation,
    zoo,
)


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
    parser.add_argument("-n", "--vertices", type=int, default=7)
    parser.add_argument("-t", "--level", type=int, default=2)
    parser.add_argument("--dont-solve", action="store_true", help="Skip SDP solve")
    args = parser.parse_args()

    n = args.vertices
    t = args.level

    print("=== Cyclic MaxCut (0-1) via Lasserre hierarcy ===")
    print(f"Vertices (n): {n}")
    print(f"Hierarchy level (t): {t}")

    # 1. Group
    perms = [Permutation(tuple((i + 1) % n for i in range(n)))]
    grp = zoo.cyclic(n).with_generators(perms)
    concrete_generators = {1: perms[0]}

    # 2. Spaces
    space = SquarefreeMonomialSpace(n)
    N_2t = space.total_monomials(2 * t)
    N_t = space.total_monomials(t)
    print(f"Monomial space 2t size: {N_2t}")
    print(f"Monomial space t size: {N_t}")

    orbits_2t = timed_step(
        "Build orbits for 2t",
        lambda: space.get_orbit_reps_and_sizes(grp, 2 * t),
    )

    # 3. SAB Transform
    all_g = [Permutation(tuple((j + i) % n for j in range(n))) for i in range(n)]
    coset_reps = {rep_id: all_g for rep_id in range(n)}

    transform = timed_step(
        "Initialize SAB transform",
        lambda: build_sab(space, grp, concrete_generators, d=t, coset_reps=coset_reps),
    )

    # 4. Generate Lasserre Matrices
    lh = LasserreHierarchy(space, t, concrete_generators)

    orbits_2t_list = list(orbits_2t.items())
    rep_matrices = []

    def build_matrices():
        for rep, _size in orbits_2t_list:
            mat = lh.moment_matrix(rep)
            rep_matrices.append(mat)

    timed_step("Build representative matrices", build_matrices)

    # 5. Extract blocks
    def extract_blocks():
        # Scaled to exactly match explicit orbit sum
        blocks_by_rep = [
            [orbits_2t_list[i][1] * b for i, b in enumerate(block_mats)]
            for block_mats in transform(rep_matrices, reynolds=True, realize=True)
        ]

        return blocks_by_rep

    blocks_by_rep = timed_step("Extract SAB blocks (reynolds=True)", extract_blocks)

    num_orbits = len(orbits_2t)

    # 6. Formulate SDP
    def solve_sdp():
        y = cp.Variable(num_orbits)
        constraints = []

        empty_rank = space.rank_tuple(())
        empty_idx = next(
            i for i, (rep, size) in enumerate(orbits_2t_list) if rep == empty_rank
        )
        constraints.append(y[empty_idx] == 1)

        x0_rank = space.rank_tuple((0,))
        x0_idx = next(
            i for i, (rep, size) in enumerate(orbits_2t_list) if rep == x0_rank
        )

        # Find edge {0, 1} orbit representative
        # But wait! For cyclic group, the canonical rep for {i, j} is either {0, (j-i) % n} or {0, (i-j) % n}
        edge_canonical = min((0, 1), (0, n - 1))
        edge_canonical_rank = space.rank_tuple(edge_canonical)
        edge_idx = next(
            i
            for i, (rep, size) in enumerate(orbits_2t_list)
            if rep == edge_canonical_rank
        )

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
        return prob, obj, constraints

    prob, obj, constraints = timed_step("Build SDP constraints", solve_sdp)

    # Compute statistics
    total_psd_size = 0
    block_sizes = {}
    for block_list in blocks_by_rep:
        if block_list:
            size = block_list[0].shape[0]
            if size > 0:
                total_psd_size += size
                block_sizes[size] = block_sizes.get(size, 0) + 1

    print("\n--- SDP Statistics ---")
    print(f"Total PSD variable size: {total_psd_size}")
    print(f"Number of constraints: {len(constraints)}")
    print("Block size histogram:")
    for size in sorted(block_sizes.keys()):
        print(f"  {size}x{size}: {block_sizes[size]}")
    print("----------------------")

    if args.dont_solve:
        print("\nSkipping solve as requested.")
        return

    def do_solve():
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)
        return prob.value, prob.solver_stats

    val, stats = timed_step("Solve SDP", do_solve)
    print(f"\nSDP Value: {val:.6f}")

    # Exact cyclic maxcut solutions for small n are known:
    # if n is even, maxcut is n. If n is odd, maxcut is n - 1.
    print(f"Known exact maxcut for n={n}: {n if n % 2 == 0 else n - 1}")


if __name__ == "__main__":
    main()
