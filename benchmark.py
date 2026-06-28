"""
CLI Benchmark for the Monomial Symmetry Adapted Basis generation.
"""

import argparse
import time
from collections import defaultdict
from typing import Any, Callable

import numpy as np
import scipy.sparse

from monsab.core import (
    BaumClausenPaths,
    BaumClausenStage,
    Permutation,
)
from monsab.pop import (
    MonomialSpace,
    SquarefreeMonomialSpace,
    build_monomial_sab,
)
from monsab.util import is_prime, primitive_root
from monsab.zoo import affine_group_1d, cyclic, dihedral


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


def timed_step(name: str, func: Callable[[], Any]) -> Any:
    print(f"[*] {name:<50} ... ", end="", flush=True)
    start = time.perf_counter()
    res = func()
    elapsed = time.perf_counter() - start
    print(f"done in {format_time(elapsed)}")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Monomial SAB Benchmark CLI")
    parser.add_argument(
        "-n", "--variables", type=int, required=True, help="Number of variables"
    )
    parser.add_argument(
        "-d", "--degree", type=int, required=True, help="Polynomial degree"
    )
    parser.add_argument(
        "-g",
        "--group",
        type=str,
        required=True,
        choices=["cyclic", "abelian", "dihedral", "affine"],
        help="Group type",
    )
    parser.add_argument(
        "-m",
        "--monomials",
        type=str,
        default="standard",
        choices=["standard", "squarefree"],
        help="Monomial space type (default: standard)",
    )

    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help="Number of threads for multiprocessing parallelization (default: 1)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=1,
        help="Batch size for SAB transform (default: 1)",
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=1,
        help="Number of runs to average batched transform (default: 1)",
    )
    parser.add_argument(
        "-s",
        "--sparsity",
        type=float,
        default=None,
        help="Sparsity factor (average nnz per row/col, can be < 1.0)",
    )

    args = parser.parse_args()
    n = args.variables
    d = args.degree
    group_type = args.group
    space_type = args.monomials
    threads = args.threads
    batch_size = args.batch
    runs = args.runs

    # Sparsity factor s (average non-zeros per row)
    sparsity_factor = args.sparsity if args.sparsity is not None else 4.0

    MonomialClass = (
        MonomialSpace if space_type == "standard" else SquarefreeMonomialSpace
    )

    print("=== Benchmark Configuration ===")
    print(f"Variables (n)    : {n}")
    print(f"Degree (d)       : {d}")
    print(f"Full Dimension   : {MonomialClass(n, d).total_monomials}")
    print(f"Group (g)        : {group_type}")
    print(f"Space (m)        : {space_type}")
    print(f"Threads (t)      : {threads}")
    print(f"Batch Size (b)   : {batch_size}")
    print(f"Runs (r)         : {runs}")
    print(f"Sparsity (s)     : {sparsity_factor} (target nnz/row)")
    print("===============================\n")

    # 1. Instantiate Group & Assertions
    def instantiate_group() -> tuple[Any, dict[int, Permutation]]:
        grp = None
        perms = []
        if group_type == "cyclic":
            grp = cyclic(n)
            perms.append(Permutation(tuple((i + 1) % n for i in range(n))))
        elif group_type == "dihedral":
            assert n >= 3, "Dihedral group requires n >= 3"
            grp = dihedral(n)
            perms.append(Permutation(tuple((i + 1) % n for i in range(n))))
            perms.append(Permutation(tuple((-i) % n for i in range(n))))
        elif group_type == "affine":
            assert is_prime(n), f"Affine group requires prime n, but got {n}"
            grp = affine_group_1d(n)
            g = primitive_root(n)
            perms.append(Permutation(tuple((i + 1) % n for i in range(n))))
            perms.append(Permutation(tuple((g * i) % n for i in range(n))))
        else:
            raise ValueError(f"Unknown group {group_type}")

        assert grp.description.verify(tuple(perms)), (
            "Generated permutations do not satisfy the polycyclic presentation!"
        )

        return grp, {i + 1: p for i, p in enumerate(perms)}

    grp, concrete_generators = timed_step("Instantiate group", instantiate_group)

    # 2. Baum-Clausen
    def run_baum_clausen() -> list[BaumClausenStage]:
        stages = [
            BaumClausenStage.trivial(
                e=grp.description.order, presentation=grp.description
            )
        ]
        for k, order in enumerate(grp.description.orders, start=1):
            nxt = BaumClausenStage.next_level(stages[-1], g_i=k, p=order)
            stages.append(nxt)
        return stages

    bc_stages = timed_step(
        f"Baum-Clausen on abstract group of order {grp.description.order}",
        run_baum_clausen,
    )

    # 3. Build Monomials
    def build_monomials() -> tuple[Any, list[tuple[tuple[int, ...], ...]]]:
        space = MonomialClass(n, d)
        orbit_data = space.get_full_orbits(concrete_generators, num_threads=threads)
        return space, orbit_data

    monomial_space, orbits = timed_step(
        "Build monomial space and orbits",
        build_monomials,
    )

    # 4. Initialize SAB Transform
    def init_sab() -> Any:
        abstract = BaumClausenPaths.from_baum_clausen(tuple(bc_stages))

        return build_monomial_sab(
            abstract,
            concrete_generators,
            orbits,
            monomial_space,
            num_threads=threads,
        )

    transform = timed_step("Initialize SAB transform", init_sab)
    final_blocks = transform.blocks
    monomial_space_d = MonomialClass(n, d)

    def generate_intertwiner_batch() -> list[Any]:

        N = monomial_space_d.total_monomials

        # We don't really care whether the sparse matrices are mathematically
        # commuting with the group action for the sake of the benchmark.
        # The performance of the transform purely depends on the matrix size
        # and non-zero density. Thus, we simplify by generating pure random
        # sparse matrices with Gaussian entries to drastically speed up the benchmark.

        rng = np.random.default_rng(42)
        target_nnz = int(sparsity_factor * N)

        batch = []
        for _ in range(batch_size):
            rows = rng.integers(0, N, size=target_nnz, dtype=np.int32)
            cols = rng.integers(0, N, size=target_nnz, dtype=np.int32)
            data = rng.standard_normal(size=target_nnz, dtype=np.float64)

            mat = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
            batch.append(mat)

        return batch

    # 5. Apply Batched
    def apply_batched() -> tuple[float, float]:
        times = []
        total_nnz = 0
        total_elements = 0

        for _ in range(runs):
            # Shuffle the pre-generated batch instead of generating new ones
            batch_matrices = (
                generate_intertwiner_batch()
            )  # Ensure we have a fresh batch for each run

            for mat in batch_matrices:
                total_nnz += mat.nnz
                total_elements += mat.shape[0] * mat.shape[1]

            start_t = time.perf_counter()
            _ = transform(batch_matrices)
            end_t = time.perf_counter()
            times.append(end_t - start_t)

        avg_batch_time = sum(times) / runs
        avg_per_matrix = avg_batch_time / batch_size
        return avg_batch_time, avg_per_matrix

    avg_batch_time, avg_per_matrix = timed_step("Apply batched", apply_batched)

    # Group by rep_id to find the true Isotypic Component sizes across the whole space
    isotypic_dims = defaultdict(int)
    for b in final_blocks:
        isotypic_dims[b.rep_id] += b.dim

    print("\n=== Statistics ===")
    print(f"Reduced Dimension           : {sum(b.dim for b in final_blocks)}")
    print(f"Total Isotypic Components   : {len(isotypic_dims)}")
    print("Block Size Distribution:")
    size_counts = defaultdict(int)
    for dim in isotypic_dims.values():
        size_counts[dim] += 1
    sorted_sizes = sorted(size_counts.items(), key=lambda x: x[0], reverse=True)
    for size, count in sorted_sizes:
        print(f"  Size {size:<4} -> {count} blocks")

    print("\n=== Batched Transform Performance ===")
    print(f"Average Batch Time (b={batch_size}) : {format_time(avg_batch_time)}")
    print(f"Average Time per Matrix   : {format_time(avg_per_matrix)}")
    print("===============================\n")


if __name__ == "__main__":
    main()
