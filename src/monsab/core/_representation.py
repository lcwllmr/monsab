"""
Monomial representations and character induction analysis.

This module provides data structures and algorithms for handling representations induced from
1-dimensional characters of subgroups, computing Frobenius-Schur indicators, and resolving
real-realization intertwiners for Symmetry-Adapted Basis constructions.
"""

import operator
from collections import deque
from collections.abc import Mapping
from monsab import _backend
from monsab._backend import (
    Permutation,
    MonomialRepresentation,
    MonomialRepresentationBundle,
)
from ._baum_clausen import BaumClausenPaths

type InductionStep = tuple[int, list[tuple[int, int]], int]
type PathDict = Mapping[int, list[InductionStep]]


def analyze_monomial_representations(
    abstract: BaumClausenPaths,
    G_gens: Mapping[int, Permutation],
    n: int,
) -> MonomialRepresentationBundle:
    """
    Analyze abstract character induction paths to compute Frobenius-Schur indicators,
    real-realization transformation matrices, and conjugate representation pairings.

    Args:
        abstract: The abstract Baum-Clausen paths defining the irreducible representations.
        G_gens: The generators of the group $G$, mapping generator IDs to `Permutation` objects.
        n: The degree of the permutation group action (number of domain points).

    Returns:
        MonomialRepresentationBundle: A bundle containing analyzed representations and intertwiner data.
    """
    g_inv_dict = {gen_id: list((~gen).data) for gen_id, gen in G_gens.items()}

    paths_dict: dict[int, list[InductionStep]] = {}
    for rep_id, path_entries in abstract.paths.items():
        new_entries: list[InductionStep] = []
        for g_k, adm, lambda_i in path_entries:
            new_entries.append((g_k, list(adm), lambda_i))
        paths_dict[rep_id] = new_entries

    identity_data = tuple(range(len(g_inv_dict[next(iter(g_inv_dict))])))

    fs_indicators: dict[int, int] = {}
    v_matrices: dict[int, list[int]] = {}

    H_visited_all: dict[int, dict[tuple[int, ...], int]] = {}
    is_complex_chi_all: dict[int, bool] = {}
    H_gens_all: dict[int, tuple[dict[int, tuple[int, ...]], dict[int, int]]] = {}

    for rep_id, paths in abstract.paths.items():
        if abstract.conjugates.get(rep_id, rep_id) != rep_id:
            fs_indicators[rep_id] = 0
            continue

        is_1d = all(g_k != 0 for g_k, _, _ in paths)
        if is_1d:
            # 1D reps that are self-conjugate are real-valued.
            fs_indicators[rep_id] = 1
            is_complex_chi_all[rep_id] = False
            continue

        H_gens: dict[int, list[int]] = {}
        H_gens_fwd: dict[int, tuple[int, ...]] = {}
        char_phases_rep: dict[int, int] = {}
        for g_k, _, lambda_i in paths:
            if g_k != 0:
                H_gens[g_k] = g_inv_dict[g_k]

                inv_data = g_inv_dict[g_k]
                fwd_data = [0] * len(inv_data)
                for i, v in enumerate(inv_data):
                    fwd_data[v] = i
                H_gens_fwd[g_k] = tuple(fwd_data)

                char_phases_rep[g_k] = lambda_i

        H_gens_all[rep_id] = (H_gens_fwd, char_phases_rep)

        H_visited: dict[tuple[int, ...], int] = {identity_data: 0}
        H_queue = deque([identity_data])
        H_gens_ig = {
            g_k: operator.itemgetter(*g_data) for g_k, g_data in H_gens.items()
        }

        while H_queue:
            curr = H_queue.popleft()
            curr_phase = H_visited[curr]

            for g_k, ig in H_gens_ig.items():
                nxt = ig(curr)
                if nxt not in H_visited:
                    nxt_phase = (curr_phase + char_phases_rep[g_k]) % abstract.e
                    H_visited[nxt] = nxt_phase
                    H_queue.append(nxt)

        H_visited_all[rep_id] = H_visited

        is_complex_chi = False
        for p in H_visited.values():
            if p != 0 and p * 2 != abstract.e:
                is_complex_chi = True
                break
        is_complex_chi_all[rep_id] = is_complex_chi

    g_gens_data = [list(g.data) for g in G_gens.values()]
    rust_fs, raw_v_matrices = _backend.compute_fs_and_t_data(
        g_gens_data,
        n,
        abstract.e,
        H_visited_all,
        {rep_id: H_gens_all[rep_id][0] for rep_id in H_gens_all},
        {rep_id: H_gens_all[rep_id][1] for rep_id in H_gens_all},
    )
    fs_indicators.update(rust_fs)

    v_matrices = {k: v for k, v in raw_v_matrices.items() if v is not None}

    realize_skip_reps: set[int] = set()
    matched: set[int] = set()
    for r1, nu2 in fs_indicators.items():
        if nu2 == 0 and r1 not in matched:
            r2 = abstract.conjugates.get(r1, r1)
            matched.add(r1)
            matched.add(r2)
            realize_skip_reps.add(max(r1, r2))

    dims = getattr(abstract, "dims", None)
    reps_list: list[MonomialRepresentation] = []
    for rep_id, path_entries in abstract.paths.items():
        if dims is not None and rep_id in dims:
            dim = dims[rep_id]
        else:
            dim = 2 ** sum(1 for g_k, _, _ in path_entries if g_k == 0)
        reps_list.append(
            MonomialRepresentation(
                id=rep_id,
                dim=dim,
                e=abstract.e,
                conjugate_id=abstract.conjugates.get(rep_id, rep_id),
                fs_indicator=fs_indicators.get(rep_id),
                v_matrix=v_matrices.get(rep_id),
            )
        )

    return MonomialRepresentationBundle(
        representations=tuple(reps_list),
        paths_dict=paths_dict,
        fs_indicators=fs_indicators,
        v_matrices=v_matrices,
        realize_skip_reps=realize_skip_reps,
        e=abstract.e,
    )
