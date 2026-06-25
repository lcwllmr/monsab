"""
Monomial spaces and generation.
"""

from monsab.core._transform import SABBlock, SABTransform
from monsab.core._baum_clausen import BaumClausenPaths
from collections.abc import Mapping
import typing

import math
from collections import deque
import array

from monsab.core._permutation import Permutation


def _compute_orbit_chunk(
    args: tuple[int, int, tuple[tuple[int, ...], ...], MonomialSpace],
) -> bytes:
    start, end, g_inv_data, space = args
    n_gens = len(g_inv_data)
    res = array.array("i", [0] * ((end - start) * n_gens))
    idx = 0

    offsets = space.offsets
    b2 = space.binom_2
    b3 = space.binom_3

    m = start
    while m < end:
        if space.d >= 0 and m < offsets[1]:
            # k = 0
            for _ in g_inv_data:
                res[idx] = 0
                idx += 1
            m += 1
            continue

        elif space.d >= 1 and m < offsets[2]:
            # k = 1
            v0 = m - offsets[1]
            end_k = min(end, offsets[2])
            for _ in range(end_k - m):
                for inv in g_inv_data:
                    res[idx] = offsets[1] + inv[v0]
                    idx += 1
                v0 += 1
            m = end_k
            continue

        elif space.d >= 2 and m < offsets[3]:
            # k = 2
            c1, c2_minus_1 = space.unrank_tuple(m)
            v = [c1, c2_minus_1]
            end_k = min(end, offsets[3])

            for _ in range(end_k - m):
                for inv in g_inv_data:
                    w0, w1 = inv[v[0]], inv[v[1]]
                    if w0 > w1:
                        w0, w1 = w1, w0
                    res[idx] = offsets[2] + b2[w1 + 1] + w0
                    idx += 1

                if v[0] < v[1]:
                    v[0] += 1
                else:
                    v[1] += 1
                    v[0] = 0
            m = end_k
            continue

        elif space.d >= 3 and m < offsets[4]:
            # k = 3
            tup = space.unrank_tuple(m)
            v = list(tup)
            end_k = min(end, offsets[4])

            for _ in range(end_k - m):
                for inv in g_inv_data:
                    w0, w1, w2 = inv[v[0]], inv[v[1]], inv[v[2]]
                    if w0 > w1:
                        w0, w1 = w1, w0
                    if w1 > w2:
                        w1, w2 = w2, w1
                    if w0 > w1:
                        w0, w1 = w1, w0
                    res[idx] = offsets[3] + b3[w2 + 2] + b2[w1 + 1] + w0
                    idx += 1

                if v[0] < v[1]:
                    v[0] += 1
                elif v[1] < v[2]:
                    v[1] += 1
                    v[0] = 0
                else:
                    v[2] += 1
                    v[0] = 0
                    v[1] = 0
            m = end_k
            continue

        else:
            # k > 3
            for inv in g_inv_data:
                res[idx] = space.apply_gen(m, inv)
                idx += 1
            m += 1

    return res.tobytes()


class MonomialSpace:
    """
    Handles the abstract space of monomials of degree up to d in n variables.
    Uses O(1) memory via a mathematically dense Combinatorial Bijection, avoiding all tuple objects.
    """

    def __init__(self, n: int, d: int):
        self.n = n
        self.d = d

        # Precompute offsets for each degree k
        self.offsets = [0] * (d + 2)
        total = 0
        for k in range(d + 1):
            self.offsets[k] = total
            total += math.comb(n + k - 1, k)
        self.offsets[d + 1] = total
        self.total_monomials = total

        # Precompute binomial lookup tables for extremely fast unranking
        self.binom_2 = tuple(math.comb(i, 2) for i in range(n + 3))
        self.binom_3 = tuple(math.comb(i, 3) for i in range(n + 3))

    def unrank_tuple(self, m_id: int) -> tuple[int, ...]:
        """Convert an integer ID back to the sparse variable tuple."""
        k = 0
        for i in range(self.d + 1):
            if self.offsets[i] <= m_id < self.offsets[i + 1]:
                k = i
                break

        if k == 0:
            return ()

        rank = m_id - self.offsets[k]

        if k == 1:
            return (rank,)
        elif k == 2:
            c2 = int((rank * 2.0) ** 0.5)
            while self.binom_2[c2] > rank:
                c2 -= 1
            while self.binom_2[c2 + 1] <= rank:
                c2 += 1
            c1 = rank - self.binom_2[c2]
            return (c1, c2 - 1)
        elif k == 3:
            c3 = int((rank * 6.0) ** 0.3333333333333333)
            while self.binom_3[c3] > rank:
                c3 -= 1
            while self.binom_3[c3 + 1] <= rank:
                c3 += 1
            rem = rank - self.binom_3[c3]

            c2 = int((rem * 2.0) ** 0.5)
            while self.binom_2[c2] > rem:
                c2 -= 1
            while self.binom_2[c2 + 1] <= rem:
                c2 += 1
            c1 = rem - self.binom_2[c2]
            return (c1, c2 - 1, c3 - 2)
        else:
            # Generic unrank for d > 3
            c = []
            rem = rank
            for i in range(k, 0, -1):
                if rem == 0:
                    guess = i - 1
                else:
                    guess = int(math.pow(rem * math.factorial(i), 1.0 / i))
                while math.comb(guess, i) > rem:
                    guess -= 1
                while math.comb(guess + 1, i) <= rem:
                    guess += 1
                c.append(guess)
                rem -= math.comb(guess, i)
            return tuple(c[k - 1 - j] - j for j in range(k))

    def rank_tuple(self, tup: tuple[int, ...]) -> int:
        """Convert a sparse variable tuple back to an integer ID."""
        k = len(tup)
        if k == 0:
            return 0
        rank = sum(math.comb(tup[i] + i, i + 1) for i in range(k))
        return self.offsets[k] + rank

    def apply_gen(self, m_id: int, inv_data: tuple[int, ...]) -> int:
        """Apply a permutation strictly on the packed integer representation."""
        if self.d >= 1 and m_id < self.offsets[2]:
            if m_id < self.offsets[1]:
                return 0
            # k = 1
            v = m_id - self.offsets[1]
            return self.offsets[1] + inv_data[v]

        elif self.d >= 2 and m_id < self.offsets[3]:
            # k = 2
            rank = m_id - self.offsets[2]
            c2 = int((rank * 2.0) ** 0.5)
            while self.binom_2[c2] > rank:
                c2 -= 1
            while self.binom_2[c2 + 1] <= rank:
                c2 += 1
            c1 = rank - self.binom_2[c2]

            v0, v1 = inv_data[c1], inv_data[c2 - 1]
            if v0 > v1:
                v0, v1 = v1, v0
            return self.offsets[2] + self.binom_2[v1 + 1] + v0

        elif self.d >= 3 and m_id < self.offsets[4]:
            # k = 3
            rank = m_id - self.offsets[3]
            c3 = int((rank * 6.0) ** 0.3333333333333333)
            while self.binom_3[c3] > rank:
                c3 -= 1
            while self.binom_3[c3 + 1] <= rank:
                c3 += 1
            rem = rank - self.binom_3[c3]

            c2 = int((rem * 2.0) ** 0.5)
            while self.binom_2[c2] > rem:
                c2 -= 1
            while self.binom_2[c2 + 1] <= rem:
                c2 += 1
            c1 = rem - self.binom_2[c2]

            v0, v1, v2 = inv_data[c1], inv_data[c2 - 1], inv_data[c3 - 2]
            if v0 > v1:
                v0, v1 = v1, v0
            if v1 > v2:
                v1, v2 = v2, v1
            if v0 > v1:
                v0, v1 = v1, v0
            return self.offsets[3] + self.binom_3[v2 + 2] + self.binom_2[v1 + 1] + v0

        else:
            # Generic fallback for d > 3
            tup = self.unrank_tuple(m_id)
            new_tup = sorted(inv_data[v] for v in tup)
            return self.rank_tuple(tuple(new_tup))

    def get_orbits(self, G_gens: dict[int, Permutation]) -> list[tuple[int, ...]]:
        """Groups all monomials into G-orbits and returns one representative per orbit."""
        visited = bytearray(self.total_monomials)
        orbit_reps = []

        g_inv_data = [(~gen).data for gen in G_gens.values()]

        for m in range(self.total_monomials):
            if visited[m]:
                continue

            # Return the sparse tuple representation for compatibility with the block format
            orbit_reps.append(self.unrank_tuple(m))

            queue = deque([m])
            visited[m] = 1
            while queue:
                curr = queue.popleft()
                for inv_data in g_inv_data:
                    nxt = self.apply_gen(curr, inv_data)
                    if not visited[nxt]:
                        visited[nxt] = 1
                        queue.append(nxt)

        return orbit_reps

    def get_full_orbits(
        self, G_gens: dict[int, Permutation], num_threads: int = 1
    ) -> list[tuple[int, ...]]:
        """Groups all monomials into G-orbits and returns a list of full orbits containing integer IDs."""
        visited = bytearray(self.total_monomials)
        full_orbits = []

        g_inv_data = tuple((~gen).data for gen in G_gens.values())
        n_gens = len(g_inv_data)

        if num_threads > 1:
            from concurrent.futures import ProcessPoolExecutor

            chunk_size = math.ceil(self.total_monomials / (num_threads * 4))
            chunks = []
            for i in range(0, self.total_monomials, chunk_size):
                chunks.append(
                    (i, min(i + chunk_size, self.total_monomials), g_inv_data, self)
                )

            edges = array.array("i")
            with ProcessPoolExecutor(max_workers=num_threads) as executor:
                for res_bytes in executor.map(_compute_orbit_chunk, chunks):
                    edges.frombytes(res_bytes)
        else:
            chunk = (0, self.total_monomials, g_inv_data, self)
            res_bytes = _compute_orbit_chunk(chunk)
            edges = array.array("i")
            edges.frombytes(res_bytes)

        for m in range(self.total_monomials):
            if visited[m]:
                continue

            orbit = [m]
            queue = deque([m])
            visited[m] = 1
            while queue:
                curr = queue.popleft()
                base_idx = curr * n_gens
                for i in range(n_gens):
                    nxt = edges[base_idx + i]
                    if not visited[nxt]:
                        visited[nxt] = 1
                        orbit.append(nxt)
                        queue.append(nxt)

            full_orbits.append(tuple(orbit))

        return full_orbits


def _extract_blocks_worker(orbits_chunk, abstract, G_gens, space):
    blocks = []
    for orbit in orbits_chunk:
        blocks.extend(_extract_blocks_for_orbit_dfs(abstract, G_gens, orbit, space))
    return blocks


def build_monomial_sab(
    abstract: BaumClausenPaths,
    G_gens: Mapping[int, Permutation],
    orbits: list[tuple[int, ...]],
    space: "MonomialSpace",
    num_threads: int = 1,
) -> SABTransform:
    blocks = []
    if num_threads > 1:
        from concurrent.futures import ProcessPoolExecutor
        import functools

        import math

        func = functools.partial(
            _extract_blocks_worker,
            abstract=abstract,
            G_gens=G_gens,
            space=space,
        )

        chunksize = max(1, math.ceil(len(orbits) / (num_threads * 4)))
        chunks = [orbits[i : i + chunksize] for i in range(0, len(orbits), chunksize)]

        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            for chunk_blocks in executor.map(func, chunks):
                blocks.extend(chunk_blocks)
    else:
        for orbit in orbits:
            orbit_blocks = _extract_blocks_for_orbit_dfs(abstract, G_gens, orbit, space)
            blocks.extend(orbit_blocks)

    blocks.sort(key=lambda b: b.rep_id)

    merged_blocks = []
    current_rep = None
    merged_data = None
    N_total = space.total_monomials

    for b in blocks:
        if b.rep_id != current_rep:
            if merged_data is not None:
                for k in ["orbit_reps", "orbit_reps_flat", "orbit_sizes"]:
                    merged_data[k] = tuple(merged_data[k])
                merged_blocks.append(SABBlock(**merged_data))

            current_rep = b.rep_id

            # Initialize dense mappings for the merged block
            col_j = [-1] * N_total
            col_l = [0] * N_total
            for local_idx, y in enumerate(b.orbit):
                col_j[y] = b.col_to_j[local_idx]
                col_l[y] = b.col_to_l[local_idx]

            merged_data = {
                "rep_id": b.rep_id,
                "dim": b.dim,
                "e": b.e,
                "orbit_reps": list(b.orbit_reps),
                "orbit_reps_flat": list(b.orbit_reps_flat),
                "orbit_sizes": list(b.orbit_sizes),
                "orbit": (),  # No longer needed after merging into dense arrays
                "col_to_j": col_j,
                "col_to_l": col_l,
            }
        else:
            dim_offset = merged_data["dim"]
            merged_data["dim"] += b.dim
            merged_data["orbit_reps"].extend(b.orbit_reps)
            merged_data["orbit_reps_flat"].extend(b.orbit_reps_flat)
            merged_data["orbit_sizes"].extend(b.orbit_sizes)

            for local_idx, y in enumerate(b.orbit):
                if b.col_to_j[local_idx] != -1:
                    merged_data["col_to_j"][y] = b.col_to_j[local_idx] + dim_offset
                    merged_data["col_to_l"][y] = b.col_to_l[local_idx]

    if merged_data is not None:
        for k in ["orbit_reps", "orbit_reps_flat", "orbit_sizes"]:
            merged_data[k] = tuple(merged_data[k])
        merged_blocks.append(SABBlock(**merged_data))

    return SABTransform(blocks=tuple(merged_blocks), N=N_total)


def _extract_blocks_for_orbit_dfs(
    abstract: BaumClausenPaths,
    G_gens: Mapping[int, Permutation],
    orbit: tuple[int, ...],
    space: "MonomialSpace",
) -> typing.Iterator[SABBlock]:
    e = abstract.e
    g_inv_data = {gen_id: (~gen).data for gen_id, gen in G_gens.items()}
    point_to_id = {v: i for i, v in enumerate(orbit)}
    num_points = len(orbit)
    k_levels = len(next(iter(abstract.paths.values()))) + 1

    def dfs(
        level: int,
        rep_ids: list[int],
        reps: list[int],
        col_to_j: list[int],
        col_to_l: list[int],
    ) -> typing.Iterator[SABBlock]:
        if level == k_levels:
            orbit_reps_mapped = tuple(space.unrank_tuple(orbit[r]) for r in reps)
            orbit_reps_flat = tuple(orbit[r] for r in reps)

            counts = [0] * len(reps)
            for j in col_to_j:
                if j != -1:
                    counts[j] += 1
            orbit_sizes = tuple(counts)

            for rep_id in rep_ids:
                if len(reps) > 0:
                    yield SABBlock(
                        rep_id=rep_id,
                        dim=len(reps),
                        e=e,
                        orbit_reps=orbit_reps_mapped,
                        orbit_reps_flat=orbit_reps_flat,
                        orbit_sizes=orbit_sizes,
                        orbit=orbit,
                        col_to_j=tuple(col_to_j),
                        col_to_l=tuple(col_to_l),
                    )
            return

        geom_groups = {}
        for rep_id in rep_ids:
            p_data = abstract.paths[rep_id][level - 1]
            geom_groups.setdefault((p_data[0], p_data[1]), []).append(
                (rep_id, p_data[2])
            )

        for (g_i, t_word), rep_lambda_list in geom_groups.items():
            if g_i == 0:
                yield from dfs(
                    level + 1,
                    [r for r, _ in rep_lambda_list],
                    reps,
                    col_to_j,
                    col_to_l,
                )
                continue

            visited_orbits = set()
            cycles = []
            w_of = {}

            for a in range(len(reps)):
                if a in visited_orbits:
                    continue

                curr_val = orbit[reps[a]]
                for gen, exp in reversed(t_word):
                    inv_data = g_inv_data[gen]
                    for _ in range(exp):
                        curr_val = space.apply_gen(curr_val, inv_data)
                w_id = point_to_id[curr_val]
                w_of[a] = w_id
                target_a = col_to_j[w_id]

                if target_a == a:
                    cycles.append([a])
                    visited_orbits.add(a)
                else:
                    cycle = []
                    curr_a = a
                    while curr_a not in visited_orbits:
                        cycle.append(curr_a)
                        visited_orbits.add(curr_a)

                        curr_val = orbit[reps[curr_a]]
                        for gen, exp in reversed(t_word):
                            inv_data = g_inv_data[gen]
                            for _ in range(exp):
                                curr_val = space.apply_gen(curr_val, inv_data)
                        w_id_curr = point_to_id[curr_val]
                        target_curr = col_to_j[w_id_curr]
                        w_of[curr_a] = w_id_curr
                        curr_a = target_curr

                    cycles.append(cycle)

            cycle_data = []
            for cycle in cycles:
                if len(cycle) == 1:
                    a = cycle[0]
                    phi = col_to_l[w_of[a]]
                    cycle_data.append((cycle, phi, None))
                else:
                    phi_list = []
                    phi_sum = 0
                    for r in range(len(cycle)):
                        a = cycle[r]
                        phi = col_to_l[w_of[a]]
                        phi_list.append(phi)
                        phi_sum += phi
                    cycle_data.append((cycle, phi_sum, phi_list))

            lambda_groups = {}
            for rep_id, lambda_i in rep_lambda_list:
                lambda_groups.setdefault(lambda_i, []).append(rep_id)

            lambda_cycles = {}
            for lambda_i in lambda_groups.keys():
                admissible = []
                for cycle_idx, (cycle, phi_sum, _) in enumerate(cycle_data):
                    L = len(cycle)
                    if L == 1:
                        if lambda_i == phi_sum:
                            admissible.append(cycle_idx)
                    else:
                        C_L = (L * lambda_i - phi_sum) % e
                        if C_L == 0:
                            admissible.append(cycle_idx)
                lambda_cycles[lambda_i] = tuple(admissible)

            admissibility_groups = {}
            for lambda_i, rep_ids_grp in lambda_groups.items():
                adm = lambda_cycles[lambda_i]
                admissibility_groups.setdefault(adm, []).append((lambda_i, rep_ids_grp))

            for adm, lam_rep_list in admissibility_groups.items():
                new_reps = []
                state_map = [-1] * len(reps)

                for cycle_idx in adm:
                    cycle, _, _ = cycle_data[cycle_idx]
                    new_idx = len(new_reps)
                    new_reps.append(reps[cycle[0]])
                    for a in cycle:
                        state_map[a] = new_idx

                new_col_to_j = [state_map[j] for j in col_to_j]

                for lambda_i, rep_ids_grp in lam_rep_list:
                    all_len_1 = True
                    for cycle_idx in adm:
                        if len(cycle_data[cycle_idx][0]) > 1:
                            all_len_1 = False
                            break

                    if all_len_1:
                        new_col_to_l = col_to_l
                    else:
                        state_l_offset = [0] * len(reps)
                        for cycle_idx in adm:
                            cycle, _, phi_list = cycle_data[cycle_idx]
                            if len(cycle) > 1:
                                C_r = 0
                                for r, curr_a in enumerate(cycle):
                                    if r > 0:
                                        C_r = (C_r + lambda_i - phi_list[r - 1]) % e
                                    state_l_offset[curr_a] = C_r

                        new_col_to_l = [
                            (col_to_l[y] + state_l_offset[col_to_j[y]]) % e
                            for y in range(num_points)
                        ]

                    yield from dfs(
                        level + 1,
                        rep_ids_grp,
                        new_reps,
                        new_col_to_j,
                        new_col_to_l,
                    )

    yield from dfs(
        1,
        list(abstract.paths.keys()),
        list(range(num_points)),
        list(range(num_points)),
        [0] * num_points,
    )
