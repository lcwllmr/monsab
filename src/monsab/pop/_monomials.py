"""
Monomial spaces and generation.
"""

from monsab.core._transform import SABTransform
from monsab.core._baum_clausen import BaumClausenPaths
from collections.abc import Mapping

import math
from collections import deque
import array

from monsab.core._permutation import Permutation
from monsab import _backend


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


def build_monomial_sab(
    abstract: BaumClausenPaths,
    G_gens: Mapping[int, Permutation],
    orbits: list[tuple[int, ...]],
    space: "MonomialSpace | SquarefreeMonomialSpace",
    num_threads: int = 1,
    coset_reps: dict[int, list[Permutation]] | None = None,
) -> SABTransform:
    """
    Build a Fast SAB Transform for a given space of monomials.

    Args:
        abstract: The abstract Baum-Clausen paths defining the irreducible representations.
        G_gens: The generators of the group $G$, mapping generator IDs to `Permutation` objects.
        orbits: A list of $G$-orbits of monomials, where each orbit is a tuple of monomial IDs.
        space: The monomial space (`MonomialSpace` or `SquarefreeMonomialSpace`) defining the action.
        num_threads: The number of threads to use for parallel execution.
        coset_reps: An optional dictionary mapping representation IDs to a list of coset representatives
            for $G / H_k$. If provided, this precalculates forward and inverse action matrices enabling
            the efficient application of Method 1 (Coset Averaging) for evaluating the Reynolds operator.

    Returns:
        SABTransform: A transform object that can evaluate the SAB basis.
    """
    from monsab.core._transform import SABTransform

    d = space.d
    is_squarefree = isinstance(space, SquarefreeMonomialSpace)

    g_inv_dict = {gen_id: list((~gen).data) for gen_id, gen in G_gens.items()}

    paths_dict = {}
    for rep_id, path_entries in abstract.paths.items():
        new_entries = []
        for g_k, adm, lambda_i in path_entries:
            new_entries.append((g_k, list(adm), lambda_i))
        paths_dict[rep_id] = new_entries

    from collections import deque, Counter
    import cmath

    identity_data = tuple(range(len(g_inv_dict[next(iter(g_inv_dict))])))
    visited = {identity_data}
    queue = deque([identity_data])
    elements = [identity_data]

    gens_data = list(g_inv_dict.values())
    import operator

    gens_ig = [operator.itemgetter(*g_data) for g_data in gens_data]

    while queue:
        curr = queue.popleft()
        for ig in gens_ig:
            nxt = ig(curr)
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
                elements.append(nxt)

    S_G = Counter()
    for y in elements:
        ig_y = operator.itemgetter(*y)
        y2 = ig_y(y)
        S_G[y2] += 1

    fs_indicators = {}
    v_matrices = {}  # to store v for real-type irreps

    for rep_id, paths in abstract.paths.items():
        if abstract.conjugates.get(rep_id, rep_id) != rep_id:
            fs_indicators[rep_id] = 0
            continue

        H_gens = {}
        char_phases_rep = {}
        for g_k, t_word, lambda_i in paths:
            if g_k != 0:
                H_gens[g_k] = g_inv_dict[g_k]
                char_phases_rep[g_k] = lambda_i

        H_visited = {identity_data: 0}
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

        nu2_complex = 0j
        for h, phase in H_visited.items():
            if h in S_G:
                chi_h = cmath.exp(2j * math.pi * phase / abstract.e)
                nu2_complex += S_G[h] * chi_h

        nu2 = round((nu2_complex / len(H_visited)).real)
        fs_indicators[rep_id] = nu2

        # If nu2 == 1, check if chi is complex
        is_complex_chi = False
        for p in H_visited.values():
            if p != 0 and p * 2 != abstract.e:
                is_complex_chi = True
                break

        if nu2 == 1 and is_complex_chi:
            # Find t in G \ H such that t^-1 h t = h^-1
            t_data = None
            for y in elements:
                if y in H_visited:
                    continue
                # check conjugation
                conjugates = True
                for h, phase in H_visited.items():
                    # h y
                    hy = tuple(h[i] for i in y)
                    # y^-1
                    y_inv = [0] * len(y)
                    for i, v in enumerate(y):
                        y_inv[v] = i
                    y_inv_h_y = tuple(y_inv[i] for i in hy)

                    target_phase = (-phase) % abstract.e
                    if (
                        y_inv_h_y not in H_visited
                        or H_visited[y_inv_h_y] != target_phase
                    ):
                        conjugates = False
                        break
                if conjugates:
                    t_data = y
                    break

            if t_data is not None:
                t_mapped = tuple(
                    space.apply_gen(i, t_data) for i in range(space.total_monomials)
                )
                v_matrices[rep_id] = t_mapped

    realize_skip_reps = set()
    matched = set()
    for r1, nu2 in fs_indicators.items():
        if nu2 == 0 and r1 not in matched:
            r2 = abstract.conjugates.get(r1, r1)
            matched.add(r1)
            matched.add(r2)
            realize_skip_reps.add(max(r1, r2))

    import numpy as np

    coset_actions_dict = {}
    coset_actions_inv_dict = {}
    if coset_reps is not None:
        for rep_id, reps in coset_reps.items():
            actions = np.zeros((len(reps), space.total_monomials), dtype=np.uint32)
            actions_inv = np.zeros((len(reps), space.total_monomials), dtype=np.uint32)
            for m, s in enumerate(reps):
                s_data = s.data
                s_inv_data = (~s).data
                for i in range(space.total_monomials):
                    actions[m, i] = space.apply_gen(i, s_data)
                    actions_inv[m, i] = space.apply_gen(i, s_inv_data)
            coset_actions_dict[rep_id] = actions.flatten()
            coset_actions_inv_dict[rep_id] = actions_inv.flatten()

    rust_transform = _backend.build_sab_blocks(
        orbits,
        paths_dict,
        g_inv_dict,
        space.n,
        d,
        abstract.e,
        is_squarefree,
        space.total_monomials,
        fs_indicators,
        v_matrices,
        coset_actions_dict if coset_reps is not None else None,
        coset_actions_inv_dict if coset_reps is not None else None,
    )

    return SABTransform(
        _rust_transform=rust_transform, _realize_skip_reps=realize_skip_reps
    )


def _compute_orbit_chunk_squarefree(
    args: tuple[int, int, tuple[tuple[int, ...], ...], "SquarefreeMonomialSpace"],
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
            c1, c2 = space.unrank_tuple(m)
            v = [c1, c2]
            end_k = min(end, offsets[3])

            for _ in range(end_k - m):
                for inv in g_inv_data:
                    w0, w1 = inv[v[0]], inv[v[1]]
                    if w0 > w1:
                        w0, w1 = w1, w0
                    res[idx] = offsets[2] + b2[w1] + w0
                    idx += 1

                if v[0] < v[1] - 1:
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
                    res[idx] = offsets[3] + b3[w2] + b2[w1] + w0
                    idx += 1

                if v[0] < v[1] - 1:
                    v[0] += 1
                elif v[1] < v[2] - 1:
                    v[1] += 1
                    v[0] = 0
                else:
                    v[2] += 1
                    v[0] = 0
                    v[1] = 1
            m = end_k
            continue

        else:
            # k > 3
            for inv in g_inv_data:
                res[idx] = space.apply_gen(m, inv)
                idx += 1
            m += 1

    return res.tobytes()


class SquarefreeMonomialSpace:
    """
    Handles the abstract space of squarefree monomials of degree up to d in n variables.
    Uses O(1) memory via a mathematically dense Combinatorial Bijection, avoiding all tuple objects.
    """

    def __init__(self, n: int, d: int):
        self.n = n
        self.d = min(d, n)

        # Precompute offsets for each degree k
        self.offsets = [0] * (self.d + 2)
        total = 0
        for k in range(self.d + 1):
            self.offsets[k] = total
            total += math.comb(n, k)
        self.offsets[self.d + 1] = total
        self.total_monomials = total

        # Precompute binomial lookup tables for extremely fast unranking
        self.binom_2 = tuple(math.comb(i, 2) for i in range(n + 1))
        self.binom_3 = tuple(math.comb(i, 3) for i in range(n + 1))

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
            return (c1, c2)
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
            return (c1, c2, c3)
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
            return tuple(c[k - 1 - j] for j in range(k))

    def rank_tuple(self, tup: tuple[int, ...]) -> int:
        """Convert a sparse variable tuple back to an integer ID."""
        k = len(tup)
        if k == 0:
            return 0
        rank = sum(math.comb(tup[i], i + 1) for i in range(k))
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

            v0, v1 = inv_data[c1], inv_data[c2]
            if v0 > v1:
                v0, v1 = v1, v0
            return self.offsets[2] + self.binom_2[v1] + v0

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

            v0, v1, v2 = inv_data[c1], inv_data[c2], inv_data[c3]
            if v0 > v1:
                v0, v1 = v1, v0
            if v1 > v2:
                v1, v2 = v2, v1
            if v0 > v1:
                v0, v1 = v1, v0
            return self.offsets[3] + self.binom_3[v2] + self.binom_2[v1] + v0

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
                for res_bytes in executor.map(_compute_orbit_chunk_squarefree, chunks):
                    edges.frombytes(res_bytes)
        else:
            chunk = (0, self.total_monomials, g_inv_data, self)
            res_bytes = _compute_orbit_chunk_squarefree(chunk)
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
