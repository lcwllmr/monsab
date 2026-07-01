"""
Monomial spaces and generation.
"""

from monsab.core._transform import SABTransform
from monsab.core._baum_clausen import BaumClausenPaths
from collections.abc import Mapping

import math
from collections import deque
import array

from typing import TYPE_CHECKING
from monsab.core._permutation import Permutation
from monsab import _backend

if TYPE_CHECKING:
    from monsab.core import PcGroup


def _compute_orbit_chunk(
    args: tuple[int, int, tuple[tuple[int, ...], ...], "MonomialSpace", int],
) -> bytes:
    start, end, g_inv_data, space, d = args
    n_gens = len(g_inv_data)
    res = array.array("i", [0] * ((end - start) * n_gens))
    idx = 0

    offsets = space.offsets
    b2 = space.binom_2
    b3 = space.binom_3

    m = start
    while m < end:
        if d >= 0 and m < offsets[1]:
            # k = 0
            for _ in g_inv_data:
                res[idx] = 0
                idx += 1
            m += 1
            continue

        elif d >= 1 and m < offsets[2]:
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

        elif d >= 2 and m < offsets[3]:
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

        elif d >= 3 and m < offsets[4]:
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

    def __init__(self, n: int):
        self.n = n
        self.offsets = [
            0,
            1,
        ]  # At least populated up to d=0 (k=0 gives offset 0, k=1 gives offset 1)

        self.binom_2 = tuple(math.comb(i, 2) for i in range(n + 3))
        self.binom_3 = tuple(math.comb(i, 3) for i in range(n + 3))

    def _ensure_d(self, d: int) -> None:
        if len(self.offsets) <= d + 1:
            total = self.offsets[-1]
            for k in range(len(self.offsets) - 1, d + 1):
                total += math.comb(self.n + k - 1, k)
                self.offsets.append(total)

    def total_monomials(self, d: int) -> int:
        self._ensure_d(d)
        return self.offsets[d + 1]

    def unrank_tuple(self, m_id: int) -> tuple[int, ...]:
        """Convert an integer ID back to the sparse variable tuple."""
        # Lazily ensure offsets array is large enough for the requested m_id
        while m_id >= self.offsets[-1]:
            self._ensure_d(len(self.offsets))

        k = 0
        for i in range(len(self.offsets) - 1):
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
        self._ensure_d(k)
        rank = sum(math.comb(tup[i] + i, i + 1) for i in range(k))
        return self.offsets[k] + rank

    def apply_gen(self, m_id: int, inv_data: tuple[int, ...]) -> int:
        """Apply a permutation strictly on the packed integer representation."""
        if len(self.offsets) > 2 and m_id < self.offsets[2]:
            if m_id < self.offsets[1]:
                return 0
            # k = 1
            v = m_id - self.offsets[1]
            return self.offsets[1] + inv_data[v]

        elif len(self.offsets) > 3 and m_id < self.offsets[3]:
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

        elif len(self.offsets) > 4 and m_id < self.offsets[4]:
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

    def get_orbits(
        self, G_gens: dict[int, Permutation], d: int
    ) -> list[tuple[int, ...]]:
        """Groups all monomials into G-orbits and returns one representative per orbit."""
        self._ensure_d(d)
        total_monomials = self.offsets[d + 1]
        visited = bytearray(total_monomials)
        orbit_reps = []

        g_inv_data = [(~gen).data for gen in G_gens.values()]

        for m in range(total_monomials):
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

    def get_orbit_reps_and_sizes(self, group: "PcGroup", d: int) -> dict[int, int]:
        """
        Uses the fast PyOrbitLifter to evaluate the canonical orbit representative
        for every monomial up to degree d and returns a dictionary mapping rep_id to orbit_size.
        """
        from monsab._backend import PyOrbitLifter

        if d > 4:
            raise NotImplementedError("Fast orbit lifter not implemented for D > 4")

        self._ensure_d(d)
        lifters = {k: PyOrbitLifter(group, k, False) for k in range(1, d + 1)}
        rep_counts = {}
        for m in range(self.total_monomials(d)):
            tup = self.unrank_tuple(m)
            if not tup:
                canonical = ()
            else:
                canonical = lifters[len(tup)].canonicalize(list(tup))
            rep_rank = self.rank_tuple(tuple(canonical))
            rep_counts[rep_rank] = rep_counts.get(rep_rank, 0) + 1
        return rep_counts

    def get_full_orbits(
        self, G_gens: dict[int, Permutation], d: int, num_threads: int = 1
    ) -> list[tuple[int, ...]]:
        """Groups all monomials into G-orbits and returns a list of full orbits containing integer IDs."""
        self._ensure_d(d)
        total_monomials = self.offsets[d + 1]
        visited = bytearray(total_monomials)
        full_orbits = []

        g_inv_data = tuple((~gen).data for gen in G_gens.values())
        n_gens = len(g_inv_data)

        if num_threads > 1:
            from concurrent.futures import ProcessPoolExecutor

            chunk_size = math.ceil(total_monomials / (num_threads * 4))
            chunks = []
            for i in range(0, total_monomials, chunk_size):
                chunks.append(
                    (i, min(i + chunk_size, total_monomials), g_inv_data, self, d)
                )

            edges = array.array("i")
            with ProcessPoolExecutor(max_workers=num_threads) as executor:
                for res_bytes in executor.map(_compute_orbit_chunk, chunks):
                    edges.frombytes(res_bytes)
        else:
            chunk = (0, total_monomials, g_inv_data, self, d)
            res_bytes = _compute_orbit_chunk(chunk)
            edges = array.array("i")
            edges.frombytes(res_bytes)

        for m in range(total_monomials):
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
    d: int,
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

    is_squarefree = isinstance(space, SquarefreeMonomialSpace)

    g_inv_dict = {gen_id: list((~gen).data) for gen_id, gen in G_gens.items()}

    paths_dict = {}
    for rep_id, path_entries in abstract.paths.items():
        new_entries = []
        for g_k, adm, lambda_i in path_entries:
            new_entries.append((g_k, list(adm), lambda_i))
        paths_dict[rep_id] = new_entries

    from collections import deque

    identity_data = tuple(range(len(g_inv_dict[next(iter(g_inv_dict))])))
    import operator

    fs_indicators = {}
    v_matrices = {}  # to store v for real-type irreps

    H_visited_all = {}
    is_complex_chi_all = {}
    H_gens_all = {}
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

        H_gens = {}
        H_gens_fwd = {}
        char_phases_rep = {}
        for g_k, t_word, lambda_i in paths:
            if g_k != 0:
                H_gens[g_k] = g_inv_dict[g_k]

                inv_data = g_inv_dict[g_k]
                fwd_data = [0] * len(inv_data)
                for i, v in enumerate(inv_data):
                    fwd_data[v] = i
                H_gens_fwd[g_k] = tuple(fwd_data)

                char_phases_rep[g_k] = lambda_i

        H_gens_all[rep_id] = (H_gens_fwd, char_phases_rep)

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

        H_visited_all[rep_id] = H_visited

        is_complex_chi = False
        for p in H_visited.values():
            if p != 0 and p * 2 != abstract.e:
                is_complex_chi = True
                break
        is_complex_chi_all[rep_id] = is_complex_chi

    g_gens_data = [list(g.data) for g in G_gens.values()]
    rust_fs, v_matrices = _backend.compute_fs_and_t_data(
        g_gens_data,
        space.n,
        abstract.e,
        H_visited_all,
        {rep_id: H_gens_all[rep_id][0] for rep_id in H_gens_all},
        {rep_id: H_gens_all[rep_id][1] for rep_id in H_gens_all},
    )
    fs_indicators.update(rust_fs)

    # Optional: Python creates the v_matrices dict missing keys where t_data is None.
    # We should unpack v_matrices which might have None values.
    v_matrices = {k: v for k, v in v_matrices.items() if v is not None}

    for rep_id in H_visited_all.keys():
        if rep_id not in fs_indicators:
            # Fallback (handled in Rust, but just in case)
            pass

    realize_skip_reps = set()
    matched = set()
    for r1, nu2 in fs_indicators.items():
        if nu2 == 0 and r1 not in matched:
            r2 = abstract.conjugates.get(r1, r1)
            matched.add(r1)
            matched.add(r2)
            realize_skip_reps.add(max(r1, r2))

    coset_reps_dict = {}
    coset_reps_inv_dict = {}
    if coset_reps is not None:
        for rep_id, reps in coset_reps.items():
            reps_data = []
            reps_inv_data = []
            for m, s in enumerate(reps):
                reps_data.append(list(s.data))
                reps_inv_data.append(list((~s).data))
            coset_reps_dict[rep_id] = reps_data
            coset_reps_inv_dict[rep_id] = reps_inv_data

    rust_transform = _backend.build_sab_blocks(
        orbits,
        paths_dict,
        g_inv_dict,
        space.n,
        d,
        abstract.e,
        is_squarefree,
        space.total_monomials(d),
        fs_indicators,
        v_matrices,
        coset_reps_dict if coset_reps is not None else None,
        coset_reps_inv_dict if coset_reps is not None else None,
    )

    return SABTransform(
        _rust_transform=rust_transform, _realize_skip_reps=realize_skip_reps
    )


def _compute_orbit_chunk_squarefree(
    args: tuple[int, int, tuple[tuple[int, ...], ...], "SquarefreeMonomialSpace", int],
) -> bytes:
    start, end, g_inv_data, space, d = args
    n_gens = len(g_inv_data)
    res = array.array("i", [0] * ((end - start) * n_gens))
    idx = 0

    offsets = space.offsets
    b2 = space.binom_2
    b3 = space.binom_3

    m = start
    while m < end:
        if d >= 0 and m < offsets[1]:
            # k = 0
            for _ in g_inv_data:
                res[idx] = 0
                idx += 1
            m += 1
            continue

        elif d >= 1 and m < offsets[2]:
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

        elif d >= 2 and m < offsets[3]:
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

        elif d >= 3 and m < offsets[4]:
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

    def __init__(self, n: int):
        self.n = n
        self.offsets = [0, 1]

        self.binom_2 = tuple(math.comb(i, 2) for i in range(n + 3))
        self.binom_3 = tuple(math.comb(i, 3) for i in range(n + 3))

    def _ensure_d(self, d: int) -> None:
        if len(self.offsets) <= d + 1:
            total = self.offsets[-1]
            for k in range(len(self.offsets) - 1, d + 1):
                total += math.comb(self.n, k)
                self.offsets.append(total)

    def total_monomials(self, d: int) -> int:
        self._ensure_d(d)
        return self.offsets[d + 1]

    def unrank_tuple(self, m_id: int) -> tuple[int, ...]:
        """Convert an integer ID back to the sparse variable tuple."""
        # Lazily ensure offsets array is large enough for the requested m_id
        while m_id >= self.offsets[-1]:
            self._ensure_d(len(self.offsets))

        k = 0
        for i in range(len(self.offsets) - 1):
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
        self._ensure_d(k)
        rank = sum(math.comb(tup[i], i + 1) for i in range(k))
        return self.offsets[k] + rank

    def apply_gen(self, m_id: int, inv_data: tuple[int, ...]) -> int:
        """Apply a permutation strictly on the packed integer representation."""
        if len(self.offsets) > 2 and m_id < self.offsets[2]:
            if m_id < self.offsets[1]:
                return 0
            # k = 1
            v = m_id - self.offsets[1]
            return self.offsets[1] + inv_data[v]

        elif len(self.offsets) > 3 and m_id < self.offsets[3]:
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

        elif len(self.offsets) > 4 and m_id < self.offsets[4]:
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

    def get_orbits(
        self, G_gens: dict[int, Permutation], d: int
    ) -> list[tuple[int, ...]]:
        """Groups all monomials into G-orbits and returns one representative per orbit."""
        self._ensure_d(d)
        total_monomials = self.offsets[d + 1]
        visited = bytearray(total_monomials)
        orbit_reps = []

        g_inv_data = [(~gen).data for gen in G_gens.values()]

        for m in range(total_monomials):
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

    def get_orbit_reps_and_sizes(self, group: "PcGroup", d: int) -> dict[int, int]:
        """
        Uses the fast PyOrbitLifter to evaluate the canonical orbit representative
        for every squarefree monomial and returns a dictionary mapping rep_id to orbit_size.
        """
        from monsab._backend import PyOrbitLifter

        if d > 4:
            raise NotImplementedError("Fast orbit lifter not implemented for D > 4")

        self._ensure_d(d)
        lifters = {k: PyOrbitLifter(group, k, True) for k in range(1, d + 1)}
        rep_counts = {}
        for m in range(self.total_monomials(d)):
            tup = self.unrank_tuple(m)
            if not tup:
                canonical = ()
            else:
                canonical = lifters[len(tup)].canonicalize(list(tup))
            rep_rank = self.rank_tuple(tuple(canonical))
            rep_counts[rep_rank] = rep_counts.get(rep_rank, 0) + 1
        return rep_counts

    def get_full_orbits(
        self, G_gens: dict[int, Permutation], d: int, num_threads: int = 1
    ) -> list[tuple[int, ...]]:
        """Groups all monomials into G-orbits and returns a list of full orbits containing integer IDs."""
        self._ensure_d(d)
        total_monomials = self.offsets[d + 1]
        visited = bytearray(total_monomials)
        full_orbits = []

        g_inv_data = tuple((~gen).data for gen in G_gens.values())
        n_gens = len(g_inv_data)

        if num_threads > 1:
            from concurrent.futures import ProcessPoolExecutor

            chunk_size = math.ceil(total_monomials / (num_threads * 4))
            chunks = []
            for i in range(0, total_monomials, chunk_size):
                chunks.append(
                    (i, min(i + chunk_size, total_monomials), g_inv_data, self, d)
                )

            edges = array.array("i")
            with ProcessPoolExecutor(max_workers=num_threads) as executor:
                for res_bytes in executor.map(_compute_orbit_chunk_squarefree, chunks):
                    edges.frombytes(res_bytes)
        else:
            chunk = (0, total_monomials, g_inv_data, self, d)
            res_bytes = _compute_orbit_chunk_squarefree(chunk)
            edges = array.array("i")
            edges.frombytes(res_bytes)

        for m in range(total_monomials):
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
