"""
Baum-Clausen algorithm for representations.
"""

import typing
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from monsab.util import extract_roots

from ._matrix import MonomialMatrix
from ._polycyclic import PolycyclicPresentation, Word


@dataclass(frozen=True, slots=True)
class Representation:
    id: int
    dim: int
    matrices: Mapping[int, MonomialMatrix]
    is_induced: bool = False
    children: tuple[int, ...] | None = None
    cluster: tuple[int, ...] | None = None
    conjugate_ptr: int | None = None

    def evaluate(self, word: Word) -> MonomialMatrix:
        """Evaluate a word in the generators to produce a monomial matrix."""
        result = MonomialMatrix.identity(dim=self.dim, e=self.matrices[0].e)
        for gen_index, exponent in word:
            result = result @ (self.matrices[gen_index] ** exponent)
        return result


@dataclass(frozen=True, slots=True)
class BaumClausenStage:
    """
    Represents the state of the Baum-Clausen algorithm at a specific level (group G_i).
    """

    level: int
    e: int
    representations: tuple[Representation, ...]
    X_dict: Mapping[int, Mapping[int, MonomialMatrix]]
    tau_dict: Mapping[int, Mapping[int, int]]
    power_relations: Mapping[int, Word]
    conjugation_relations: Mapping[tuple[int, int], Word]
    conjugation_exponents: Mapping[tuple[int, int], int]
    higher_generators: tuple[int, ...]

    @classmethod
    def trivial(
        cls,
        e: int,
        presentation: PolycyclicPresentation,
    ) -> typing.Self:
        """Produce the level 0 object for the trivial subgroup."""
        trivial_rep = Representation(
            id=0,
            dim=1,
            matrices=MappingProxyType({0: MonomialMatrix.identity(dim=1, e=e)}),
            is_induced=False,
            children=None,
            cluster=None,
            conjugate_ptr=0,
        )

        # 1-based indexing for generators
        n_gens = presentation.number_of_generators
        generators = list(range(1, n_gens + 1))

        X_dict = {
            g_j: MappingProxyType({0: MonomialMatrix.identity(dim=1, e=e)})
            for g_j in generators
        }
        tau_dict = {g_j: MappingProxyType({0: 0}) for g_j in generators}

        power_relations = {
            k + 1: tuple((g + 1, exp) for g, exp in word)
            for k, word in presentation.power_tails.items()
        }
        conjugation_relations = {
            (j + 1, k + 1): tuple((g + 1, exp) for g, exp in word)
            for (j, k), word in presentation.conjugation_tails.items()
        }
        conjugation_exponents = {
            (j + 1, k + 1): exp
            for (j, k), exp in presentation.conjugation_exponents.items()
        }

        return cls(
            level=0,
            e=e,
            representations=(trivial_rep,),
            X_dict=MappingProxyType(X_dict),
            tau_dict=MappingProxyType(tau_dict),
            power_relations=power_relations,
            conjugation_relations=conjugation_relations,
            conjugation_exponents=conjugation_exponents,
            higher_generators=tuple(sorted(generators)),
        )

    @classmethod
    def next_level(cls, previous: typing.Self, g_i: int, p: int) -> typing.Self:
        level = previous.level + 1
        e = previous.e
        power_relations = previous.power_relations
        conjugation_relations = previous.conjugation_relations
        conjugation_exponents = previous.conjugation_exponents
        higher_generators = tuple(g for g in previous.higher_generators if g != g_i)

        id_to_rep = {rep.id: rep for rep in previous.representations}
        visited: set[int] = set()
        new_representations: list[Representation] = []
        X_dict: dict[int, dict[int, MonomialMatrix]] = {
            g_j: {} for g_j in higher_generators
        }
        tau_dict: dict[int, dict[int, int]] = {g_j: {} for g_j in higher_generators}

        next_id = (max(id_to_rep) + 1) if id_to_rep else 0

        case1_clusters: list[tuple[Representation, list[Representation]]] = []
        case2_orbits: list[tuple[Representation, list[Representation]]] = []

        # Phase 1: Computation of representations for G_i
        for F in previous.representations:
            if F.id in visited:
                continue

            if previous.tau_dict[g_i][F.id] == F.id:
                # Case 1: pi_i F = F
                visited.add(F.id)
                X_iF = previous.X_dict[g_i][F.id]

                F_gp = F.evaluate(power_relations.get(g_i, ()))
                X_iF_p = X_iF**p
                target_val = F_gp.vals[0] if F_gp.vals else 0
                current_val = X_iF_p.vals[0] if X_iF_p.vals else 0
                c_pow_p = (target_val - current_val) % e
                roots = extract_roots(c_pow_p, p, e)

                extensions: list[Representation] = []
                extension_ids = tuple(range(next_id, next_id + len(roots)))

                for c_k in roots:
                    matrices = {
                        gen: MonomialMatrix(
                            perm=matrix.perm, vals=matrix.vals, e=matrix.e
                        )
                        for gen, matrix in F.matrices.items()
                        if gen != 0
                    }
                    matrices[0] = MonomialMatrix.identity(dim=F.dim, e=e)
                    matrices[g_i] = MonomialMatrix(
                        perm=X_iF.perm,
                        vals=tuple((value + c_k) % e for value in X_iF.vals),
                        e=e,
                    )
                    D_k = Representation(
                        id=next_id,
                        dim=F.dim,
                        matrices=MappingProxyType(matrices),
                        is_induced=False,
                        children=(F.id,),
                        cluster=extension_ids,
                    )
                    next_id += 1
                    new_representations.append(D_k)
                    extensions.append(D_k)

                case1_clusters.append((F, extensions))

            else:
                # Case 2: pi_i F != F
                orbit = [F]
                visited.add(F.id)
                cursor = F
                for _ in range(p - 1):
                    next_orbit_id = previous.tau_dict[g_i][cursor.id]
                    cursor = id_to_rep[next_orbit_id]
                    orbit.append(cursor)
                    visited.add(cursor.id)

                dim = F.dim
                full_dim = p * dim
                matrices: dict[int, MonomialMatrix] = {}

                lower_generators = sorted(
                    gen
                    for gen in orbit[0].matrices
                    if isinstance(gen, int) and gen < g_i and gen != 0
                )
                matrices[0] = MonomialMatrix.identity(dim=full_dim, e=e)
                for gen in lower_generators:
                    matrices[gen] = sum(
                        [rep.matrices[gen] for rep in orbit],
                        start=MonomialMatrix.identity(dim=0, e=e),
                    )

                X_blocks = [MonomialMatrix.identity(dim=dim, e=e)]
                for k in range(1, p):
                    X_prev = X_blocks[k - 1]
                    X_k = previous.X_dict[g_i][orbit[k - 1].id] @ X_prev
                    X_blocks.append(X_k)

                gi_perm = [0] * full_dim
                gi_vals = [0] * full_dim
                F0_gp = orbit[0].evaluate(power_relations.get(g_i, ()))

                for k in range(p):
                    if k < p - 1:
                        block_mat = X_blocks[k + 1] @ X_blocks[k].inverse()
                        target_block = k + 1
                    else:
                        block_mat = X_blocks[0] @ F0_gp @ X_blocks[p - 1].inverse()
                        target_block = 0

                    for r in range(dim):
                        global_row = k * dim + r
                        global_col = target_block * dim + block_mat.perm[r]
                        gi_perm[global_row] = global_col
                        gi_vals[global_row] = block_mat.vals[r]

                matrices[g_i] = MonomialMatrix(
                    perm=tuple(gi_perm), vals=tuple(gi_vals), e=e
                )
                D = Representation(
                    id=next_id,
                    dim=full_dim,
                    matrices=MappingProxyType(matrices),
                    is_induced=True,
                    children=tuple(rep.id for rep in orbit),
                    cluster=None,
                )
                next_id += 1
                new_representations.append(D)
                case2_orbits.append((D, orbit))

        # Phase 3: Wire up conjugate pointers
        # Build lookup from base id to its extensions
        base_to_extensions = {base.id: exts for base, exts in case1_clusters}
        # Build lookup from orbit-representative id to induced representation id
        rep_to_induced = {rep.id: D.id for D, orbit in case2_orbits for rep in orbit}

        conj_map = {}

        # Rule 3: Extension
        for base, exts in case1_clusters:
            conj_base_id = base.conjugate_ptr
            conj_exts = base_to_extensions[conj_base_id]
            p_exts = len(exts)
            for a, D_a in enumerate(exts):
                conj_a = (-a) % p_exts
                conj_map[D_a.id] = conj_exts[conj_a].id

        # Rule 2: Induction
        for D, orbit in case2_orbits:
            F = orbit[0]
            conj_F_id = F.conjugate_ptr
            conj_D_id = rep_to_induced[conj_F_id]
            conj_map[D.id] = conj_D_id

        # Re-build new representations with conjugate_ptr
        import dataclasses

        new_representations_wired = tuple(
            dataclasses.replace(rep, conjugate_ptr=conj_map[rep.id])
            for rep in new_representations
        )

        # Phase 4: Computation of tau_j and X_j (intertwiners Y_jD)
        base_to_extensions = {base.id: exts for base, exts in case1_clusters}
        rep_to_induced = {rep.id: D for D, orbit in case2_orbits for rep in orbit}

        for g_j in higher_generators:
            # Case 1
            for base_rep, extensions in case1_clusters:
                pi_j_F_id = previous.tau_dict[g_j][base_rep.id]
                target_extensions = base_to_extensions[pi_j_F_id]
                X_jF = previous.X_dict[g_j][base_rep.id]

                # Extract exponent directly from the presentation!
                a_j = conjugation_exponents.get((g_i, g_j), 1)

                word = conjugation_relations.get((g_i, g_j), ())
                D0 = extensions[0]
                D0_gi_gj = D0.evaluate(word)
                M = X_jF @ D0_gi_gj @ X_jF.inverse()

                target_delta_0 = target_extensions[0].matrices[g_i]
                X_i_Phi = previous.X_dict[g_i][pi_j_F_id]

                # Correct derived formula: (X * c_0)^a_j * M = X * c_l
                # => c_l = M + a_j * c_0 + (a_j - 1) * X
                val_M = M.vals[0] if M.vals else 0
                val_t0 = target_delta_0.vals[0] if target_delta_0.vals else 0
                val_X = X_i_Phi.vals[0] if X_i_Phi.vals else 0

                c_l = (val_M + a_j * val_t0 + (a_j - 1) * val_X) % e

                # Find l such that roots[l] corresponds to c_l
                l_idx = None
                for idx, E in enumerate(target_extensions):
                    val_E = E.matrices[g_i].vals[0] if E.matrices[g_i].vals else 0
                    if val_E == c_l:
                        l_idx = idx
                        break

                if l_idx is None:
                    raise ValueError(
                        "Failed to match extension shift in Phase 2 Case 1."
                    )

                for k, D_k in enumerate(extensions):
                    X_dict[g_j][D_k.id] = X_jF
                    target_idx = (l_idx + k * a_j) % p
                    tau_dict[g_j][D_k.id] = target_extensions[target_idx].id

            # Case 2
            for D, orbit in case2_orbits:
                dim = orbit[0].dim
                pi_j_F0_id = previous.tau_dict[g_j][orbit[0].id]
                target_D = rep_to_induced[pi_j_F0_id]
                tau_dict[g_j][D.id] = target_D.id

                word = conjugation_relations.get((g_i, g_j), ())
                D_gi_gj = D.evaluate(word)
                D_gi_gj_inv_perm = tuple(
                    D_gi_gj.perm.index(i) for i in range(D_gi_gj.dim)
                )

                target_D_gi = target_D.matrices[g_i]
                target_D_gi_inv_perm = tuple(
                    target_D_gi.perm.index(i) for i in range(target_D_gi.dim)
                )

                target_orbit_ids = target_D.children
                assert target_orbit_ids is not None

                c = {0: 0}
                adj: dict[int, list[tuple[int, int]]] = {k: [] for k in range(p)}

                Y_tmp_perm = [0] * D.dim
                Y_tmp_vals = [0] * D.dim

                for k, F_k in enumerate(orbit):
                    pi_j_Fk_id = previous.tau_dict[g_j][F_k.id]
                    sigma_k = target_orbit_ids.index(pi_j_Fk_id)
                    X_k = previous.X_dict[g_j][F_k.id]

                    for r in range(dim):
                        global_row = sigma_k * dim + r
                        global_col = k * dim + X_k.perm[r]
                        Y_tmp_perm[global_row] = global_col
                        Y_tmp_vals[global_row] = X_k.vals[r]

                    # Extract A_k
                    A_k_perm = [0] * dim
                    A_k_vals = [0] * dim
                    pi_k = None
                    for i in range(dim):
                        g_col = k * dim + i
                        g_row = D_gi_gj_inv_perm[g_col]
                        if pi_k is None:
                            pi_k = g_row // dim
                        r = g_row % dim
                        A_k_perm[r] = i
                        A_k_vals[r] = D_gi_gj.vals[g_row]
                    A_k = MonomialMatrix(
                        perm=tuple(A_k_perm), vals=tuple(A_k_vals), e=e
                    )

                    # Extract B_sk
                    B_sk_perm = [0] * dim
                    B_sk_vals = [0] * dim
                    for i in range(dim):
                        g_col = sigma_k * dim + i
                        g_row = target_D_gi_inv_perm[g_col]
                        r = g_row % dim
                        B_sk_perm[r] = i
                        B_sk_vals[r] = target_D_gi.vals[g_row]
                    B_sk = MonomialMatrix(
                        perm=tuple(B_sk_perm), vals=tuple(B_sk_vals), e=e
                    )

                    X_pi_k = previous.X_dict[g_j][orbit[pi_k].id]
                    M = X_pi_k @ A_k @ X_k.inverse()

                    weight = (B_sk.vals[0] - M.vals[0]) % e
                    assert pi_k is not None
                    adj[pi_k].append((k, weight))
                    adj[k].append((pi_k, (-weight) % e))

                for start_node in range(p):
                    if start_node not in c:
                        c[start_node] = 0
                        queue = deque([start_node])
                        while queue:
                            curr = queue.popleft()
                            for nxt, weight in adj[curr]:
                                if nxt not in c:
                                    c[nxt] = (c[curr] - weight) % e
                                    queue.append(nxt)

                Y_perm = Y_tmp_perm
                Y_vals = [0] * D.dim
                for r in range(D.dim):
                    k = Y_perm[r] // dim
                    Y_vals[r] = (Y_tmp_vals[r] + c[k]) % e

                X_dict[g_j][D.id] = MonomialMatrix(
                    perm=tuple(Y_perm), vals=tuple(Y_vals), e=e
                )

        frozen_tau_dict = MappingProxyType(
            {g_j: MappingProxyType(d) for g_j, d in tau_dict.items()}
        )

        return cls(
            level=level,
            e=e,
            representations=new_representations_wired,
            X_dict=MappingProxyType(
                {k: MappingProxyType(v) for k, v in X_dict.items()}
            ),
            tau_dict=frozen_tau_dict,
            power_relations=power_relations,
            conjugation_relations=conjugation_relations,
            conjugation_exponents=conjugation_exponents,
            higher_generators=higher_generators,
        )


@dataclass(frozen=True, slots=True)
class BaumClausenPaths:
    # A mapping from final_rep_id to a list of (g_i, t_word, lambda_i) for each k
    # t_word is a list of (generator, exponent)
    paths: Mapping[int, tuple[tuple[int, tuple[tuple[int, int], ...], int], ...]]
    e: int
    conjugates: Mapping[int, int]

    @classmethod
    def from_baum_clausen(
        cls, stages: tuple[BaumClausenStage, ...]
    ) -> "BaumClausenPaths":
        if not stages:
            raise ValueError("Must provide at least one stage.")

        final_stage = stages[-1]
        e = final_stage.e

        rep_maps = [{rep.id: rep for rep in stage.representations} for stage in stages]
        paths = {}

        # Precompute g_i for each level transition
        g_i_seq = [0] * len(stages)
        for k in range(1, len(stages)):
            prev_stage = stages[k - 1]
            g_i_set = set(prev_stage.higher_generators) - set(
                stages[k].higher_generators
            )
            g_i_seq[k] = next(iter(g_i_set))

        for D in final_stage.representations:
            lineage = [D]
            for k in range(len(stages) - 1, 0, -1):
                children = lineage[-1].children
                assert children is not None
                parent_id = children[0]
                parent = rep_maps[k - 1][parent_id]
                lineage.append(parent)

            lineage.reverse()

            lineage_path = []
            for k in range(1, len(stages)):
                rep = lineage[k]
                if rep.is_induced:
                    lineage_path.append((0, (), 0))  # Placeholder for Case 2
                    continue

                g_i = g_i_seq[k]
                prev_stage = stages[k - 1]
                parent = lineage[k - 1]
                X_iF = prev_stage.X_dict[g_i][parent.id]
                j = X_iF.perm[0]

                u_j_word = []
                current_j = j
                for step_k in range(k - 1, 0, -1):
                    step_rep = lineage[step_k]
                    if step_rep.is_induced:
                        prev_dim = lineage[step_k - 1].dim
                        m, r = divmod(current_j, prev_dim)
                        if m > 0:
                            step_g_i = g_i_seq[step_k]
                            u_j_word.append((step_g_i, m))
                        current_j = r

                u_j_word.reverse()
                t_word = tuple([(g_i, 1)] + u_j_word)

                current_pos = 0
                current_val = 0
                for gen, exp in reversed(t_word):
                    for _ in range(exp):
                        mat = D.matrices[gen]
                        current_val = (current_val + mat.vals[current_pos]) % e
                        current_pos = mat.perm[current_pos]

                lambda_i = current_val
                lineage_path.append((g_i, t_word, lambda_i))

            paths[D.id] = tuple(lineage_path)

        conjugates = {
            rep.id: rep.conjugate_ptr
            for rep in final_stage.representations
            if rep.conjugate_ptr is not None
        }
        return cls(paths=paths, e=e, conjugates=conjugates)
