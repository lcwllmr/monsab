#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(deprecated)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_return)]
#![allow(clippy::type_complexity)]
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp;
use std::collections::HashMap;

fn factorial(n: usize) -> f64 {
    (1..=n).map(|v| v as f64).product()
}

fn math_comb(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = cmp::min(k, n - k);
    let mut res = 1;
    for i in 1..=k {
        res = res * (n - i + 1) / i;
    }
    res
}

fn unrank_tuple(
    m_id: usize,
    n: usize,
    d: usize,
    offsets: &[usize],
    binom_2: &[usize],
    binom_3: &[usize],
    is_squarefree: bool,
) -> Vec<usize> {
    let mut k = 0;
    for i in 0..=d {
        if offsets[i] <= m_id && m_id < offsets[i + 1] {
            k = i;
            break;
        }
    }

    if k == 0 {
        return vec![];
    }

    let rank = m_id - offsets[k];

    if k == 1 {
        return vec![rank];
    } else if k == 2 {
        if is_squarefree {
            // Combinadic: rank = comb(c2, 2) + c1, c1 < c2
            let mut c2 = ((rank as f64) * 2.0).sqrt() as usize;
            while binom_2[c2] > rank {
                c2 -= 1;
            }
            while binom_2[c2 + 1] <= rank {
                c2 += 1;
            }
            let c1 = rank - binom_2[c2];
            return vec![c1, c2];
        } else {
            // With-repetition: rank = comb(c2+1, 2) + c1, c1 <= c2
            let mut c2 = ((rank as f64) * 2.0).sqrt() as usize;
            while binom_2[c2 + 1] > rank {
                c2 -= 1;
            }
            while binom_2[c2 + 2] <= rank {
                c2 += 1;
            }
            let c1 = rank - binom_2[c2 + 1];
            return vec![c1, c2];
        }
    } else if k == 3 {
        let mut c3 = ((rank as f64) * 6.0).powf(1.0 / 3.0) as usize;
        while binom_3[c3] > rank {
            c3 -= 1;
        }
        while binom_3[c3 + 1] <= rank {
            c3 += 1;
        }
        let rem = rank - binom_3[c3];

        let mut c2 = ((rem as f64) * 2.0).sqrt() as usize;
        while binom_2[c2] > rem {
            c2 -= 1;
        }
        while binom_2[c2 + 1] <= rem {
            c2 += 1;
        }
        let c1 = rem - binom_2[c2];

        return vec![c1, c2, c3];
    }

    let mut c = Vec::new();
    let mut rem = rank;
    for i in (1..=k).rev() {
        let mut guess = if rem == 0 {
            i - 1
        } else {
            ((rem as f64) * factorial(i)).powf(1.0 / (i as f64)) as usize
        };
        while math_comb(guess, i) > rem {
            guess -= 1;
        }
        while math_comb(guess + 1, i) <= rem {
            guess += 1;
        }
        c.push(guess);
        rem -= math_comb(guess, i);
    }

    let mut res = vec![0; k];
    for j in 0..k {
        res[j] = c[k - 1 - j];
    }
    res
}

fn rank_tuple(
    tup: &[usize],
    n: usize,
    d: usize,
    offsets: &[usize],
    binom_2: &[usize],
    binom_3: &[usize],
    is_squarefree: bool,
) -> usize {
    let k = tup.len();
    if k == 0 {
        return 0;
    }

    let mut rank = 0;
    if k == 1 {
        rank = tup[0];
    } else if k == 2 {
        // Both squarefree and non-squarefree use the generic formula:
        // rank = sum(comb(tup[i] + offset, i+1) for i in range(k))
        // For squarefree: rank = comb(tup[0], 1) + comb(tup[1], 2) = tup[0] + binom_2[tup[1]]
        // For non-squarefree: rank = comb(tup[0], 1) + comb(tup[1]+1, 2) = tup[0] + binom_2[tup[1]+1]
        if is_squarefree {
            rank = tup[0] + binom_2[tup[1]];
        } else {
            rank = tup[0] + binom_2[tup[1] + 1];
        }
    } else if k == 3 {
        if is_squarefree {
            rank = binom_3[tup[2]] + binom_2[tup[1]] + tup[0];
        } else {
            rank = binom_3[tup[2] + 2] + binom_2[tup[1] + 1] + tup[0];
        }
    } else {
        for i in 0..k {
            if is_squarefree {
                rank += math_comb(tup[i], i + 1);
            } else {
                rank += math_comb(tup[i] + i, i + 1);
            }
        }
    }

    offsets[k] + rank
}

fn apply_permutation(m_tuple: &[usize], p: &[usize], is_squarefree: bool) -> Vec<usize> {
    let mut mapped = Vec::with_capacity(m_tuple.len());
    for &val in m_tuple {
        mapped.push(p[val]);
    }
    mapped.sort_unstable();
    if !is_squarefree {
        let mut j = 0;
        for i in 1..mapped.len() {
            if mapped[i] != mapped[j] {
                j += 1;
                mapped[j] = mapped[i];
            }
        }
        mapped.truncate(j + 1);
    }
    mapped
}

fn apply_permutation_arr(
    m_id: usize,
    inv_data: &[usize],
    n: usize,
    d: usize,
    offsets: &[usize],
    binom_2: &[usize],
    binom_3: &[usize],
    is_squarefree: bool,
) -> usize {
    if d >= 1 && m_id < offsets[2] {
        if m_id < offsets[1] {
            return 0;
        }
        let v = m_id - offsets[1];
        return offsets[1] + inv_data[v];
    } else if d >= 2 && m_id < offsets[3] && !is_squarefree {
        let rank = m_id - offsets[2];
        // Unrank: rank = binom_2[c2+1] + c1 where c1 <= c2
        let mut c2 = ((rank as f64) * 2.0).sqrt() as usize;
        while binom_2[c2 + 1] > rank {
            c2 -= 1;
        }
        while binom_2[c2 + 2] <= rank {
            c2 += 1;
        }
        let c1 = rank - binom_2[c2 + 1];
        let mut v0 = inv_data[c1];
        let mut v1 = inv_data[c2];
        if v0 > v1 {
            std::mem::swap(&mut v0, &mut v1);
        }
        return offsets[2] + binom_2[v1 + 1] + v0;
    } else if d >= 2 && m_id < offsets[3] && is_squarefree {
        let rank = m_id - offsets[2];
        let mut c2 = ((rank as f64) * 2.0).sqrt() as usize;
        while binom_2[c2] > rank {
            c2 -= 1;
        }
        while binom_2[c2 + 1] <= rank {
            c2 += 1;
        }
        let c1 = rank - binom_2[c2];
        let mut v0 = inv_data[c1];
        let mut v1 = inv_data[c2];
        if v0 > v1 {
            std::mem::swap(&mut v0, &mut v1);
        }
        return offsets[2] + binom_2[v1] + v0;
    } else if d >= 3 && m_id < offsets[4] && is_squarefree {
        let rank = m_id - offsets[3];
        let mut c3 = ((rank as f64) * 6.0).powf(1.0 / 3.0) as usize;
        while binom_3[c3] > rank {
            c3 -= 1;
        }
        while binom_3[c3 + 1] <= rank {
            c3 += 1;
        }
        let rem = rank - binom_3[c3];

        let mut c2 = ((rem as f64) * 2.0).sqrt() as usize;
        while binom_2[c2] > rem {
            c2 -= 1;
        }
        while binom_2[c2 + 1] <= rem {
            c2 += 1;
        }
        let c1 = rem - binom_2[c2];

        let mut v0 = inv_data[c1];
        let mut v1 = inv_data[c2];
        let mut v2 = inv_data[c3];

        if v0 > v1 {
            std::mem::swap(&mut v0, &mut v1);
        }
        if v1 > v2 {
            std::mem::swap(&mut v1, &mut v2);
        }
        if v0 > v1 {
            std::mem::swap(&mut v0, &mut v1);
        }

        return offsets[3] + binom_3[v2] + binom_2[v1] + v0;
    } else {
        let tup = unrank_tuple(m_id, n, d, offsets, binom_2, binom_3, is_squarefree);
        let mapped = apply_permutation(&tup, inv_data, is_squarefree);
        return rank_tuple(&mapped, n, d, offsets, binom_2, binom_3, is_squarefree);
    }
}

#[derive(Clone)]
pub struct SABBlockOutput {
    pub rep_id: usize,
    pub dim: usize,
    pub e: usize,
    pub orbit_reps: Vec<Vec<usize>>,
    pub orbit_reps_flat: Vec<usize>,
    pub orbit_sizes: Vec<usize>,
    pub valid_cols: Vec<usize>,
    pub j_values: Vec<usize>,
    pub l_values: Vec<usize>,
}

#[derive(Clone)]
#[pyclass]
pub struct RustSABBlock {
    #[pyo3(get)]
    pub rep_id: usize,
    #[pyo3(get)]
    pub dim: usize,
    #[pyo3(get)]
    pub e: usize,
    #[pyo3(get)]
    pub orbit_reps_flat: Vec<usize>,
    #[pyo3(get)]
    pub orbit_sizes: Vec<usize>,
    pub valid_cols: Vec<u32>,
    pub j_values: Vec<u32>,
    pub l_values: Vec<u32>,
    #[pyo3(get)]
    pub d_k: usize,
    pub coset_reps: Option<Vec<Vec<usize>>>,
    pub coset_reps_inv: Option<Vec<Vec<usize>>>,
    #[pyo3(get)]
    pub fs_indicator: Option<i32>,
    pub v_matrix_data: Option<Vec<usize>>,
    pub txj_cache: Option<Vec<u32>>,
    pub v_cols: Option<Vec<Vec<(usize, num_complex::Complex64)>>>,
    pub v_dagger_rows: Option<Vec<Vec<(usize, num_complex::Complex64)>>>,
}

#[derive(Clone)]
#[pyclass]
pub struct RustSABTransform {
    pub blocks: Vec<RustSABBlock>,
    #[pyo3(get)]
    pub n_monomials: usize,
    pub n: usize,
    pub d: usize,
    pub offsets: Vec<usize>,
    pub binom_2: Vec<usize>,
    pub binom_3: Vec<usize>,
    pub is_squarefree: bool,
}

#[pymethods]
impl RustSABTransform {
    #[getter]
    fn get_blocks(&self) -> Vec<RustSABBlock> {
        self.blocks.clone()
    }

    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (batch_data, batch_indices, batch_indptr, reynolds=false, realize=false))]
    fn extract_batch<'py>(
        &self,
        py: pyo3::Python<'py>,
        batch_data: Vec<numpy::PyReadonlyArray1<'py, f64>>,
        batch_indices: Vec<numpy::PyReadonlyArray1<'py, i32>>,
        batch_indptr: Vec<numpy::PyReadonlyArray1<'py, i32>>,
        reynolds: bool,
        realize: bool,
    ) -> pyo3::PyResult<
        Vec<
            Vec<(
                pyo3::Bound<'py, numpy::PyArray1<num_complex::Complex64>>,
                pyo3::Bound<'py, numpy::PyArray1<i32>>,
                pyo3::Bound<'py, numpy::PyArray1<i32>>,
                usize,
            )>,
        >,
    > {
        let num_matrices = batch_data.len();

        let mut data_slices = Vec::with_capacity(num_matrices);
        let mut indices_slices = Vec::with_capacity(num_matrices);
        let mut indptr_slices = Vec::with_capacity(num_matrices);

        for i in 0..num_matrices {
            data_slices.push(batch_data[i].as_slice()?);
            indices_slices.push(batch_indices[i].as_slice()?);
            indptr_slices.push(batch_indptr[i].as_slice()?);
        }

        use rayon::prelude::*;
        let blocks_results: Vec<Vec<(Vec<num_complex::Complex64>, Vec<i32>, Vec<i32>, usize)>> =
            self.blocks
                .par_iter()
                .map(|block| {
                    let mut block_batch = Vec::with_capacity(num_matrices);
                    let m_k = block.dim;

                    let mut phase_map = Vec::with_capacity(block.e);
                    let e_f64 = block.e as f64;
                    for i in 0..block.e {
                        let angle = -2.0 * std::f64::consts::PI * (i as f64) / e_f64;
                        phase_map.push(num_complex::Complex64::new(angle.cos(), angle.sin()));
                    }

                    for mat_idx in 0..num_matrices {
                        let data = data_slices[mat_idx];
                        let indices = indices_slices[mat_idx];
                        let indptr = indptr_slices[mat_idx];

                        let mut vals = Vec::with_capacity(block.valid_cols.len());
                        let mut rows = Vec::with_capacity(block.valid_cols.len());
                        let mut cols = Vec::with_capacity(block.valid_cols.len());

                        let m_iter = if reynolds && block.coset_reps.is_some() {
                            block.d_k
                        } else {
                            1
                        };
                        let norm_factor = if reynolds { 1.0 / (m_iter as f64) } else { 1.0 };

                        for (local_row, &y) in block.orbit_reps_flat.iter().enumerate() {
                            for m in 0..m_iter {
                                let y_prime = if reynolds {
                                    if let Some(reps_inv) = &block.coset_reps_inv {
                                        apply_permutation_arr(
                                            y,
                                            &reps_inv[m],
                                            self.n,
                                            self.d,
                                            &self.offsets,
                                            &self.binom_2,
                                            &self.binom_3,
                                            self.is_squarefree,
                                        )
                                    } else {
                                        y
                                    }
                                } else {
                                    y
                                };

                                let row_start = indptr[y_prime] as usize;
                                let row_end = indptr[y_prime + 1] as usize;

                                for i in row_start..row_end {
                                    let c = indices[i] as usize;
                                    let w = if reynolds {
                                        if let Some(reps) = &block.coset_reps {
                                            apply_permutation_arr(
                                                c,
                                                &reps[m],
                                                self.n,
                                                self.d,
                                                &self.offsets,
                                                &self.binom_2,
                                                &self.binom_3,
                                                self.is_squarefree,
                                            )
                                        } else {
                                            c
                                        }
                                    } else {
                                        c
                                    };

                                    if let Ok(idx) = block.valid_cols.binary_search(&(w as u32)) {
                                        let j_val = block.j_values[idx];
                                        let l_val = block.l_values[idx] as usize;

                                        let val = data[i];
                                        let phase = phase_map[l_val];
                                        let norm = ((block.orbit_sizes[local_row] as f64)
                                            / (block.orbit_sizes[j_val as usize] as f64))
                                            .sqrt();

                                        vals.push(phase * val * norm * norm_factor);
                                        rows.push(local_row as i32);
                                        cols.push(j_val as i32);
                                    }
                                }
                            }
                        }
                        if realize && block.fs_indicator == Some(0) {
                            let mut real_vals = Vec::with_capacity(vals.len() * 4);
                            let mut real_rows = Vec::with_capacity(vals.len() * 4);
                            let mut real_cols = Vec::with_capacity(vals.len() * 4);

                            // For fs = 0, output 2m_k x 2m_k real matrix
                            // T_real = [A, B; -B, A]
                            // where T = A + iB
                            for i in 0..vals.len() {
                                let r = rows[i];
                                let c = cols[i];
                                let v = vals[i];
                                let m = m_k as i32;

                                if v.re.abs() > 1e-12 || v.im.abs() > 1e-12 {
                                    // A
                                    real_vals.push(num_complex::Complex64::new(v.re, 0.0));
                                    real_rows.push(r);
                                    real_cols.push(c);
                                    real_vals.push(num_complex::Complex64::new(v.re, 0.0));
                                    real_rows.push(r + m);
                                    real_cols.push(c + m);
                                    // B
                                    real_vals.push(num_complex::Complex64::new(v.im, 0.0));
                                    real_rows.push(r);
                                    real_cols.push(c + m);
                                    // -B
                                    real_vals.push(num_complex::Complex64::new(-v.im, 0.0));
                                    real_rows.push(r + m);
                                    real_cols.push(c);
                                }
                            }
                            block_batch.push((real_vals, real_rows, real_cols, 2 * m_k));
                        } else if realize && block.fs_indicator == Some(-1) {
                            let mut real_vals = Vec::with_capacity(vals.len() * 4);
                            let mut real_rows = Vec::with_capacity(vals.len() * 4);
                            let mut real_cols = Vec::with_capacity(vals.len() * 4);

                            for i in 0..vals.len() {
                                let r = rows[i];
                                let c = cols[i];
                                let v = vals[i];
                                let m = m_k as i32;

                                if v.re.abs() > 1e-12 || v.im.abs() > 1e-12 {
                                    // T_real = [A, B; -B, A]
                                    real_vals.push(num_complex::Complex64::new(v.re, 0.0));
                                    real_rows.push(r);
                                    real_cols.push(c);
                                    real_vals.push(num_complex::Complex64::new(v.re, 0.0));
                                    real_rows.push(r + m);
                                    real_cols.push(c + m);
                                    real_vals.push(num_complex::Complex64::new(v.im, 0.0));
                                    real_rows.push(r);
                                    real_cols.push(c + m);
                                    real_vals.push(num_complex::Complex64::new(-v.im, 0.0));
                                    real_rows.push(r + m);
                                    real_cols.push(c);
                                }
                            }

                            block_batch.push((real_vals, real_rows, real_cols, 2 * m_k));
                        } else if realize
                            && block.fs_indicator == Some(1)
                            && block.v_matrix_data.is_some()
                        {
                            let mut real_vals = Vec::with_capacity(vals.len() * 2);
                            let mut real_rows = Vec::with_capacity(vals.len() * 2);
                            let mut real_cols = Vec::with_capacity(vals.len() * 2);

                            if let (Some(v_cols), Some(v_dagger_rows)) =
                                (&block.v_cols, &block.v_dagger_rows)
                            {
                                let mut tmp2: HashMap<usize, num_complex::Complex64> =
                                    HashMap::new();

                                for i in 0..vals.len() {
                                    let r = rows[i] as usize;
                                    let c = cols[i] as usize;
                                    let v = vals[i];

                                    for &(i_idx, v_dag_val) in &v_dagger_rows[r] {
                                        for &(j_idx, v_val) in &v_cols[c] {
                                            let val = v_dag_val * v * v_val;
                                            let flat_idx = i_idx * m_k + j_idx;
                                            *tmp2.entry(flat_idx).or_insert(
                                                num_complex::Complex64::new(0.0, 0.0),
                                            ) += val;
                                        }
                                    }
                                }

                                for (flat_idx, val) in tmp2 {
                                    if val.re.abs() > 1e-12 {
                                        real_vals.push(num_complex::Complex64::new(val.re, 0.0));
                                        real_rows.push((flat_idx / m_k) as i32);
                                        real_cols.push((flat_idx % m_k) as i32);
                                    }
                                }
                            }
                            block_batch.push((real_vals, real_rows, real_cols, m_k));
                        } else {
                            if realize {
                                // Realize true but nu2 = 1 and V is identity/real (no complex chi), output real part only
                                let mut real_vals = Vec::with_capacity(vals.len());
                                let mut real_rows = Vec::with_capacity(vals.len());
                                let mut real_cols = Vec::with_capacity(vals.len());
                                for i in 0..vals.len() {
                                    if vals[i].re.abs() > 1e-12 {
                                        real_vals
                                            .push(num_complex::Complex64::new(vals[i].re, 0.0));
                                        real_rows.push(rows[i]);
                                        real_cols.push(cols[i]);
                                    }
                                }
                                block_batch.push((real_vals, real_rows, real_cols, m_k));
                            } else {
                                block_batch.push((vals, rows, cols, m_k));
                            }
                        }
                    }
                    block_batch
                })
                .collect();

        let mut final_results = Vec::with_capacity(self.blocks.len());
        for block_batch in blocks_results {
            let mut py_block_batch = Vec::with_capacity(num_matrices);
            for (vals, rows, cols, m_k) in block_batch {
                use numpy::IntoPyArray;
                py_block_batch.push((
                    vals.into_pyarray(py),
                    rows.into_pyarray(py),
                    cols.into_pyarray(py),
                    m_k,
                ));
            }
            final_results.push(py_block_batch);
        }

        Ok(final_results)
    }
}

fn dfs(
    n: usize,
    level: usize,
    rep_ids: &[usize],
    reps: &[usize],
    col_to_j: &[isize],
    col_to_l: &[usize],
    k_levels: usize,
    e: usize,
    orbit: &[usize],
    point_to_id: &[usize],
    abstract_paths: &std::collections::HashMap<usize, Vec<(usize, Vec<(usize, usize)>, usize)>>,
    gen_powers: &std::collections::HashMap<usize, Vec<Vec<usize>>>,
    d: usize,
    offsets: &[usize],
    binom_2: &[usize],
    binom_3: &[usize],
    is_squarefree: bool,
    results: &mut Vec<SABBlockOutput>,
) {
    if level == k_levels {
        let mut orbit_reps = Vec::with_capacity(reps.len());
        let mut orbit_reps_flat = Vec::with_capacity(reps.len());

        for &r in reps {
            let m_id = orbit[r];
            let tup = unrank_tuple(m_id, n, d, offsets, binom_2, binom_3, is_squarefree);
            orbit_reps.push(tup);
            orbit_reps_flat.push(m_id);
        }

        let mut counts = vec![0_usize; reps.len()];
        let mut valid_cols = Vec::new();
        let mut j_values = Vec::new();
        let mut l_values = Vec::new();

        for (local_idx, &j_val) in col_to_j.iter().enumerate() {
            if j_val != -1 {
                counts[j_val as usize] += 1;
                valid_cols.push(orbit[local_idx]);
                j_values.push(j_val as usize);
                l_values.push(col_to_l[local_idx]);
            }
        }

        for &rep_id in rep_ids {
            if !reps.is_empty() {
                results.push(SABBlockOutput {
                    rep_id,
                    dim: reps.len(),
                    e,
                    orbit_reps: orbit_reps.clone(),
                    orbit_reps_flat: orbit_reps_flat.clone(),
                    orbit_sizes: counts.clone(),
                    valid_cols: valid_cols.clone(),
                    j_values: j_values.clone(),
                    l_values: l_values.clone(),
                });
            }
        }
        return;
    }

    let mut geom_groups: std::collections::HashMap<
        (usize, Vec<(usize, usize)>),
        Vec<(usize, usize)>,
    > = std::collections::HashMap::new();
    for &rep_id in rep_ids {
        let p_data = &abstract_paths[&rep_id][level - 1];
        geom_groups
            .entry((p_data.0, p_data.1.clone()))
            .or_default()
            .push((rep_id, p_data.2));
    }

    for ((g_i, t_word), rep_lambda_list) in geom_groups.into_iter() {
        if g_i == 0 {
            let next_rep_ids: Vec<usize> = rep_lambda_list.iter().map(|x| x.0).collect();
            dfs(
                n,
                level + 1,
                &next_rep_ids,
                reps,
                col_to_j,
                col_to_l,
                k_levels,
                e,
                orbit,
                point_to_id,
                abstract_paths,
                gen_powers,
                d,
                offsets,
                binom_2,
                binom_3,
                is_squarefree,
                results,
            );
            continue;
        }

        let mut visited_orbits = vec![false; reps.len()];
        let mut cycles = Vec::new();
        let mut w_of = std::collections::HashMap::new();

        for a in 0..reps.len() {
            if visited_orbits[a] {
                continue;
            }

            let mut curr_val = orbit[reps[a]];
            for &(gen, exp) in t_word.iter().rev() {
                let inv_data = &gen_powers[&gen][exp - 1];
                curr_val = apply_permutation_arr(
                    curr_val,
                    inv_data,
                    n,
                    d,
                    offsets,
                    binom_2,
                    binom_3,
                    is_squarefree,
                );
            }

            let w_id = point_to_id[curr_val];
            w_of.insert(a, w_id);

            let target_a = col_to_j[w_id];

            if target_a == a as isize {
                cycles.push(vec![a]);
                visited_orbits[a] = true;
            } else {
                let mut cycle = Vec::new();
                let mut curr_a = a;
                while !visited_orbits[curr_a] {
                    cycle.push(curr_a);
                    visited_orbits[curr_a] = true;

                    let mut curr_val = orbit[reps[curr_a]];
                    for &(gen, exp) in t_word.iter().rev() {
                        let inv_data = &gen_powers[&gen][exp - 1];
                        curr_val = apply_permutation_arr(
                            curr_val,
                            inv_data,
                            n,
                            d,
                            offsets,
                            binom_2,
                            binom_3,
                            is_squarefree,
                        );
                    }
                    let w_id_curr = point_to_id[curr_val];
                    let target_curr = col_to_j[w_id_curr];
                    w_of.insert(curr_a, w_id_curr);
                    curr_a = target_curr as usize;
                }
                cycles.push(cycle);
            }
        }

        let mut cycle_data = Vec::new();
        for cycle in &cycles {
            let l = cycle.len();
            if l == 1 {
                let a = cycle[0];
                let phi = col_to_l[w_of[&a]] as i32;
                cycle_data.push((cycle.clone(), phi, Vec::new()));
            } else {
                let mut phi_list = Vec::new();
                let mut phi_sum = 0;
                for &a in cycle {
                    let phi = col_to_l[w_of[&a]] as i32;
                    phi_list.push(phi);
                    phi_sum += phi;
                }
                cycle_data.push((cycle.clone(), phi_sum, phi_list));
            }
        }

        let mut lambda_groups: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for &(rep_id, lambda_i) in rep_lambda_list.iter() {
            lambda_groups.entry(lambda_i).or_default().push(rep_id);
        }

        let mut lambda_cycles: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for &lambda_i in lambda_groups.keys() {
            let mut admissible = Vec::new();
            for (cycle_idx, (cycle, phi_sum, _)) in cycle_data.iter().enumerate() {
                let l = cycle.len();
                if l == 1 {
                    if lambda_i as i32 == *phi_sum {
                        admissible.push(cycle_idx);
                    }
                } else {
                    let c_l = (l as i32) * (lambda_i as i32) - phi_sum;
                    if c_l % (e as i32) == 0 {
                        admissible.push(cycle_idx);
                    }
                }
            }
            lambda_cycles.insert(lambda_i, admissible);
        }

        let mut admissibility_groups: std::collections::HashMap<
            Vec<usize>,
            Vec<(usize, Vec<usize>)>,
        > = std::collections::HashMap::new();
        for (&lambda_i, rep_ids_grp) in lambda_groups.iter() {
            let adm = lambda_cycles[&lambda_i].clone();
            admissibility_groups
                .entry(adm)
                .or_default()
                .push((lambda_i, rep_ids_grp.clone()));
        }

        for (adm, lam_rep_list) in admissibility_groups.iter() {
            let mut new_reps = Vec::new();
            let mut state_map = vec![usize::MAX; reps.len()];

            for &cycle_idx in adm {
                let cycle = &cycle_data[cycle_idx].0;
                let new_idx = new_reps.len();
                new_reps.push(reps[cycle[0]]);
                for &a in cycle {
                    state_map[a] = new_idx;
                }
            }

            let mut new_col_to_j = vec![-1_isize; col_to_j.len()];
            for (j_idx, &j_val) in col_to_j.iter().enumerate() {
                if j_val != -1 && state_map[j_val as usize] != usize::MAX {
                    new_col_to_j[j_idx] = state_map[j_val as usize] as isize;
                }
            }

            for (lam_i, rep_ids_grp) in lam_rep_list.iter() {
                let mut all_len_1 = true;
                for &cycle_idx in adm {
                    if cycle_data[cycle_idx].0.len() > 1 {
                        all_len_1 = false;
                        break;
                    }
                }

                if all_len_1 {
                    dfs(
                        n,
                        level + 1,
                        rep_ids_grp,
                        &new_reps,
                        &new_col_to_j,
                        col_to_l,
                        k_levels,
                        e,
                        orbit,
                        point_to_id,
                        abstract_paths,
                        gen_powers,
                        d,
                        offsets,
                        binom_2,
                        binom_3,
                        is_squarefree,
                        results,
                    );
                } else {
                    let mut state_l_offset = vec![0_i32; reps.len()];
                    for &cycle_idx in adm {
                        let (cycle, _, phi_list) = &cycle_data[cycle_idx];
                        let l = cycle.len();
                        if l > 1 {
                            let mut c_r = 0_i32;
                            for r in 0..l {
                                let curr_a = cycle[r];
                                if r > 0 {
                                    c_r = (c_r + (*lam_i as i32) - phi_list[r - 1]) % (e as i32);
                                }
                                let mut val = c_r % (e as i32);
                                if val < 0 {
                                    val += e as i32;
                                }
                                state_l_offset[curr_a] = val;
                            }
                        }
                    }
                    let mut new_col_to_l = vec![0_usize; orbit.len()];
                    for y in 0..orbit.len() {
                        if col_to_j[y] != -1 {
                            let offset = state_l_offset[col_to_j[y] as usize];
                            new_col_to_l[y] = ((col_to_l[y] as i32 + offset) % (e as i32)) as usize;
                        }
                    }
                    dfs(
                        n,
                        level + 1,
                        rep_ids_grp,
                        &new_reps,
                        &new_col_to_j,
                        &new_col_to_l,
                        k_levels,
                        e,
                        orbit,
                        point_to_id,
                        abstract_paths,
                        gen_powers,
                        d,
                        offsets,
                        binom_2,
                        binom_3,
                        is_squarefree,
                        results,
                    );
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (orbits_list, abstract_paths_dict, g_gens_dict, n, d, e, is_squarefree, n_monomials, fs_indicators, v_matrices, coset_reps=None, coset_reps_inv=None))]
fn build_sab_blocks(
    orbits_list: Bound<'_, pyo3::types::PyList>,
    abstract_paths_dict: Bound<'_, pyo3::types::PyDict>,
    g_gens_dict: Bound<'_, pyo3::types::PyDict>,
    n: usize,
    d: usize,
    e: usize,
    is_squarefree: bool,
    n_monomials: usize,
    fs_indicators: Bound<'_, pyo3::types::PyDict>,
    v_matrices: Bound<'_, pyo3::types::PyDict>,
    coset_reps: Option<std::collections::HashMap<usize, Vec<Vec<usize>>>>,
    coset_reps_inv: Option<std::collections::HashMap<usize, Vec<Vec<usize>>>>,
) -> PyResult<RustSABTransform> {
    let mut abstract_paths: HashMap<usize, Vec<(usize, Vec<(usize, usize)>, usize)>> =
        HashMap::new();

    for (level_obj, paths_obj) in abstract_paths_dict.iter() {
        let level: usize = level_obj.extract()?;
        let mut paths_for_level = Vec::new();
        let paths_list: Bound<'_, pyo3::types::PyList> = paths_obj.extract()?;
        for item in paths_list.iter() {
            let tuple: Bound<'_, pyo3::types::PyTuple> = item.extract()?;
            let g_k: usize = tuple.get_item(0)?.extract()?;
            let adm_list: Bound<'_, pyo3::types::PyList> = tuple.get_item(1)?.extract()?;
            let mut adm = Vec::new();
            for adm_item in adm_list.iter() {
                let adm_tuple: Bound<'_, pyo3::types::PyTuple> = adm_item.extract()?;
                adm.push((
                    adm_tuple.get_item(0)?.extract()?,
                    adm_tuple.get_item(1)?.extract()?,
                ));
            }
            let lambda_i: usize = tuple.get_item(2)?.extract()?;
            paths_for_level.push((g_k, adm, lambda_i));
        }
        abstract_paths.insert(level, paths_for_level);
    }

    let mut orbit_data: Vec<Vec<usize>> = Vec::new();
    for v_obj in orbits_list.iter() {
        // The items are tuples or lists, so we can extract as a generic PySequence or try PyTuple/PyList.
        // Or simply extract as a Vec<usize> directly! PyO3 can extract standard Python lists/tuples into Rust Vecs.
        let v_vec: Vec<usize> = v_obj.extract()?;
        orbit_data.push(v_vec);
    }

    let mut max_exp_map: HashMap<usize, usize> = HashMap::new();
    for paths_for_level in abstract_paths.values() {
        for (_, adm, _) in paths_for_level {
            for &(gen, exp) in adm {
                let curr = max_exp_map.entry(gen).or_insert(0);
                if exp > *curr {
                    *curr = exp;
                }
            }
        }
    }

    let mut gen_powers: HashMap<usize, Vec<Vec<usize>>> = HashMap::new();
    for (k_obj, v_obj) in g_gens_dict.iter() {
        let g_k: usize = k_obj.extract()?;
        let inv_data: Vec<usize> = v_obj.extract()?;

        let max_e = std::cmp::max(1, *max_exp_map.get(&g_k).unwrap_or(&1));
        let mut powers = Vec::new();
        let mut current_p = inv_data.clone();
        powers.push(current_p.clone());

        for _ in 1..max_e {
            let mut next_p = vec![0; n];
            for i in 0..n {
                next_p[i] = current_p[inv_data[i]];
            }
            current_p = next_p;
            powers.push(current_p.clone());
        }
        gen_powers.insert(g_k, powers);
    }

    let mut offsets = vec![0; d + 2];
    let mut total = 0;
    for k in 0..=d {
        offsets[k] = total;
        let count = if k == 0 {
            1
        } else {
            math_comb(n + k - 1 - (if is_squarefree { k - 1 } else { 0 }), k)
        };
        total += count;
    }
    offsets[d + 1] = total;

    // binom_2[i] = comb(i, 2), binom_3[i] = comb(i, 3)
    let binom_size = n + 3; // match Python's range(n + 3) for non-squarefree, range(n + 1) for squarefree
    let mut binom_2 = Vec::with_capacity(binom_size);
    let mut binom_3 = Vec::with_capacity(binom_size);
    for i in 0..binom_size {
        binom_2.push(math_comb(i, 2));
        binom_3.push(math_comb(i, 3));
    }

    let k_levels = abstract_paths.values().next().map(|v| v.len()).unwrap_or(0) + 1;

    let rep_ids: Vec<usize> = abstract_paths.keys().copied().collect();

    let all_blocks: Vec<Vec<SABBlockOutput>> = orbit_data
        .par_iter()
        .map(|orbit| {
            let mut results = Vec::new();
            let col_to_j: Vec<isize> = (0..orbit.len() as isize).collect();
            let col_to_l = vec![0_usize; orbit.len()];
            let all_reps: Vec<usize> = (0..orbit.len()).collect();

            // Construct O(1) point_to_id map
            let mut point_to_id = vec![0_usize; n_monomials];
            for (idx, &val) in orbit.iter().enumerate() {
                point_to_id[val] = idx;
            }

            dfs(
                n,
                1,
                &rep_ids,
                &all_reps,
                &col_to_j,
                &col_to_l,
                k_levels,
                e,
                orbit,
                &point_to_id,
                &abstract_paths,
                &gen_powers,
                d,
                &offsets,
                &binom_2,
                &binom_3,
                is_squarefree,
                &mut results,
            );

            results
        })
        .collect();

    let mut all_blocks_flat = Vec::new();
    for mut blocks in all_blocks {
        all_blocks_flat.append(&mut blocks);
    }
    all_blocks_flat.sort_by_key(|b| b.rep_id);

    let mut merged_blocks = Vec::new();

    let mut current_rep: Option<usize> = None;

    let mut current_dim = 0;
    let mut current_e = 0;
    let mut current_orbit_reps_flat = Vec::new();
    let mut current_orbit_sizes = Vec::new();

    let mut current_valid_cols = Vec::new();
    let mut current_j_values = Vec::new();
    let mut current_l_values = Vec::new();

    for b in all_blocks_flat {
        if current_rep.is_none() || current_rep.unwrap() != b.rep_id {
            if let Some(rep) = current_rep {
                let action_fwd = coset_reps.as_ref().and_then(|m| m.get(&rep).cloned());
                let action_inv = coset_reps_inv.as_ref().and_then(|m| m.get(&rep).cloned());
                let d_k = if let Some(ref a) = action_fwd {
                    a.len()
                } else {
                    1
                };

                let fs_ind = fs_indicators
                    .get_item(rep)
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<i32>().ok());
                let v_mat = v_matrices
                    .get_item(rep)
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract::<Vec<usize>>().ok());

                let mut sorted_indices: Vec<usize> = (0..current_valid_cols.len()).collect();
                sorted_indices.sort_unstable_by_key(|&i| current_valid_cols[i]);

                let mut sorted_valid = Vec::with_capacity(current_valid_cols.len());
                let mut sorted_j = Vec::with_capacity(current_valid_cols.len());
                let mut sorted_l = Vec::with_capacity(current_valid_cols.len());

                for i in sorted_indices {
                    sorted_valid.push(current_valid_cols[i]);
                    sorted_j.push(current_j_values[i]);
                    sorted_l.push(current_l_values[i]);
                }

                let mut v_cols = None;
                let mut v_dagger_rows = None;
                let mut txj_cache = None;
                if let Some(t_data) = &v_mat {
                    let mut cache = Vec::with_capacity(current_dim);
                    for j in 0..current_dim {
                        let xj = current_orbit_reps_flat[j];
                        let txj = apply_permutation_arr(
                            xj,
                            t_data,
                            n,
                            d,
                            &offsets,
                            &binom_2,
                            &binom_3,
                            is_squarefree,
                        );
                        cache.push(txj as u32);
                    }

                    if fs_ind == Some(1) {
                        let mut vs = vec![Vec::new(); current_dim];
                        let mut vd = vec![Vec::new(); current_dim];
                        let mut visited_orbits = vec![false; current_dim];
                        for j in 0..current_dim {
                            if visited_orbits[j] {
                                continue;
                            }
                            let txj = cache[j];

                            let mut pi_j = usize::MAX;
                            let mut h_phase = 0;
                            if let Ok(idx) = sorted_valid.binary_search(&{ txj }) {
                                pi_j = sorted_j[idx] as usize;
                                h_phase = sorted_l[idx];
                            }

                            if pi_j == j {
                                let phase =
                                    std::f64::consts::PI * 2.0 * (h_phase as f64) / (e as f64);
                                let v_val = num_complex::Complex64::new(
                                    (phase / 2.0).cos(),
                                    (phase / 2.0).sin(),
                                );
                                vs[j].push((j, v_val));
                                vd[j].push((j, v_val.conj()));
                                visited_orbits[j] = true;
                            } else {
                                let phase =
                                    std::f64::consts::PI * 2.0 * (h_phase as f64) / (e as f64);
                                let v_val = num_complex::Complex64::new(
                                    (phase / 2.0).cos(),
                                    (phase / 2.0).sin(),
                                ) / std::f64::consts::SQRT_2;

                                vs[j].push((j, v_val));
                                vs[j].push((pi_j, v_val));
                                vs[pi_j].push((j, v_val * num_complex::Complex64::new(0.0, 1.0)));
                                vs[pi_j]
                                    .push((pi_j, v_val * num_complex::Complex64::new(0.0, -1.0)));

                                vd[j].push((j, v_val.conj()));
                                vd[pi_j].push((j, v_val.conj()));
                                vd[j].push((
                                    pi_j,
                                    (v_val * num_complex::Complex64::new(0.0, 1.0)).conj(),
                                ));
                                vd[pi_j].push((
                                    pi_j,
                                    (v_val * num_complex::Complex64::new(0.0, -1.0)).conj(),
                                ));

                                visited_orbits[j] = true;
                                visited_orbits[pi_j] = true;
                            }
                        }
                        v_cols = Some(vs);
                        v_dagger_rows = Some(vd);
                    }

                    txj_cache = Some(cache);
                }

                merged_blocks.push(RustSABBlock {
                    rep_id: rep,
                    dim: current_dim,
                    e: current_e,
                    orbit_reps_flat: current_orbit_reps_flat.clone(),
                    orbit_sizes: current_orbit_sizes.clone(),
                    valid_cols: sorted_valid,
                    j_values: sorted_j,
                    l_values: sorted_l,
                    d_k,
                    coset_reps: action_fwd,
                    coset_reps_inv: action_inv,
                    fs_indicator: fs_ind,
                    v_matrix_data: v_mat,
                    txj_cache,
                    v_cols,
                    v_dagger_rows,
                });
            }

            current_rep = Some(b.rep_id);
            current_dim = b.dim;
            current_e = b.e;
            current_orbit_reps_flat = b.orbit_reps_flat.clone();
            current_orbit_sizes = b.orbit_sizes.clone();

            current_valid_cols.clear();
            current_j_values.clear();
            current_l_values.clear();

            for (i, &col) in b.valid_cols.iter().enumerate() {
                current_valid_cols.push(col as u32);
                current_j_values.push(b.j_values[i] as u32);
                current_l_values.push(b.l_values[i] as u32);
            }
        } else {
            let dim_offset = current_dim;
            current_dim += b.dim;
            current_orbit_reps_flat.extend_from_slice(&b.orbit_reps_flat);
            current_orbit_sizes.extend_from_slice(&b.orbit_sizes);

            for (i, &col) in b.valid_cols.iter().enumerate() {
                current_valid_cols.push(col as u32);
                current_j_values.push((b.j_values[i] + dim_offset) as u32);
                current_l_values.push(b.l_values[i] as u32);
            }
        }
    }

    if let Some(rep) = current_rep {
        let action_fwd = coset_reps.as_ref().and_then(|m| m.get(&rep).cloned());
        let action_inv = coset_reps_inv.as_ref().and_then(|m| m.get(&rep).cloned());
        let d_k = if let Some(ref a) = action_fwd {
            a.len()
        } else {
            1
        };
        let fs_ind = fs_indicators
            .get_item(rep)
            .ok()
            .flatten()
            .and_then(|v| v.extract::<i32>().ok());
        let v_mat = v_matrices
            .get_item(rep)
            .ok()
            .flatten()
            .and_then(|v| v.extract::<Vec<usize>>().ok());

        let mut sorted_indices: Vec<usize> = (0..current_valid_cols.len()).collect();
        sorted_indices.sort_unstable_by_key(|&i| current_valid_cols[i]);

        let mut sorted_valid = Vec::with_capacity(current_valid_cols.len());
        let mut sorted_j = Vec::with_capacity(current_valid_cols.len());
        let mut sorted_l = Vec::with_capacity(current_valid_cols.len());

        for i in sorted_indices {
            sorted_valid.push(current_valid_cols[i]);
            sorted_j.push(current_j_values[i]);
            sorted_l.push(current_l_values[i]);
        }

        let mut v_cols = None;
        let mut v_dagger_rows = None;
        let mut txj_cache = None;
        if let Some(t_data) = &v_mat {
            let mut cache = Vec::with_capacity(current_dim);
            for j in 0..current_dim {
                let xj = current_orbit_reps_flat[j];
                let txj = apply_permutation_arr(
                    xj,
                    t_data,
                    n,
                    d,
                    &offsets,
                    &binom_2,
                    &binom_3,
                    is_squarefree,
                );
                cache.push(txj as u32);
            }

            if fs_ind == Some(1) {
                let mut vs = vec![Vec::new(); current_dim];
                let mut vd = vec![Vec::new(); current_dim];
                let mut visited_orbits = vec![false; current_dim];
                for j in 0..current_dim {
                    if visited_orbits[j] {
                        continue;
                    }
                    let txj = cache[j];

                    let mut pi_j = usize::MAX;
                    let mut h_phase = 0;
                    if let Ok(idx) = sorted_valid.binary_search(&{ txj }) {
                        pi_j = sorted_j[idx] as usize;
                        h_phase = sorted_l[idx];
                    }

                    if pi_j == j {
                        let phase = std::f64::consts::PI * 2.0 * (h_phase as f64) / (e as f64);
                        let v_val =
                            num_complex::Complex64::new((phase / 2.0).cos(), (phase / 2.0).sin());
                        vs[j].push((j, v_val));
                        vd[j].push((j, v_val.conj()));
                        visited_orbits[j] = true;
                    } else {
                        let phase = std::f64::consts::PI * 2.0 * (h_phase as f64) / (e as f64);
                        let v_val =
                            num_complex::Complex64::new((phase / 2.0).cos(), (phase / 2.0).sin())
                                / std::f64::consts::SQRT_2;

                        vs[j].push((j, v_val));
                        vs[j].push((pi_j, v_val));
                        vs[pi_j].push((j, v_val * num_complex::Complex64::new(0.0, 1.0)));
                        vs[pi_j].push((pi_j, v_val * num_complex::Complex64::new(0.0, -1.0)));

                        vd[j].push((j, v_val.conj()));
                        vd[pi_j].push((j, v_val.conj()));
                        vd[j].push((pi_j, (v_val * num_complex::Complex64::new(0.0, 1.0)).conj()));
                        vd[pi_j].push((
                            pi_j,
                            (v_val * num_complex::Complex64::new(0.0, -1.0)).conj(),
                        ));

                        visited_orbits[j] = true;
                        visited_orbits[pi_j] = true;
                    }
                }
                v_cols = Some(vs);
                v_dagger_rows = Some(vd);
            }

            txj_cache = Some(cache);
        }

        merged_blocks.push(RustSABBlock {
            rep_id: rep,
            dim: current_dim,
            e: current_e,
            orbit_reps_flat: current_orbit_reps_flat,
            orbit_sizes: current_orbit_sizes,
            valid_cols: sorted_valid,
            j_values: sorted_j,
            l_values: sorted_l,
            d_k,
            coset_reps: action_fwd,
            coset_reps_inv: action_inv,
            fs_indicator: fs_ind,
            v_matrix_data: v_mat,
            txj_cache,
            v_cols,
            v_dagger_rows,
        });
    }

    Ok(RustSABTransform {
        blocks: merged_blocks,
        n_monomials,
        n,
        d,
        offsets,
        binom_2,
        binom_3,
        is_squarefree,
    })
}

use std::collections::{HashSet, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};

fn hash_vec(v: &[usize]) -> u64 {
    let mut hasher = DefaultHasher::new();
    v.hash(&mut hasher);
    hasher.finish()
}

#[pyfunction]
#[pyo3(signature = (g_gens, n, e, h_visited_all, h_gens_fwd_all, char_phases_rep_all))]
fn compute_fs_and_t_data(
    g_gens: Vec<Vec<usize>>,
    n: usize,
    e: usize,
    h_visited_all: HashMap<usize, HashMap<Vec<usize>, usize>>,
    h_gens_fwd_all: HashMap<usize, HashMap<usize, Vec<usize>>>,
    char_phases_rep_all: HashMap<usize, HashMap<usize, usize>>,
) -> PyResult<(HashMap<usize, i32>, HashMap<usize, Option<Vec<usize>>>)> {
    let mut fs_indicators = HashMap::new();
    let mut needs_t_data = HashSet::new();

    let mut nu2_complex_all = HashMap::new();
    for &rep_id in h_visited_all.keys() {
        nu2_complex_all.insert(rep_id, Complex64::new(0.0, 0.0));
    }

    if !h_visited_all.is_empty() {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let identity: Vec<usize> = (0..n).collect();
        visited.insert(hash_vec(&identity));
        queue.push_back(identity);

        while let Some(curr) = queue.pop_front() {
            let mut y2 = vec![0; n];
            for i in 0..n {
                y2[i] = curr[curr[i]];
            }

            for (rep_id, h_visited) in &h_visited_all {
                if let Some(&phase) = h_visited.get(&y2) {
                    let angle = 2.0 * std::f64::consts::PI * (phase as f64) / (e as f64);
                    let chi_h = Complex64::new(angle.cos(), angle.sin());
                    *nu2_complex_all.get_mut(rep_id).unwrap() += chi_h;
                }
            }

            for gen in &g_gens {
                let mut nxt = vec![0; n];
                for i in 0..n {
                    nxt[i] = curr[gen[i]];
                }
                if visited.insert(hash_vec(&nxt)) {
                    queue.push_back(nxt);
                }
            }
        }
    }

    for (rep_id, h_visited) in &h_visited_all {
        let nu2_complex = nu2_complex_all[rep_id];
        let h_len = h_visited.len() as f64;
        let nu2 = (nu2_complex.re / h_len).round() as i32;
        fs_indicators.insert(*rep_id, nu2);

        let mut is_complex_chi = false;
        for &p in h_visited.values() {
            if p != 0 && p * 2 != e {
                is_complex_chi = true;
                break;
            }
        }

        if nu2 == 1 && is_complex_chi {
            needs_t_data.insert(*rep_id);
        }
    }

    let mut found_t_data = HashMap::new();
    for &rep_id in &needs_t_data {
        found_t_data.insert(rep_id, None);
    }

    if !needs_t_data.is_empty() {
        let mut visited_t = HashSet::new();
        let mut queue_t = VecDeque::new();
        let identity: Vec<usize> = (0..n).collect();
        visited_t.insert(hash_vec(&identity));
        queue_t.push_back(identity);

        while let Some(y) = queue_t.pop_front() {
            let mut reps_to_remove = Vec::new();

            for &rep_id in &needs_t_data {
                let h_visited = &h_visited_all[&rep_id];
                if h_visited.contains_key(&y) {
                    continue;
                }

                let mut conjugates = true;
                let h_gens_fwd = &h_gens_fwd_all[&rep_id];
                let char_phases = &char_phases_rep_all[&rep_id];

                for (g_k, h) in h_gens_fwd {
                    let phase = char_phases[g_k];
                    let mut hy = vec![0; n];
                    for i in 0..n {
                        hy[i] = y[h[i]];
                    }
                    let mut y_inv = vec![0; n];
                    for i in 0..n {
                        y_inv[y[i]] = i;
                    }
                    let mut y_inv_h_y = vec![0; n];
                    for i in 0..n {
                        y_inv_h_y[i] = y_inv[hy[i]];
                    }

                    let target_phase = (e - phase) % e;
                    if let Some(&p) = h_visited.get(&y_inv_h_y) {
                        if p != target_phase {
                            conjugates = false;
                            break;
                        }
                    } else {
                        conjugates = false;
                        break;
                    }
                }

                if conjugates {
                    found_t_data.insert(rep_id, Some(y.clone()));
                    reps_to_remove.push(rep_id);
                }
            }

            for rep_id in reps_to_remove {
                needs_t_data.remove(&rep_id);
            }

            if needs_t_data.is_empty() {
                break;
            }

            for gen in &g_gens {
                let mut nxt = vec![0; n];
                for i in 0..n {
                    nxt[i] = y[gen[i]];
                }
                if visited_t.insert(hash_vec(&nxt)) {
                    queue_t.push_back(nxt);
                }
            }
        }
    }

    Ok((fs_indicators, found_t_data))
}

#[pymodule]
fn _backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustSABTransform>()?;
    m.add_function(wrap_pyfunction!(build_sab_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(compute_fs_and_t_data, m)?)?;
    Ok(())
}

#[pymethods]
impl RustSABBlock {
    #[getter]
    fn get_valid_cols<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<u32>> {
        use numpy::IntoPyArray;
        self.valid_cols.clone().into_pyarray(py)
    }

    #[getter]
    fn get_j_values<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<u32>> {
        use numpy::IntoPyArray;
        self.j_values.clone().into_pyarray(py)
    }

    #[getter]
    fn get_l_values<'py>(&self, py: pyo3::Python<'py>) -> pyo3::Bound<'py, numpy::PyArray1<u32>> {
        use numpy::IntoPyArray;
        self.l_values.clone().into_pyarray(py)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_unrank_tuple() {
        let n = 4;
        let d = 2;
        let is_squarefree = true;

        let mut offsets = vec![0; d + 2];
        let mut total = 0;
        for k in 0..=d {
            let count = if k == 0 {
                1
            } else {
                math_comb(n + k - 1 - (if is_squarefree { k - 1 } else { 0 }), k)
            };
            total += count;
            offsets[k + 1] = total;
        }

        let binom_size = n + 3;
        let mut binom_2 = Vec::with_capacity(binom_size);
        let mut binom_3 = Vec::with_capacity(binom_size);
        for i in 0..binom_size {
            binom_2.push(math_comb(i, 2));
            binom_3.push(math_comb(i, 3));
        }

        // Test ranking and unranking for squarefree tuples
        for i in 0..offsets[d + 1] {
            let tup = unrank_tuple(i, n, d, &offsets, &binom_2, &binom_3, is_squarefree);
            let ranked = rank_tuple(&tup, n, d, &offsets, &binom_2, &binom_3, is_squarefree);
            assert_eq!(i, ranked, "Failed on id {} -> {:?} -> {}", i, tup, ranked);
        }
    }
}
