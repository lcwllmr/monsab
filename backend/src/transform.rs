use crate::monomial::apply_permutation_arr;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

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
#[pyclass(name = "SABBlock", module = "monsab._backend")]
pub struct SABBlock {
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
    pub v_cols: Option<Vec<Vec<(usize, Complex64)>>>,
    pub v_dagger_rows: Option<Vec<Vec<(usize, Complex64)>>>,
}

#[derive(Clone)]
#[pyclass(name = "SABTransform", module = "monsab._backend")]
pub struct SABTransform {
    pub blocks: Vec<SABBlock>,
    #[pyo3(get)]
    pub n_monomials: usize,
    pub n: usize,
    pub d: usize,
    pub offsets: Vec<usize>,
    pub binom_2: Vec<usize>,
    pub binom_3: Vec<usize>,
    pub is_squarefree: bool,
    #[pyo3(get, set)]
    pub realize_skip_reps: HashSet<usize>,
}

#[pymethods]
impl SABTransform {
    #[new]
    #[pyo3(signature = (blocks=None, N=None, _rust_transform=None, _realize_skip_reps=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    pub fn new(
        blocks: Option<Vec<SABBlock>>,
        N: Option<usize>,
        _rust_transform: Option<Bound<'_, PyAny>>,
        _realize_skip_reps: Option<HashSet<usize>>,
    ) -> Self {
        Self {
            blocks: blocks.unwrap_or_default(),
            n_monomials: N.unwrap_or(0),
            n: 0,
            d: 0,
            offsets: vec![],
            binom_2: vec![],
            binom_3: vec![],
            is_squarefree: false,
            realize_skip_reps: _realize_skip_reps.unwrap_or_default(),
        }
    }

    #[getter]
    fn get_blocks(&self) -> Vec<SABBlock> {
        self.blocks.clone()
    }

    #[getter]
    #[allow(non_snake_case)]
    fn N(&self) -> usize {
        self.n_monomials
    }

    #[pyo3(signature = (matrices, reynolds=false, realize=false))]
    fn __call__<'py>(
        &self,
        py: pyo3::Python<'py>,
        matrices: Bound<'py, PyAny>,
        reynolds: bool,
        realize: bool,
    ) -> PyResult<Vec<Vec<Bound<'py, PyAny>>>> {
        let mat_list: Vec<Bound<'py, PyAny>> = if matrices.is_instance_of::<pyo3::types::PyList>()
            || matrices.is_instance_of::<pyo3::types::PyTuple>()
        {
            matrices.extract()?
        } else {
            vec![matrices]
        };

        if self.blocks.is_empty() {
            return Ok(vec![]);
        }

        let mut batch_data = Vec::with_capacity(mat_list.len());
        let mut batch_indices = Vec::with_capacity(mat_list.len());
        let mut batch_indptr = Vec::with_capacity(mat_list.len());

        for m in &mat_list {
            let data_obj = m.getattr("data")?;
            let indices_obj = m.getattr("indices")?;
            let indptr_obj = m.getattr("indptr")?;
            let data_arr: PyReadonlyArray1<'py, f64> = data_obj.extract()?;
            let indices_arr: PyReadonlyArray1<'py, i32> = indices_obj.extract()?;
            let indptr_arr: PyReadonlyArray1<'py, i32> = indptr_obj.extract()?;
            batch_data.push(data_arr);
            batch_indices.push(indices_arr);
            batch_indptr.push(indptr_arr);
        }

        let results = self.extract_batch(
            py,
            batch_data,
            batch_indices,
            batch_indptr,
            reynolds,
            realize,
        )?;

        let scipy_sparse = py.import("scipy.sparse")?;
        let np_mod = py.import("numpy")?;
        let float64_type = np_mod.getattr("float64")?;
        let complex128_type = np_mod.getattr("complex128")?;
        let coo_matrix = scipy_sparse.getattr("coo_matrix")?;
        let csr_matrix = scipy_sparse.getattr("csr_matrix")?;

        let mut final_blocks = Vec::with_capacity(self.blocks.len());
        for (block, block_res) in self.blocks.iter().zip(results) {
            if realize && self.realize_skip_reps.contains(&block.rep_id) {
                continue;
            }
            let mut b_list = Vec::with_capacity(block_res.len());
            for (data, rows, cols, m_k) in block_res {
                if data.len()? == 0 {
                    let dtype = if realize {
                        &float64_type
                    } else {
                        &complex128_type
                    };
                    let shape = (m_k, m_k);
                    let kwargs = pyo3::types::PyDict::new(py);
                    kwargs.set_item("dtype", dtype)?;
                    let mat = csr_matrix.call((shape,), Some(&kwargs))?;
                    b_list.push(mat);
                } else {
                    let data_val: Bound<'py, PyAny> = if realize {
                        let slice = data.readonly();
                        let data_real: Vec<f64> = slice.as_slice()?.iter().map(|c| c.re).collect();
                        data_real.into_pyarray(py).into_any()
                    } else {
                        data.into_any()
                    };
                    let rows_val = rows.into_any();
                    let cols_val = cols.into_any();
                    let coords = (rows_val, cols_val);
                    let arg0 = (data_val, coords);
                    let kwargs = pyo3::types::PyDict::new(py);
                    kwargs.set_item("shape", (m_k, m_k))?;
                    let mat = coo_matrix
                        .call((arg0,), Some(&kwargs))?
                        .getattr("tocsr")?
                        .call0()?;
                    b_list.push(mat);
                }
            }
            final_blocks.push(b_list);
        }
        Ok(final_blocks)
    }

    #[pyo3(signature = (sparse=true, realize=false))]
    fn explicit_basis<'py>(
        &self,
        py: pyo3::Python<'py>,
        sparse: bool,
        realize: bool,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        if realize {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Realization is not yet supported for explicit_basis.",
            ));
        }
        if self.blocks.is_empty() {
            return Ok(vec![]);
        }
        let n = self.n_monomials;
        let mut matrices = Vec::with_capacity(self.blocks.len());

        let scipy_sparse = if sparse {
            Some(py.import("scipy.sparse")?)
        } else {
            None
        };
        let coo_matrix = scipy_sparse
            .as_ref()
            .map(|m| m.getattr("coo_matrix"))
            .transpose()?;

        for block in &self.blocks {
            let m_k = block.dim;
            let e = block.e;
            let orbit_sizes = &block.orbit_sizes;
            let valid_cols = &block.valid_cols;
            let j_values = &block.j_values;
            let l_values = &block.l_values;

            if sparse {
                let mut vals = Vec::with_capacity(valid_cols.len());
                for (j, &ell) in j_values.iter().zip(l_values.iter()) {
                    let angle = -2.0 * std::f64::consts::PI * (ell as f64) / (e as f64);
                    let phase = Complex64::new(angle.cos(), angle.sin());
                    let val = phase / (orbit_sizes[*j as usize] as f64).sqrt();
                    vals.push(val);
                }
                let vals_arr = vals.into_pyarray(py);
                let valid_cols_arr = valid_cols.clone().into_pyarray(py);
                let j_values_arr = j_values.clone().into_pyarray(py);
                let coords = (valid_cols_arr, j_values_arr);
                let arg0 = (vals_arr, coords);
                let kwargs = pyo3::types::PyDict::new(py);
                kwargs.set_item("shape", (n, m_k))?;
                kwargs.set_item("dtype", py.import("numpy")?.getattr("complex128")?)?;
                let coo = coo_matrix.as_ref().unwrap().call((arg0,), Some(&kwargs))?;
                let csr = coo.getattr("tocsr")?.call0()?;
                matrices.push(csr);
            } else {
                let mut u_k = ndarray::Array2::<Complex64>::zeros((n, m_k));
                for ((&v, &j), &ell) in valid_cols.iter().zip(j_values.iter()).zip(l_values.iter())
                {
                    if (v as usize) < n && (j as usize) < m_k {
                        let angle = -2.0 * std::f64::consts::PI * (ell as f64) / (e as f64);
                        let phase = Complex64::new(angle.cos(), angle.sin());
                        let val = phase / (orbit_sizes[j as usize] as f64).sqrt();
                        u_k[[v as usize, j as usize]] = val;
                    }
                }
                let py_arr = u_k.into_pyarray(py);
                matrices.push(py_arr.into_any());
            }
        }
        Ok(matrices)
    }

    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (batch_data, batch_indices, batch_indptr, reynolds=false, realize=false))]
    fn extract_batch<'py>(
        &self,
        py: pyo3::Python<'py>,
        batch_data: Vec<PyReadonlyArray1<'py, f64>>,
        batch_indices: Vec<PyReadonlyArray1<'py, i32>>,
        batch_indptr: Vec<PyReadonlyArray1<'py, i32>>,
        reynolds: bool,
        realize: bool,
    ) -> PyResult<
        Vec<
            Vec<(
                Bound<'py, PyArray1<Complex64>>,
                Bound<'py, PyArray1<i32>>,
                Bound<'py, PyArray1<i32>>,
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

        let blocks_results: Vec<Vec<(Vec<Complex64>, Vec<i32>, Vec<i32>, usize)>> = self
            .blocks
            .par_iter()
            .map(|block| {
                let mut block_batch = Vec::with_capacity(num_matrices);
                let m_k = block.dim;

                let mut phase_map = Vec::with_capacity(block.e);
                let e_f64 = block.e as f64;
                for i in 0..block.e {
                    let angle = -2.0 * std::f64::consts::PI * (i as f64) / e_f64;
                    phase_map.push(Complex64::new(angle.cos(), angle.sin()));
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
                                real_vals.push(Complex64::new(v.re, 0.0));
                                real_rows.push(r);
                                real_cols.push(c);
                                real_vals.push(Complex64::new(v.re, 0.0));
                                real_rows.push(r + m);
                                real_cols.push(c + m);
                                // B
                                real_vals.push(Complex64::new(v.im, 0.0));
                                real_rows.push(r);
                                real_cols.push(c + m);
                                // -B
                                real_vals.push(Complex64::new(-v.im, 0.0));
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
                                real_vals.push(Complex64::new(v.re, 0.0));
                                real_rows.push(r);
                                real_cols.push(c);
                                real_vals.push(Complex64::new(v.re, 0.0));
                                real_rows.push(r + m);
                                real_cols.push(c + m);
                                real_vals.push(Complex64::new(v.im, 0.0));
                                real_rows.push(r);
                                real_cols.push(c + m);
                                real_vals.push(Complex64::new(-v.im, 0.0));
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
                            let mut tmp2: HashMap<usize, Complex64> = HashMap::new();

                            for i in 0..vals.len() {
                                let r = rows[i] as usize;
                                let c = cols[i] as usize;
                                let v = vals[i];

                                for &(i_idx, v_dag_val) in &v_dagger_rows[r] {
                                    for &(j_idx, v_val) in &v_cols[c] {
                                        let val = v_dag_val * v * v_val;
                                        let flat_idx = i_idx * m_k + j_idx;
                                        *tmp2
                                            .entry(flat_idx)
                                            .or_insert(Complex64::new(0.0, 0.0)) += val;
                                    }
                                }
                            }

                            for (flat_idx, val) in tmp2 {
                                if val.re.abs() > 1e-12 {
                                    real_vals.push(Complex64::new(val.re, 0.0));
                                    real_rows.push((flat_idx / m_k) as i32);
                                    real_cols.push((flat_idx % m_k) as i32);
                                }
                            }
                        }
                        block_batch.push((real_vals, real_rows, real_cols, m_k));
                    } else if realize {
                        // Realize true but nu2 = 1 and V is identity/real (no complex chi), output real part only
                        let mut real_vals = Vec::with_capacity(vals.len());
                        let mut real_rows = Vec::with_capacity(vals.len());
                        let mut real_cols = Vec::with_capacity(vals.len());
                        for i in 0..vals.len() {
                            if vals[i].re.abs() > 1e-12 {
                                real_vals.push(Complex64::new(vals[i].re, 0.0));
                                real_rows.push(rows[i]);
                                real_cols.push(cols[i]);
                            }
                        }
                        block_batch.push((real_vals, real_rows, real_cols, m_k));
                    } else {
                        block_batch.push((vals, rows, cols, m_k));
                    }
                }
                block_batch
            })
            .collect();

        let mut final_results = Vec::with_capacity(self.blocks.len());
        for block_batch in blocks_results {
            let mut py_block_batch = Vec::with_capacity(num_matrices);
            for (vals, rows, cols, m_k) in block_batch {
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

#[pymethods]
impl SABBlock {
    #[new]
    #[pyo3(signature = (rep_id, dim, e, orbit_reps_flat, orbit_sizes, valid_cols, j_values, l_values, fs_indicator=None, orbit_reps=None, d_k=0))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rep_id: usize,
        dim: usize,
        e: usize,
        orbit_reps_flat: Vec<usize>,
        orbit_sizes: Vec<usize>,
        valid_cols: Vec<u32>,
        j_values: Vec<u32>,
        l_values: Vec<u32>,
        fs_indicator: Option<i32>,
        orbit_reps: Option<Bound<'_, PyAny>>,
        d_k: usize,
    ) -> Self {
        Self {
            rep_id,
            dim,
            e,
            orbit_reps_flat,
            orbit_sizes,
            valid_cols,
            j_values,
            l_values,
            d_k,
            coset_reps: None,
            coset_reps_inv: None,
            fs_indicator,
            v_matrix_data: None,
            txj_cache: None,
            v_cols: None,
            v_dagger_rows: None,
        }
    }

    #[getter]
    fn orbit_reps(&self) -> Vec<usize> {
        vec![]
    }

    #[getter]
    fn get_valid_cols<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.valid_cols.clone().into_pyarray(py)
    }

    #[getter]
    fn get_j_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.j_values.clone().into_pyarray(py)
    }

    #[getter]
    fn get_l_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.l_values.clone().into_pyarray(py)
    }
}
