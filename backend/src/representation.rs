use num_complex::Complex64;
use pyo3::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

#[derive(Clone)]
#[pyclass(name = "MonomialRepresentation", module = "monsab._backend")]
pub struct MonomialRepresentation {
    #[pyo3(get)]
    pub id: usize,
    #[pyo3(get)]
    pub dim: usize,
    #[pyo3(get)]
    pub e: usize,
    #[pyo3(get)]
    pub conjugate_id: usize,
    #[pyo3(get)]
    pub fs_indicator: Option<i32>,
    #[pyo3(get)]
    pub v_matrix: Option<Vec<usize>>,
}

#[pymethods]
impl MonomialRepresentation {
    #[new]
    #[pyo3(signature = (id, dim, e, conjugate_id, fs_indicator=None, v_matrix=None))]
    pub fn new(
        id: usize,
        dim: usize,
        e: usize,
        conjugate_id: usize,
        fs_indicator: Option<i32>,
        v_matrix: Option<Vec<usize>>,
    ) -> Self {
        Self {
            id,
            dim,
            e,
            conjugate_id,
            fs_indicator,
            v_matrix,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MonomialRepresentation(id={}, dim={}, e={}, conjugate_id={}, fs_indicator={:?}, v_matrix={:?})",
            self.id, self.dim, self.e, self.conjugate_id, self.fs_indicator, self.v_matrix
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.id == other.id
            && self.dim == other.dim
            && self.e == other.e
            && self.conjugate_id == other.conjugate_id
            && self.fs_indicator == other.fs_indicator
            && self.v_matrix == other.v_matrix
    }
}

#[derive(Clone)]
#[pyclass(name = "MonomialRepresentationBundle", module = "monsab._backend")]
pub struct MonomialRepresentationBundle {
    #[pyo3(get)]
    pub representations: Vec<MonomialRepresentation>,
    #[pyo3(get)]
    pub paths_dict: HashMap<usize, Vec<(usize, Vec<(usize, usize)>, usize)>>,
    #[pyo3(get)]
    pub fs_indicators: HashMap<usize, i32>,
    #[pyo3(get)]
    pub v_matrices: HashMap<usize, Vec<usize>>,
    #[pyo3(get)]
    pub realize_skip_reps: HashSet<usize>,
    #[pyo3(get)]
    pub e: usize,
}

#[pymethods]
impl MonomialRepresentationBundle {
    #[new]
    #[pyo3(signature = (representations, paths_dict, fs_indicators, v_matrices, realize_skip_reps, e))]
    pub fn new(
        representations: Vec<MonomialRepresentation>,
        paths_dict: HashMap<usize, Vec<(usize, Vec<(usize, usize)>, usize)>>,
        fs_indicators: HashMap<usize, i32>,
        v_matrices: HashMap<usize, Vec<usize>>,
        realize_skip_reps: HashSet<usize>,
        e: usize,
    ) -> Self {
        Self {
            representations,
            paths_dict,
            fs_indicators,
            v_matrices,
            realize_skip_reps,
            e,
        }
    }
}

pub fn hash_vec(v: &[usize]) -> u64 {
    let mut hasher = DefaultHasher::new();
    v.hash(&mut hasher);
    hasher.finish()
}

#[pyfunction]
#[pyo3(signature = (g_gens, n, e, h_visited_all, h_gens_fwd_all, char_phases_rep_all))]
pub fn compute_fs_and_t_data(
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
