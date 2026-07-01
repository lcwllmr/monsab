#![allow(clippy::manual_is_multiple_of, clippy::result_unit_err)]
use crate::permutation::Permutation;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};

fn is_prime(n: usize) -> bool {
    if n <= 1 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n % 2 == 0 {
        return false;
    }
    let mut i = 3;
    while i * i <= n {
        if n % i == 0 {
            return false;
        }
        i += 2;
    }
    true
}

pub fn evaluate_word_rust(
    word: &[(usize, usize)],
    generators: &[Permutation],
) -> Result<Permutation, ()> {
    if generators.is_empty() {
        return Ok(Permutation::identity_rust(0));
    }
    let n = generators[0].size();
    let mut result = Permutation::identity_rust(n);
    for &(gen_index, exponent) in word {
        if gen_index >= generators.len() {
            return Err(());
        }
        let term = generators[gen_index].pow_rust(exponent as isize);
        result = result.compose(&term);
    }
    Ok(result)
}

#[pyfunction]
pub fn evaluate_word(
    word: Vec<(usize, usize)>,
    generators: Vec<Bound<'_, PyAny>>,
) -> PyResult<Permutation> {
    let mut perms = Vec::with_capacity(generators.len());
    for gen in generators {
        if let Ok(p) = gen.extract::<Permutation>() {
            perms.push(p);
        } else if let Ok(data) = gen.extract::<Vec<usize>>() {
            perms.push(Permutation::new(data));
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "generators must be Permutation instances or sequences of integers",
            ));
        }
    }
    match evaluate_word_rust(&word, &perms) {
        Ok(res) => Ok(res),
        Err(_) => Err(pyo3::exceptions::PyIndexError::new_err(
            "Generator index out of bounds in evaluate_word",
        )),
    }
}

#[pyclass(get_all, module = "monsab._backend")]
#[derive(Clone)]
pub struct PcGroup {
    pub number_of_generators: usize,
    pub orders: Vec<usize>,
    pub conjugation_exponents: HashMap<(usize, usize), usize>,
    pub power_tails: HashMap<usize, Vec<(usize, usize)>>,
    pub conjugation_tails: HashMap<(usize, usize), Vec<(usize, usize)>>,
    pub generators: Vec<Permutation>,
}

#[pymethods]
impl PcGroup {
    #[new]
    #[pyo3(signature = (number_of_generators, orders, conjugation_exponents, power_tails, conjugation_tails, generators=vec![]))]
    pub fn new(
        number_of_generators: usize,
        orders: Vec<usize>,
        conjugation_exponents: HashMap<(usize, usize), usize>,
        power_tails: HashMap<usize, Vec<(usize, usize)>>,
        conjugation_tails: HashMap<(usize, usize), Vec<(usize, usize)>>,
        generators: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut perms = Vec::with_capacity(generators.len());
        for gen in generators {
            if let Ok(p) = gen.extract::<Permutation>() {
                perms.push(p);
            } else if let Ok(data) = gen.extract::<Vec<usize>>() {
                perms.push(Permutation::new(data));
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "generators must be Permutation instances or sequences of integers",
                ));
            }
        }
        Ok(Self {
            number_of_generators,
            orders,
            conjugation_exponents,
            power_tails,
            conjugation_tails,
            generators: perms,
        })
    }

    #[getter]
    pub fn order(&self) -> usize {
        self.orders.iter().product()
    }

    #[getter]
    pub fn has_only_prime_factors(&self) -> bool {
        self.orders.iter().all(|&r| is_prime(r))
    }

    #[getter]
    pub fn is_normal_series(&self) -> bool {
        for ((j, _k), tail_word) in &self.conjugation_tails {
            for (gen_idx, _) in tail_word {
                if *gen_idx < *j {
                    return false;
                }
            }
        }
        for (k, tail_word) in &self.power_tails {
            for (gen_idx, _) in tail_word {
                if *gen_idx <= *k {
                    return false;
                }
            }
        }
        true
    }

    #[pyo3(signature = (generators))]
    pub fn verify(&self, generators: Vec<Bound<'_, PyAny>>) -> PyResult<bool> {
        let mut perms = Vec::with_capacity(generators.len());
        for gen in generators {
            if let Ok(p) = gen.extract::<Permutation>() {
                perms.push(p);
            } else if let Ok(data) = gen.extract::<Vec<usize>>() {
                perms.push(Permutation::new(data));
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "generators must be Permutation instances or sequences of integers",
                ));
            }
        }
        if self.number_of_generators != perms.len() {
            return Ok(false);
        }
        if perms.is_empty() {
            return Ok(true);
        }

        for k in 0..self.number_of_generators {
            let order = self.orders[k];
            if k >= perms.len() {
                return Ok(false);
            }
            let lhs = perms[k].pow_rust(order as isize);
            let tail = match self.power_tails.get(&k) {
                Some(t) => t.as_slice(),
                None => &[],
            };
            let rhs = match evaluate_word_rust(tail, &perms) {
                Ok(r) => r,
                Err(_) => return Ok(false),
            };
            if lhs != rhs {
                return Ok(false);
            }
        }

        let mut all_pairs: HashSet<(usize, usize)> =
            self.conjugation_exponents.keys().cloned().collect();
        for k in self.conjugation_tails.keys() {
            all_pairs.insert(*k);
        }

        for (j, k) in all_pairs {
            if j >= perms.len() || k >= perms.len() {
                return Ok(false);
            }
            let c = self
                .conjugation_exponents
                .get(&(j, k))
                .cloned()
                .unwrap_or(0);
            let tail = match self.conjugation_tails.get(&(j, k)) {
                Some(t) => t.as_slice(),
                None => &[],
            };
            let g_k_inv = perms[k].invert();
            let lhs = g_k_inv.compose(&perms[j]).compose(&perms[k]);
            let rhs_first = perms[j].pow_rust(c as isize);
            let rhs_tail = match evaluate_word_rust(tail, &perms) {
                Ok(r) => r,
                Err(_) => return Ok(false),
            };
            let rhs = rhs_first.compose(&rhs_tail);
            if lhs != rhs {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(5));
        assert!(is_prime(13));
    }
}
