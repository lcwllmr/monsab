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

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
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

impl PcGroup {
    pub fn verify_rust(&self, perms: &[Permutation]) -> bool {
        if self.number_of_generators != perms.len() {
            return false;
        }
        if perms.is_empty() {
            return true;
        }

        for k in 0..self.number_of_generators {
            let order = self.orders[k];
            if k >= perms.len() {
                return false;
            }
            let lhs = perms[k].pow_rust(order as isize);
            let tail = match self.power_tails.get(&k) {
                Some(t) => t.as_slice(),
                None => &[],
            };
            let rhs = match evaluate_word_rust(tail, perms) {
                Ok(r) => r,
                Err(_) => return false,
            };
            if lhs != rhs {
                return false;
            }
        }

        let mut all_pairs: HashSet<(usize, usize)> =
            self.conjugation_exponents.keys().cloned().collect();
        for k in self.conjugation_tails.keys() {
            all_pairs.insert(*k);
        }

        for (j, k) in all_pairs {
            if j >= perms.len() || k >= perms.len() {
                return false;
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
            let rhs_tail = match evaluate_word_rust(tail, perms) {
                Ok(r) => r,
                Err(_) => return false,
            };
            let rhs = rhs_first.compose(&rhs_tail);
            if lhs != rhs {
                return false;
            }
        }

        true
    }
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
        Ok(self.verify_rust(&perms))
    }

    #[pyo3(signature = (generators))]
    pub fn test_generators(&self, generators: Vec<Bound<'_, PyAny>>) -> PyResult<bool> {
        self.verify(generators)
    }

    #[pyo3(signature = (generators))]
    pub fn with_generators(&self, generators: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
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
        let mut clone = self.clone();
        clone.generators = perms;
        Ok(clone)
    }

    pub fn test_consistency(&self) -> bool {
        if self.number_of_generators != self.orders.len() {
            return false;
        }
        if self.orders.iter().any(|&r| !is_prime(r)) {
            return false;
        }

        for (k, tail) in &self.power_tails {
            if *k >= self.number_of_generators {
                return false;
            }
            if tail.iter().any(|&(gen_idx, exp)| {
                gen_idx <= *k
                    || gen_idx >= self.number_of_generators
                    || exp == 0
                    || exp >= self.orders[gen_idx]
            }) {
                return false;
            }
        }
        for ((j, k), tail) in &self.conjugation_tails {
            if *j >= *k || *k >= self.number_of_generators {
                return false;
            }
            if tail.iter().any(|&(gen_idx, exp)| {
                gen_idx <= *j
                    || gen_idx >= self.number_of_generators
                    || exp == 0
                    || exp >= self.orders[gen_idx]
            }) {
                return false;
            }
        }
        for (&(j, k), &c) in &self.conjugation_exponents {
            if j >= k || k >= self.number_of_generators {
                return false;
            }
            if gcd(c, self.orders[j]) != 1 {
                return false;
            }
        }
        if !self.generators.is_empty() && !self.verify_rust(&self.generators) {
            return false;
        }
        true
    }

    pub fn test_supersolvable(&self) -> bool {
        if !self.test_consistency() || !self.is_normal_series() {
            return false;
        }
        for j in 0..self.number_of_generators {
            for k in (j + 1)..self.number_of_generators {
                let c = self
                    .conjugation_exponents
                    .get(&(j, k))
                    .cloned()
                    .unwrap_or(1);
                if c % self.orders[j] != 1 % self.orders[j] {
                    return false;
                }
            }
        }
        true
    }

    pub fn __repr__(&self) -> String {
        let m = self.number_of_generators;
        if m == 0 {
            return "< | >".to_string();
        }
        let gens_str = (0..m)
            .map(|i| format!("g{}", i))
            .collect::<Vec<_>>()
            .join(", ");
        let mut rels = Vec::new();
        for k in 0..m {
            let r = self.orders[k];
            let lhs = format!("g{}^{}", k, r);
            let tail = match self.power_tails.get(&k) {
                Some(t) => t.as_slice(),
                None => &[],
            };
            let rhs = if tail.is_empty() {
                "1".to_string()
            } else {
                tail.iter()
                    .map(|&(idx, exp)| {
                        if exp == 1 {
                            format!("g{}", idx)
                        } else {
                            format!("g{}^{}", idx, exp)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" * ")
            };
            rels.push(format!("{} = {}", lhs, rhs));
        }
        for j in 0..m {
            for k in (j + 1)..m {
                let c = self
                    .conjugation_exponents
                    .get(&(j, k))
                    .cloned()
                    .unwrap_or(0);
                let tail = match self.conjugation_tails.get(&(j, k)) {
                    Some(t) => t.as_slice(),
                    None => &[],
                };
                if (!self.conjugation_exponents.contains_key(&(j, k)) && tail.is_empty())
                    || (c == 1 && tail.is_empty())
                {
                    continue;
                }
                let lhs = format!("g{}^-1 * g{} * g{}", k, j, k);
                let mut rhs_terms = Vec::new();
                if c == 1 {
                    rhs_terms.push(format!("g{}", j));
                } else if c > 1 {
                    rhs_terms.push(format!("g{}^{}", j, c));
                }
                for &(idx, exp) in tail {
                    if exp == 1 {
                        rhs_terms.push(format!("g{}", idx));
                    } else {
                        rhs_terms.push(format!("g{}^{}", idx, exp));
                    }
                }
                let rhs = if rhs_terms.is_empty() {
                    "1".to_string()
                } else {
                    rhs_terms.join(" * ")
                };
                rels.push(format!("{} = {}", lhs, rhs));
            }
        }
        format!("< {} | {} >", gens_str, rels.join(", "))
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
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
