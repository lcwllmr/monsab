use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple, PyType};
use std::hash::{Hash, Hasher};

#[pyclass(frozen, eq, hash, module = "monsab._backend")]
#[derive(Clone, Debug, Eq)]
pub struct Permutation {
    pub data: Vec<usize>,
}

impl Permutation {
    pub fn new(data: Vec<usize>) -> Self {
        Self { data }
    }

    pub fn identity_rust(n: usize) -> Self {
        Self {
            data: (0..n).collect(),
        }
    }

    #[inline(always)]
    pub fn apply(&self, i: usize) -> usize {
        if i < self.data.len() {
            self.data[i]
        } else {
            i
        }
    }

    pub fn compose(&self, other: &Self) -> Self {
        let n = std::cmp::max(self.data.len(), other.data.len());
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            data.push(self.apply(other.apply(i)));
        }
        Self { data }
    }

    pub fn invert(&self) -> Self {
        let n = self.data.len();
        let mut inv = vec![0; n];
        for (i, &val) in self.data.iter().enumerate() {
            if val < n {
                inv[val] = i;
            }
        }
        Self { data: inv }
    }

    pub fn pow_rust(&self, mut p: isize) -> Self {
        if p == 0 {
            return Self::identity_rust(self.data.len());
        }
        let base = if p < 0 {
            p = -p;
            self.invert()
        } else {
            self.clone()
        };

        let mut result = Self::identity_rust(self.data.len());
        let mut factor = base;
        let mut exp = p as usize;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.compose(&factor);
            }
            factor = factor.compose(&factor);
            exp >>= 1;
        }
        result
    }

    pub fn normalized_slice(&self) -> &[usize] {
        let mut len = self.data.len();
        while len > 0 && self.data[len - 1] == len - 1 {
            len -= 1;
        }
        &self.data[0..len]
    }
}

impl PartialEq for Permutation {
    fn eq(&self, other: &Self) -> bool {
        self.normalized_slice() == other.normalized_slice()
    }
}

impl Hash for Permutation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.normalized_slice().hash(state);
    }
}

#[pymethods]
impl Permutation {
    #[new]
    pub fn py_new(data: Vec<usize>) -> Self {
        Self::new(data)
    }

    #[classmethod]
    #[pyo3(name = "identity")]
    pub fn py_identity(_cls: &Bound<'_, PyType>, n: usize) -> Self {
        Self::identity_rust(n)
    }

    #[getter]
    pub fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.data)
    }

    #[getter]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn __getitem__(&self, index: isize) -> usize {
        let n = self.data.len() as isize;
        let idx = if index < 0 { n + index } else { index };
        if idx >= 0 && (idx as usize) < self.data.len() {
            self.data[idx as usize]
        } else if idx >= 0 {
            idx as usize
        } else {
            0
        }
    }

    pub fn __mul__(&self, other: &Self) -> Self {
        self.compose(other)
    }

    pub fn __pow__(
        &self,
        p: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if let Ok(exp) = p.extract::<isize>() {
            Ok(self.pow_rust(exp))
        } else {
            Err(PyValueError::new_err("Power must be an integer."))
        }
    }

    pub fn __invert__(&self) -> Self {
        self.invert()
    }

    pub fn __repr__(&self) -> String {
        let parts: Vec<String> = self.data.iter().map(|x| x.to_string()).collect();
        if parts.len() == 1 {
            format!("Permutation(data=({},))", parts[0])
        } else {
            format!("Permutation(data=({}))", parts.join(", "))
        }
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn __getnewargs__(&self) -> (Vec<usize>,) {
        (self.data.clone(),)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_and_apply() {
        let p = Permutation::identity_rust(5);
        for i in 0..10 {
            assert_eq!(p.apply(i), i);
        }
    }

    #[test]
    fn test_compose_and_invert() {
        let p1 = Permutation::new(vec![1, 2, 0]);
        let p2 = Permutation::new(vec![1, 0, 2]);
        let comp = p1.compose(&p2);
        assert_eq!(comp.data, vec![2, 1, 0]);

        let inv = p1.invert();
        assert_eq!(inv.data, vec![2, 0, 1]);
        assert_eq!(p1.compose(&inv), Permutation::identity_rust(3));
    }

    #[test]
    fn test_pow() {
        let p = Permutation::new(vec![1, 2, 0]);
        assert_eq!(p.pow_rust(3), Permutation::identity_rust(3));
        assert_eq!(p.pow_rust(-1), p.invert());
    }

    #[test]
    fn test_normalized_equality() {
        let p1 = Permutation::new(vec![0, 1]);
        let p2 = Permutation::new(vec![0, 1, 2]);
        assert_eq!(p1, p2);
    }
}
