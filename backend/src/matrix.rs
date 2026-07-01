use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple, PyType};

#[pyclass(frozen, eq, hash, module = "monsab._backend")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MonomialMatrix {
    pub perm: Vec<usize>,
    pub vals: Vec<usize>,
    pub e: usize,
}

impl MonomialMatrix {
    pub fn validate(perm: &[usize], vals: &[usize], e: isize) -> PyResult<()> {
        if e <= 0 {
            return Err(PyValueError::new_err(
                "Group exponent e must be a positive integer.",
            ));
        }
        if perm.len() != vals.len() {
            return Err(PyValueError::new_err(
                "perm and vals must have equal length.",
            ));
        }
        let dim = perm.len();
        let mut seen = vec![false; dim];
        for &p in perm {
            if p >= dim || seen[p] {
                return Err(PyValueError::new_err(
                    "perm must be a permutation of 0..d-1.",
                ));
            }
            seen[p] = true;
        }
        Ok(())
    }

    pub fn new_unchecked(perm: Vec<usize>, vals: Vec<usize>, e: usize) -> Self {
        Self { perm, vals, e }
    }

    pub fn identity_rust(dim: usize, e: usize) -> Self {
        Self {
            perm: (0..dim).collect(),
            vals: vec![0; dim],
            e,
        }
    }

    pub fn direct_sum(&self, other: &Self) -> PyResult<Self> {
        if self.e != other.e {
            return Err(PyValueError::new_err(
                "Cannot add monomial matrices with different exponents e.",
            ));
        }
        let dim = self.perm.len();
        let mut new_perm = Vec::with_capacity(dim + other.perm.len());
        new_perm.extend_from_slice(&self.perm);
        for &p in &other.perm {
            new_perm.push(p + dim);
        }
        let mut new_vals = Vec::with_capacity(self.vals.len() + other.vals.len());
        new_vals.extend_from_slice(&self.vals);
        new_vals.extend_from_slice(&other.vals);
        Ok(Self {
            perm: new_perm,
            vals: new_vals,
            e: self.e,
        })
    }

    pub fn compose(&self, other: &Self) -> PyResult<Self> {
        if self.e != other.e {
            return Err(PyValueError::new_err(
                "Cannot multiply monomial matrices with different exponents e.",
            ));
        }
        if self.perm.len() != other.perm.len() {
            return Err(PyValueError::new_err(
                "Cannot multiply monomial matrices with different dimensions.",
            ));
        }
        let dim = self.perm.len();
        let mut new_perm = Vec::with_capacity(dim);
        let mut new_vals = Vec::with_capacity(dim);
        for i in 0..dim {
            let j = self.perm[i];
            new_perm.push(other.perm[j]);
            new_vals.push((self.vals[i] + other.vals[j]) % self.e);
        }
        Ok(Self {
            perm: new_perm,
            vals: new_vals,
            e: self.e,
        })
    }

    pub fn invert(&self) -> Self {
        let dim = self.perm.len();
        let mut inv_perm = vec![0; dim];
        let mut inv_vals = vec![0; dim];
        for i in 0..dim {
            let j = self.perm[i];
            inv_perm[j] = i;
            inv_vals[j] = (self.e - (self.vals[i] % self.e)) % self.e;
        }
        Self {
            perm: inv_perm,
            vals: inv_vals,
            e: self.e,
        }
    }

    pub fn pow_rust(&self, mut p: isize) -> PyResult<Self> {
        if p == 0 {
            return Ok(Self::identity_rust(self.perm.len(), self.e));
        }
        let base = if p < 0 {
            p = -p;
            self.invert()
        } else {
            self.clone()
        };

        let mut result = Self::identity_rust(self.perm.len(), self.e);
        let mut factor = base;
        let mut exp = p as usize;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.compose(&factor)?;
            }
            factor = factor.compose(&factor)?;
            exp >>= 1;
        }
        Ok(result)
    }
}

#[pymethods]
impl MonomialMatrix {
    #[new]
    pub fn py_new(perm: Vec<usize>, vals: Vec<usize>, e: isize) -> PyResult<Self> {
        Self::validate(&perm, &vals, e)?;
        Ok(Self {
            perm,
            vals,
            e: e as usize,
        })
    }

    #[classmethod]
    #[pyo3(name = "identity")]
    pub fn py_identity(_cls: &Bound<'_, PyType>, dim: isize, e: isize) -> PyResult<Self> {
        if dim < 0 {
            return Err(PyValueError::new_err(
                "Matrix dimension must be a non-negative integer.",
            ));
        }
        if e <= 0 {
            return Err(PyValueError::new_err(
                "Group exponent e must be a positive integer.",
            ));
        }
        Ok(Self::identity_rust(dim as usize, e as usize))
    }

    #[getter]
    pub fn perm<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.perm)
    }

    #[getter]
    pub fn vals<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.vals)
    }

    #[getter]
    pub fn e(&self) -> usize {
        self.e
    }

    #[getter]
    pub fn dim(&self) -> usize {
        self.perm.len()
    }

    pub fn __add__(&self, other: &Self) -> PyResult<Self> {
        self.direct_sum(other)
    }

    pub fn __matmul__(&self, other: &Self) -> PyResult<Self> {
        self.compose(other)
    }

    pub fn __pow__(
        &self,
        p: &Bound<'_, PyAny>,
        _modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        if let Ok(exp) = p.extract::<isize>() {
            self.pow_rust(exp)
        } else {
            Err(PyValueError::new_err("Power must be an integer."))
        }
    }

    pub fn inverse(&self) -> Self {
        self.invert()
    }

    pub fn __repr__(&self) -> String {
        let perm_parts: Vec<String> = self.perm.iter().map(|x| x.to_string()).collect();
        let perm_str = if perm_parts.len() == 1 {
            format!("({},)", perm_parts[0])
        } else {
            format!("({})", perm_parts.join(", "))
        };

        let vals_parts: Vec<String> = self.vals.iter().map(|x| x.to_string()).collect();
        let vals_str = if vals_parts.len() == 1 {
            format!("({},)", vals_parts[0])
        } else {
            format!("({})", vals_parts.join(", "))
        };

        format!(
            "MonomialMatrix(perm={}, vals={}, e={})",
            perm_str, vals_str, self.e
        )
    }

    pub fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn __getnewargs__(&self) -> (Vec<usize>, Vec<usize>, usize) {
        (self.perm.clone(), self.vals.clone(), self.e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_and_multiply() {
        let a = MonomialMatrix::new_unchecked(vec![1, 0, 2], vec![3, 4, 5], 7);
        let b = MonomialMatrix::new_unchecked(vec![2, 1, 0], vec![6, 1, 2], 7);
        let prod = a.compose(&b).unwrap();
        assert_eq!(prod.perm, vec![1, 2, 0]);
        assert_eq!(prod.vals, vec![4, 3, 0]);
        assert_eq!(prod.e, 7);
    }

    #[test]
    fn test_inverse_and_direct_sum() {
        let m = MonomialMatrix::new_unchecked(vec![2, 0, 1, 3], vec![9, 4, 12, 5], 7);
        let inv = m.invert();
        let id = MonomialMatrix::identity_rust(4, 7);
        assert_eq!(m.compose(&inv).unwrap(), id);
        assert_eq!(inv.compose(&m).unwrap(), id);

        let a = MonomialMatrix::new_unchecked(vec![1, 0], vec![3, 4], 5);
        let b = MonomialMatrix::new_unchecked(vec![0, 1], vec![2, 1], 5);
        let ds = a.direct_sum(&b).unwrap();
        assert_eq!(ds.perm, vec![1, 0, 2, 3]);
        assert_eq!(ds.vals, vec![3, 4, 2, 1]);
    }
}
