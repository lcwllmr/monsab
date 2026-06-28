use pyo3::prelude::*;

#[pyfunction]
fn compute_fast(a: usize, b: usize) -> PyResult<usize> {
    Ok(a + b)
}

#[pymodule]
fn _backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_fast, m)?)?;
    Ok(())
}
