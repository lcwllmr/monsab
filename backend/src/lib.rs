#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(deprecated)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_return)]
#![allow(clippy::type_complexity)]

use pyo3::prelude::*;

pub mod action;
pub mod builder;
pub mod matrix;
pub mod monomial;
pub mod pc;
pub mod permutation;
pub mod transform;

use action::OrbitLifter;
use builder::{build_sab_blocks, compute_fs_and_t_data};
use matrix::MonomialMatrix;
use pc::{evaluate_word, PcGroup};
use permutation::Permutation;
use transform::{SABBlock, SABTransform};

#[pymodule]
fn _backend(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SABBlock>()?;
    m.add_class::<SABTransform>()?;
    m.add_class::<PcGroup>()?;
    m.add_class::<OrbitLifter>()?;
    m.add_class::<Permutation>()?;
    m.add_class::<MonomialMatrix>()?;
    m.add_function(wrap_pyfunction!(build_sab_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(compute_fs_and_t_data, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_word, m)?)?;
    Ok(())
}
