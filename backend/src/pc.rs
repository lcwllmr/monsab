use pyo3::prelude::*;
use std::collections::HashMap;

#[pyclass(get_all, subclass, dict)]
#[derive(Clone)]
pub struct PcGroup {
    pub number_of_generators: usize,
    pub orders: Vec<usize>,
    pub conjugation_exponents: HashMap<(usize, usize), usize>,
    pub power_tails: HashMap<usize, Vec<(usize, usize)>>,
    pub conjugation_tails: HashMap<(usize, usize), Vec<(usize, usize)>>,
    pub generators: Vec<Vec<u32>>,
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
        generators: Vec<Vec<u32>>,
    ) -> Self {
        Self {
            number_of_generators,
            orders,
            conjugation_exponents,
            power_tails,
            conjugation_tails,
            generators,
        }
    }
}
