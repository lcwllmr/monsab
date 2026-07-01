use crate::pc::PcGroup;
use crate::permutation::Permutation;
use fxhash::FxHashMap;
use pyo3::prelude::*;

pub trait Monomial<const D: usize>: Clone + Ord + Eq + std::hash::Hash {
    fn normalize(&mut self);
    fn permute(&self, p: &Permutation) -> Self;
    fn from_slice(slice: &[u32]) -> Self;
    fn to_vec(&self) -> Vec<u32>;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StandardMonomial<const D: usize> {
    pub vars: [u32; D],
}

impl<const D: usize> Monomial<D> for StandardMonomial<D> {
    #[inline(always)]
    fn normalize(&mut self) {
        self.vars.sort_unstable();
    }

    #[inline(always)]
    fn permute(&self, p: &Permutation) -> Self {
        let mut next = [0; D];
        for i in 0..D {
            next[i] = p.apply(self.vars[i] as usize) as u32;
        }
        let mut m = StandardMonomial { vars: next };
        m.normalize();
        m
    }

    #[inline(always)]
    fn from_slice(slice: &[u32]) -> Self {
        let mut vars = [0; D];
        vars.copy_from_slice(slice);
        let mut m = StandardMonomial { vars };
        m.normalize();
        m
    }

    #[inline(always)]
    fn to_vec(&self) -> Vec<u32> {
        self.vars.to_vec()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SquarefreeMonomial<const D: usize> {
    pub vars: [u32; D],
}

impl<const D: usize> Monomial<D> for SquarefreeMonomial<D> {
    #[inline(always)]
    fn normalize(&mut self) {
        self.vars.sort_unstable();
    }

    #[inline(always)]
    fn permute(&self, p: &Permutation) -> Self {
        let mut next = [0; D];
        for i in 0..D {
            next[i] = p.apply(self.vars[i] as usize) as u32;
        }
        let mut m = SquarefreeMonomial { vars: next };
        m.normalize();
        // Since the permutation is a bijection and the input was squarefree,
        // the output is guaranteed to be squarefree.
        m
    }

    #[inline(always)]
    fn from_slice(slice: &[u32]) -> Self {
        let mut vars = [0; D];
        vars.copy_from_slice(slice);
        let mut m = SquarefreeMonomial { vars };
        m.normalize();
        m
    }

    #[inline(always)]
    fn to_vec(&self) -> Vec<u32> {
        self.vars.to_vec()
    }
}

pub struct InternalOrbitLifter<M: Monomial<D>, const D: usize> {
    group: PcGroup,
    cache: Vec<FxHashMap<M, M>>,
}

impl<M: Monomial<D>, const D: usize> InternalOrbitLifter<M, D> {
    pub fn new(group: PcGroup) -> Self {
        let levels = group.orders.len() + 1;
        let mut cache = Vec::with_capacity(levels);
        for _ in 0..levels {
            cache.push(FxHashMap::default());
        }
        Self { group, cache }
    }

    pub fn clear_cache(&mut self) {
        for level_cache in &mut self.cache {
            level_cache.clear();
        }
    }

    pub fn canonicalize(&mut self, m: &M, level: usize) -> M {
        Self::canonicalize_internal(&self.group, &mut self.cache, m, level)
    }

    fn canonicalize_internal(
        group: &PcGroup,
        cache: &mut [FxHashMap<M, M>],
        m: &M,
        level: usize,
    ) -> M {
        if level == 0 {
            let mut base = m.clone();
            base.normalize();
            return base;
        }

        if let Some(cached) = cache[level].get(m) {
            return cached.clone();
        }

        let p = group.orders[level - 1];
        let gen = &group.generators[level - 1];

        let mut best = Self::canonicalize_internal(group, cache, m, level - 1);
        let mut current = m.clone();

        for _ in 1..p {
            current = current.permute(gen);
            let candidate = Self::canonicalize_internal(group, cache, &current, level - 1);
            if candidate < best {
                best = candidate;
            }
        }

        cache[level].insert(m.clone(), best.clone());
        best
    }
}

enum LifterState {
    Std1(InternalOrbitLifter<StandardMonomial<1>, 1>),
    Std2(InternalOrbitLifter<StandardMonomial<2>, 2>),
    Std3(InternalOrbitLifter<StandardMonomial<3>, 3>),
    Std4(InternalOrbitLifter<StandardMonomial<4>, 4>),
    Sq1(InternalOrbitLifter<SquarefreeMonomial<1>, 1>),
    Sq2(InternalOrbitLifter<SquarefreeMonomial<2>, 2>),
    Sq3(InternalOrbitLifter<SquarefreeMonomial<3>, 3>),
    Sq4(InternalOrbitLifter<SquarefreeMonomial<4>, 4>),
}

#[pyclass]
pub struct OrbitLifter {
    state: LifterState,
}

#[pymethods]
impl OrbitLifter {
    #[new]
    pub fn new(group: PcGroup, d: usize, is_squarefree: bool) -> PyResult<Self> {
        let state = match (d, is_squarefree) {
            (1, false) => LifterState::Std1(InternalOrbitLifter::new(group)),
            (2, false) => LifterState::Std2(InternalOrbitLifter::new(group)),
            (3, false) => LifterState::Std3(InternalOrbitLifter::new(group)),
            (4, false) => LifterState::Std4(InternalOrbitLifter::new(group)),
            (1, true) => LifterState::Sq1(InternalOrbitLifter::new(group)),
            (2, true) => LifterState::Sq2(InternalOrbitLifter::new(group)),
            (3, true) => LifterState::Sq3(InternalOrbitLifter::new(group)),
            (4, true) => LifterState::Sq4(InternalOrbitLifter::new(group)),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported degree D > 4 or D = 0",
                ))
            }
        };
        Ok(Self { state })
    }

    pub fn clear_cache(&mut self) {
        match &mut self.state {
            LifterState::Std1(l) => l.clear_cache(),
            LifterState::Std2(l) => l.clear_cache(),
            LifterState::Std3(l) => l.clear_cache(),
            LifterState::Std4(l) => l.clear_cache(),
            LifterState::Sq1(l) => l.clear_cache(),
            LifterState::Sq2(l) => l.clear_cache(),
            LifterState::Sq3(l) => l.clear_cache(),
            LifterState::Sq4(l) => l.clear_cache(),
        }
    }

    pub fn canonicalize(&mut self, monomial: Vec<u32>) -> PyResult<Vec<u32>> {
        match &mut self.state {
            LifterState::Std1(l) => {
                let m = StandardMonomial::from_slice(&monomial);
                let level = l.group.orders.len();
                Ok(l.canonicalize(&m, level).to_vec())
            }
            LifterState::Std2(l) => {
                let m = StandardMonomial::from_slice(&monomial);
                let level = l.group.orders.len();
                Ok(l.canonicalize(&m, level).to_vec())
            }
            LifterState::Std3(l) => {
                let m = StandardMonomial::from_slice(&monomial);
                let level = l.group.orders.len();
                Ok(l.canonicalize(&m, level).to_vec())
            }
            LifterState::Std4(l) => {
                let m = StandardMonomial::from_slice(&monomial);
                let level = l.group.orders.len();
                Ok(l.canonicalize(&m, level).to_vec())
            }
            LifterState::Sq1(l) => {
                let m = SquarefreeMonomial::from_slice(&monomial);
                let level = l.group.orders.len();
                Ok(l.canonicalize(&m, level).to_vec())
            }
            LifterState::Sq2(l) => {
                let m = SquarefreeMonomial::from_slice(&monomial);
                let level = l.group.orders.len();
                Ok(l.canonicalize(&m, level).to_vec())
            }
            LifterState::Sq3(l) => {
                let m = SquarefreeMonomial::from_slice(&monomial);
                let level = l.group.orders.len();
                Ok(l.canonicalize(&m, level).to_vec())
            }
            LifterState::Sq4(l) => {
                let m = SquarefreeMonomial::from_slice(&monomial);
                let level = l.group.orders.len();
                Ok(l.canonicalize(&m, level).to_vec())
            }
        }
    }
}
