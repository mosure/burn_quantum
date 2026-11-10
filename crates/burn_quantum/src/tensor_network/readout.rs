//! Streaming observables and explicit bounded dense export.
use super::{MpsState, tensor::*};
use crate::{Error, Matrix, Register, ResourceLimits, Result, StateVector};
use num_complex::Complex64;
impl MpsState {
    /// Product-operator expectation, supporting nonadjacent Pauli strings without
    /// materializing a dense observable or a dense state.
    pub fn expectation_product(&self, operators: &[(usize, Matrix)]) -> Result<Complex64> {
        for (i, (w, m)) in operators.iter().enumerate() {
            self.limits.tensor(crate::error::checked_product([
                m.dimension(),
                m.dimension(),
            ])?)?;
            if *w >= self.sites.len()
                || operators[..i].iter().any(|(v, _)| v == w)
                || m.dimension() != self.sites[*w].dims()[1]
            {
                return Err(Error::Shape("invalid MPS product operator".into()));
            }
        }
        let mut env = from(&[1.0.into()], [1, 1], &self.device(), self.sites[0].dtype())?;
        for (index, site) in self.sites.iter().enumerate() {
            let matrix = operators.iter().find(|(w, _)| *w == index).map(|(_, m)| m);
            env = environment_step(&env, site, matrix, self.limits)?;
        }
        Ok(data(&env)?[0])
    }
    /// Exact local Hermitian expectation without expanding the whole chain.
    /// A dense k-site observable is decomposed into bounded product matrix units.
    pub fn expectation(&self, observable: &crate::Observable) -> Result<f64> {
        let wires = observable.wires();
        let mut dims = Vec::with_capacity(wires.len());
        for &w in wires {
            if w >= self.sites.len() {
                return Err(Error::Wires("MPS observable wire".into()));
            }
            dims.push(self.sites[w].dims()[1]);
        }
        let register = Register::new(dims.clone())?;
        let n = register.size();
        if observable.matrix().dimension() != n {
            return Err(Error::Shape("MPS observable dimension".into()));
        }
        self.limits.tensor(crate::error::checked_product([n, n])?)?;
        let mut result = Complex64::new(0., 0.);
        for a in 0..n {
            for b in 0..n {
                let weight = observable.matrix().entries()[a * n + b];
                if weight.norm_sqr() == 0. {
                    continue;
                }
                let ad = register.decode(a)?;
                let bd = register.decode(b)?;
                let mut product = Vec::with_capacity(wires.len());
                for j in 0..wires.len() {
                    let d = dims[j];
                    let mut entries = vec![0.; d * d];
                    entries[ad[j] * d + bd[j]] = 1.;
                    product.push((wires[j], Matrix::real(d, entries)?));
                }
                result += weight * self.expectation_product(&product)?;
            }
        }
        Ok(result.re)
    }
    /// Explicit bounded dense export, for oracles/interoperability only.
    pub fn to_dense(&self, limits: ResourceLimits) -> Result<StateVector> {
        let register = Register::new(self.dimensions())?;
        limits.check_state(1, register.size())?;
        let first = &self.sites[0];
        let [_, d, r] = first.dims();
        let mut state = reshape(first, [d, r]);
        for site in self.sites.iter().skip(1) {
            let [l, d, r] = site.dims();
            let p = state.dims()[0];
            let count = crate::error::checked_product([p, d, r])?;
            limits.check_state(1, count)?;
            state = reshape(&mm(&state, &reshape(site, [l, d * r])), [p * d, r]);
        }
        StateVector::from_amplitudes(register, reshape(&state, [1, state.dims()[0]]), limits)
    }
}
