//! Exact routing, canonical gauges and controlled Schmidt truncation.
use super::{MpsState, limit, state::accumulate, tensor::*};
use crate::{ComplexTensor, Error, Matrix, Result, Unitary};
use num_complex::Complex64;
impl MpsState {
    pub(super) fn move_center(&mut self, target: usize) -> Result<()> {
        if target >= self.sites.len() {
            return Err(Error::Wires("MPS center outside chain".into()));
        }
        while self.center < target {
            let i = self.center;
            let (a, b) = move_right(&self.sites[i], &self.sites[i + 1], self.limits)?;
            self.sites[i] = a;
            self.sites[i + 1] = b;
            self.center += 1;
        }
        while self.center > target {
            let i = self.center - 1;
            let (a, b) = move_left(&self.sites[i], &self.sites[i + 1], self.limits)?;
            self.sites[i] = a;
            self.sites[i + 1] = b;
            self.center -= 1;
        }
        Ok(())
    }
    /// Apply any bounded dense unitary to ordered, possibly nonadjacent qudits.
    /// SWAP routing is undone after execution. All failures leave self unchanged.
    pub fn apply(&mut self, wires: &[usize], gate: &Unitary) -> Result<()> {
        if wires.is_empty()
            || wires
                .iter()
                .enumerate()
                .any(|(i, w)| *w >= self.sites.len() || wires[..i].contains(w))
        {
            return Err(Error::Wires("invalid MPS gate targets".into()));
        }
        let dimension =
            crate::error::checked_product(wires.iter().map(|w| self.sites[*w].dims()[1]))?;
        if dimension != gate.matrix().dimension() {
            return Err(Error::Shape("MPS gate dimension differs".into()));
        }
        self.limits
            .tensor(crate::error::checked_product([dimension, dimension])?)?;
        let mut next = self.clone();
        let mut order: (Vec<usize>, Vec<usize>) = ((0..self.sites.len()).collect(), Vec::new());
        let start = *wires.iter().min().unwrap();
        for (j, w) in wires.iter().enumerate() {
            let mut pos = order.0.iter().position(|v| v == w).unwrap();
            while pos > start + j {
                next.swap_sites(pos - 1)?;
                order.0.swap(pos - 1, pos);
                order.1.push(pos - 1);
                pos -= 1;
            }
        }
        next.block(start, wires.len(), gate.matrix(), None)?;
        for i in order.1.into_iter().rev() {
            next.swap_sites(i)?;
        }
        limit(
            "MPS persistent state",
            next.storage_elements(),
            next.limits.max_state_elements,
        )?;
        *self = next;
        Ok(())
    }
    fn swap_sites(&mut self, i: usize) -> Result<()> {
        let d = self.sites[i].dims()[1];
        let e = self.sites[i + 1].dims()[1];
        let n = crate::error::checked_product([d, e])?;
        self.limits.tensor(crate::error::checked_product([n, n])?)?;
        let mut v = vec![Complex64::new(0., 0.); n * n];
        for a in 0..d {
            for b in 0..e {
                v[(b * d + a) * n + a * e + b] = 1.0.into();
            }
        }
        self.block(i, 2, &Matrix::new(n, v)?, Some(vec![e, d]))
    }
    pub(super) fn block(
        &mut self,
        start: usize,
        length: usize,
        gate: &Matrix,
        output_dims: Option<Vec<usize>>,
    ) -> Result<()> {
        self.move_center(start)?;
        let dims = output_dims.unwrap_or_else(|| {
            self.sites[start..start + length]
                .iter()
                .map(|s| s.dims()[1])
                .collect()
        });
        let l = self.sites[start].dims()[0];
        let r = self.sites[start + length - 1].dims()[2];
        let physical = crate::error::checked_product(dims.iter().copied())?;
        self.limits
            .tensor(crate::error::checked_product([physical, physical])?)?;
        self.limits
            .tensor(crate::error::checked_product([l, physical, r])?)?;
        let mut theta = self.sites[start].clone();
        for next in &self.sites[start + 1..start + length] {
            let [left, p, bond] = theta.dims();
            let [_, d, right] = next.dims();
            self.limits
                .tensor(crate::error::checked_product([left, p, d, right])?)?;
            theta = reshape(
                &mm(
                    &reshape(&theta, [left * p, bond]),
                    &reshape(next, [bond, d * right]),
                ),
                [left, p * d, right],
            );
        }
        let trans = ComplexTensor {
            re: theta.re.permute([1, 0, 2]).reshape([physical, l * r]),
            im: theta.im.permute([1, 0, 2]).reshape([physical, l * r]),
        };
        let u = gate.to_tensor(&self.device(), self.sites[0].dtype())?;
        let out = mm(&reshape(&u, [physical, physical]), &trans);
        let mut remaining = ComplexTensor {
            re: out.re.reshape([physical, l, r]).permute([1, 0, 2]),
            im: out.im.reshape([physical, l, r]).permute([1, 0, 2]),
        };
        let norm = crate::complex::read(remaining.abs_squared().sum())?[0];
        if !norm.is_finite() || norm <= 0. || (norm - 1.).abs() > 1e-3 {
            return Err(Error::InvalidValue("MPS gate lost normalization".into()));
        }
        self.report.max_norm_drift = self.report.max_norm_drift.max((norm - 1.).abs());
        remaining = scale(&remaining, 1. / norm.sqrt());
        let mut left = l;
        let mut p = physical;
        for (offset, &d) in dims.iter().enumerate().take(length - 1) {
            let s = svd(&reshape(&remaining, [left * d, (p / d) * r]), self.limits)?;
            let rank = s.s.len().min(self.limits.max_bond);
            let total: f64 = s.s.iter().map(|v| v * v).sum();
            let lost: f64 = s.s[rank..].iter().map(|v| v * v).sum::<f64>() / total;
            self.report = accumulate(self.report, lost, self.limits)?;
            let kept = (total * (1. - lost)).sqrt();
            if kept <= 0. || !kept.is_finite() {
                return Err(Error::InvalidValue(
                    "MPS truncation removed whole state".into(),
                ));
            }
            let u = s.u.columns(0, rank).into_owned();
            let mut v = s.v.rows(0, rank).into_owned();
            for a in 0..rank {
                for b in 0..v.ncols() {
                    v[(a, b)] *= s.s[a] / kept;
                }
            }
            self.sites[start + offset] = from(
                &rows(&u),
                [left, d, rank],
                &self.device(),
                self.sites[0].dtype(),
            )?;
            p /= d;
            left = rank;
            remaining = from(
                &rows(&v),
                [left, p, r],
                &self.device(),
                self.sites[0].dtype(),
            )?;
        }
        self.sites[start + length - 1] = remaining;
        self.center = start + length - 1;
        limit(
            "MPS persistent state",
            self.storage_elements(),
            self.limits.max_state_elements,
        )
    }
}
