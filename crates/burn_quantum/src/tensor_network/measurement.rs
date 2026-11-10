//! Local Kraus trajectories and selective measurement with error accounting.
use super::{MpsState, state::validate_report, tensor::*};
use crate::{ComplexTensor, Error, Result};
use rand::Rng;
impl MpsState {
    /// Sample one local CPTP Kraus trajectory, including nonunitary no-jump
    /// filters (for example asymmetric dual-rail attenuation). The returned index
    /// and probability describe the sampled environment record. This is a pure
    /// trajectory, not a density matrix or a differentiable sampled decision.
    pub fn trajectory<R: Rng + ?Sized>(
        &mut self,
        site: usize,
        channel: &crate::KrausChannel,
        rng: &mut R,
    ) -> Result<(usize, f64)> {
        let mut next = self.clone();
        next.move_center(site)?;
        let source = &next.sites[site];
        let [l, d, r] = source.dims();
        if channel.operators()[0].dimension() != d {
            return Err(Error::Shape("MPS channel dimension".into()));
        }
        next.limits.tensor(crate::error::checked_product([d, d])?)?;
        let input = ComplexTensor {
            re: source.re.clone().permute([1, 0, 2]).reshape([d, l * r]),
            im: source.im.clone().permute([1, 0, 2]).reshape([d, l * r]),
        };
        let mut probabilities = Vec::with_capacity(channel.operators().len());
        for op in channel.operators() {
            let tensor = op.to_tensor(&source.device(), source.dtype())?;
            let out = mm(&reshape(&tensor, [d, d]), &input);
            probabilities.push(crate::complex::read(out.abs_squared().sum())?[0]);
        }
        let total: f64 = probabilities.iter().sum();
        if !total.is_finite() || (total - 1.).abs() > 1e-4 {
            return Err(Error::InvalidValue(
                "MPS Kraus probabilities do not sum to one".into(),
            ));
        }
        let mut draw = rng.random::<f64>() * total;
        let mut selected = probabilities.len() - 1;
        for (i, p) in probabilities.iter().enumerate() {
            if draw < *p {
                selected = i;
                break;
            }
            draw -= p;
        }
        let p = probabilities[selected];
        if p <= 0. {
            return Err(Error::InvalidValue("zero MPS Kraus probability".into()));
        }
        let op = channel.operators()[selected].to_tensor(&source.device(), source.dtype())?;
        let out = scale(&mm(&reshape(&op, [d, d]), &input), 1. / p.sqrt());
        next.sites[site] = ComplexTensor {
            re: out.re.reshape([d, l, r]).permute([1, 0, 2]),
            im: out.im.reshape([d, l, r]).permute([1, 0, 2]),
        };
        next.report.trace_distance_bound = (2. * next.report.trace_distance_bound / p).min(1.);
        validate_report(next.report, next.limits)?;
        *self = next;
        Ok((selected, p))
    }
    /// Destructive selective computational-basis measurement, retaining site size.
    /// Returns the physical probability; impossible outcomes fail atomically.
    pub fn condition(&mut self, site: usize, outcome: usize) -> Result<f64> {
        let mut next = self.clone();
        next.move_center(site)?;
        let s = &next.sites[site];
        let [l, d, r] = s.dims();
        if outcome >= d {
            return Err(Error::Basis);
        }
        let mut v = data(s)?;
        let mut p = 0.;
        for a in 0..l {
            for b in 0..d {
                for c in 0..r {
                    let i = (a * d + b) * r + c;
                    if b == outcome {
                        p += v[i].norm_sqr();
                    } else {
                        v[i] = 0.0.into();
                    }
                }
            }
        }
        if !p.is_finite() || p <= 0. {
            return Err(Error::InvalidValue("zero-probability MPS condition".into()));
        }
        for x in &mut v {
            *x /= p.sqrt();
        }
        next.report.trace_distance_bound = (2. * next.report.trace_distance_bound / p).min(1.);
        if next.report.trace_distance_bound > next.limits.max_truncation_distance + 1e-12 {
            return Err(Error::InvalidValue(
                "MPS conditioning exceeds error budget".into(),
            ));
        }
        next.sites[site] = from(&v, [l, d, r], &s.device(), s.dtype())?;
        *self = next;
        Ok(p)
    }
    /// One exact chain-rule sample from this represented state; self is unchanged.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<usize>> {
        let mut next = self.clone();
        let mut digits = Vec::with_capacity(self.sites.len());
        // Sampling reports the distribution of the represented approximate state;
        // it does not promise each rare conditioned branch meets the input budget.
        next.limits.max_truncation_distance = 1.;
        for i in 0..self.sites.len() {
            next.move_center(i)?;
            let [l, d, r] = next.sites[i].dims();
            let v = data(&next.sites[i])?;
            let mut p = vec![0.; d];
            for a in 0..l {
                for b in 0..d {
                    for c in 0..r {
                        p[b] += v[(a * d + b) * r + c].norm_sqr();
                    }
                }
            }
            let mut draw = rng.random::<f64>() * p.iter().sum::<f64>();
            let mut outcome = d - 1;
            for (j, mass) in p.iter().enumerate() {
                if draw < *mass {
                    outcome = j;
                    break;
                }
                draw -= mass;
            }
            next.condition(i, outcome)?;
            digits.push(outcome);
        }
        Ok(digits)
    }
}
