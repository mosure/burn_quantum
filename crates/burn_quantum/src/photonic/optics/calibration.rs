//! Joint optical preparation/noise mixtures and identifiable count calibration.
use super::{FockBasis, FockDensity, ModeId};
use crate::{ComplexTensor, Error, ResourceLimits, Result};
use burn::tensor::{DType, Device, Tensor, TensorData};
use std::collections::BTreeMap;

/// Complete finite joint photon-number distribution; mode emissions need not be independent.
#[derive(Clone, Debug)]
pub struct JointPhotonSource {
    modes: usize,
    branches: Vec<(Vec<usize>, f64)>,
    calibration: String,
}
impl JointPhotonSource {
    /// Combine duplicate occupations and validate all probability mass before preparation.
    pub fn new(
        modes: usize,
        branches: Vec<(Vec<usize>, f64)>,
        calibration: String,
    ) -> Result<Self> {
        identity(&calibration)?;
        if modes == 0
            || modes > 256
            || branches.is_empty()
            || branches.len() > 65_536
            || branches.len().checked_mul(modes).ok_or(Error::Overflow)? > 1 << 24
        {
            return Err(Error::InvalidValue(
                "invalid joint source dimensions".into(),
            ));
        }
        weights(branches.iter().map(|x| x.1))?;
        let mut merged = BTreeMap::new();
        for (occupation, p) in branches {
            if occupation.len() != modes
                || occupation
                    .iter()
                    .try_fold(0usize, |s, n| s.checked_add(*n))
                    .is_none_or(|n| n > 64)
            {
                return Err(Error::InvalidValue(
                    "joint source occupation exceeds reference support".into(),
                ));
            }
            if p > 0. {
                *merged.entry(occupation).or_insert(0.) += p;
            }
        }
        Ok(Self {
            modes,
            branches: merged.into_iter().collect(),
            calibration,
        })
    }
    /// Complete nonzero occupation weights.
    pub fn branches(&self) -> &[(Vec<usize>, f64)] {
        &self.branches
    }
    /// Caller-supplied immutable calibration identity.
    pub fn calibration(&self) -> &str {
        &self.calibration
    }
    /// Prepare the full correlated mixture; any omitted nonzero occupation is rejected.
    pub fn prepare(
        &self,
        basis: FockBasis,
        batch: usize,
        device: &Device,
        dtype: DType,
        limits: ResourceLimits,
    ) -> Result<FockDensity> {
        if basis.modes() != self.modes {
            return Err(Error::Shape("joint source basis mode mismatch".into()));
        }
        crate::backend::supported_dtype(dtype)?;
        let d = basis.size();
        let size = d.checked_mul(d).ok_or(Error::Overflow)?;
        limits.check_state(batch, size)?;
        let indices = self
            .branches
            .iter()
            .map(|(o, p)| Ok((basis.index(o)?, *p)))
            .collect::<Result<Vec<_>>>()?;
        let mut data = vec![0.; batch * size];
        for b in 0..batch {
            for &(i, p) in &indices {
                data[b * size + i * d + i] = p;
            }
        }
        let re = Tensor::from_data(TensorData::new(data, [batch, d, d]), (device, dtype));
        FockDensity::from_entries(
            basis,
            ComplexTensor::new(re.clone(), re.zeros_like())?,
            limits,
        )
    }
}
/// One latent calibration event, jointly determining all mode transmissions and phases.
#[derive(Clone, Debug)]
pub struct OpticalNoiseBranch {
    /// Probability per attempted use, before any detector conditioning.
    pub probability: f64,
    /// Intensity transmissions, one per coherent fine mode.
    pub transmissions: Vec<f64>,
    /// Coherent phase offsets in radians, one per fine mode.
    pub phases: Vec<f64>,
}
/// A bounded correlated classical mixture of coherent phase and pure-loss channels.
#[derive(Clone, Debug)]
pub struct CalibratedNoiseEnsemble {
    branches: Vec<OpticalNoiseBranch>,
    modes: usize,
    calibration: String,
}
impl CalibratedNoiseEnsemble {
    /// Validate a complete joint latent distribution (at most 64 events).
    pub fn new(branches: Vec<OpticalNoiseBranch>, calibration: String) -> Result<Self> {
        identity(&calibration)?;
        if branches.is_empty() || branches.len() > 64 {
            return Err(Error::InvalidValue(
                "invalid optical noise ensemble size".into(),
            ));
        }
        weights(branches.iter().map(|b| b.probability))?;
        let modes = branches[0].phases.len();
        if modes == 0
            || modes > 256
            || branches.iter().any(|b| {
                b.phases.len() != modes
                    || b.transmissions.len() != modes
                    || b.phases.iter().any(|p| !p.is_finite())
                    || b.transmissions
                        .iter()
                        .any(|p| !p.is_finite() || !(0.0..=1.).contains(p))
            })
        {
            return Err(Error::InvalidValue(
                "invalid correlated optical phases/transmissions".into(),
            ));
        }
        Ok(Self {
            branches,
            modes,
            calibration,
        })
    }
    /// Immutable latent model, suitable for reproducing shared phase/loss correlations.
    pub fn branches(&self) -> &[OpticalNoiseBranch] {
        &self.branches
    }
    /// Caller-supplied immutable calibration identity.
    pub fn calibration(&self) -> &str {
        &self.calibration
    }
    /// Propagate all latent events natively, preserving gradients and every loss sector.
    pub fn apply(&self, state: &FockDensity, limits: ResourceLimits) -> Result<FockDensity> {
        if state.basis().modes() != self.modes {
            return Err(Error::Shape("noise ensemble mode mismatch".into()));
        }
        let mut re = state.entries().real().zeros_like();
        let mut im = re.zeros_like();
        for branch in &self.branches {
            if branch.probability == 0. {
                continue;
            }
            let mut evolved = state.clone();
            for mode in 0..self.modes {
                let angle = Tensor::from_data(
                    TensorData::new(vec![branch.phases[mode]], [1]),
                    (&state.entries().device(), state.entries().dtype()),
                );
                evolved = evolved.phase_shift(ModeId(mode), angle, limits)?.lose(
                    ModeId(mode),
                    branch.transmissions[mode],
                    limits,
                )?;
            }
            re = re + evolved.entries().real() * branch.probability;
            im = im + evolved.entries().imag() * branch.probability;
        }
        FockDensity::from_entries(state.basis().clone(), ComplexTensor::new(re, im)?, limits)
    }
}
/// Raw independent calibration counts at a commanded Mach-Zehnder phase.
#[derive(Clone, Copy, Debug)]
pub struct CalibrationProbe {
    /// Known commanded phase in radians.
    pub phase: f64,
    /// Complete `[no click, output mode zero, output mode one]` counts.
    pub counts: [usize; 3],
}
/// Identifiable effective parameters of P(0/1 click)=eta*(1±v*cos(x+phi))/2.
/// Source brightness, optical transmission and detector efficiency cannot be separated
/// by these counts alone; `survival` denotes their identifiable combined response.
#[derive(Clone, Copy, Debug)]
pub struct MachZehnderCalibration {
    /// Unconditional observed probability of a single click.
    pub survival: f64,
    /// Identifiable fringe visibility after coherent/statistical imperfections.
    pub visibility: f64,
    /// Phase offset modulo 2*pi, in [-pi,pi].
    pub phase_offset: f64,
    /// Physical calibration attempts, including every no-click event.
    pub attempts: usize,
    /// Whether finite-sample visibility needed projection to the physical interval.
    pub projected_visibility: bool,
}
impl MachZehnderCalibration {
    /// Fit weighted linear sine/cosine coefficients; reject rank-deficient probe phases
    /// and unresolved phase at zero visibility. No separate loss-factor fit is implied.
    pub fn fit(probes: &[CalibrationProbe], minimum_visibility: f64) -> Result<Self> {
        if probes.len() < 2
            || probes.len() > 4096
            || !minimum_visibility.is_finite()
            || minimum_visibility <= 0.
            || minimum_visibility >= 1.
        {
            return Err(Error::InvalidValue(
                "invalid optical calibration probe design".into(),
            ));
        }
        let mut cc = 0.;
        let mut ss = 0.;
        let mut cs = 0.;
        let mut cy = 0.;
        let mut sy = 0.;
        let mut attempts = 0usize;
        let mut clicks = 0usize;
        for probe in probes {
            if !probe.phase.is_finite() {
                return Err(Error::InvalidValue("nonfinite calibration phase".into()));
            }
            let n = probe
                .counts
                .iter()
                .try_fold(0usize, |a, b| a.checked_add(*b))
                .ok_or(Error::Overflow)?;
            if n == 0 {
                return Err(Error::InvalidValue("empty calibration probe".into()));
            }
            attempts = attempts.checked_add(n).ok_or(Error::Overflow)?;
            clicks = clicks
                .checked_add(probe.counts[1])
                .and_then(|n| n.checked_add(probe.counts[2]))
                .ok_or(Error::Overflow)?;
            let c = probe.phase.cos();
            let s = probe.phase.sin();
            let y = probe.counts[1] as f64 - probe.counts[2] as f64;
            cc += n as f64 * c * c;
            ss += n as f64 * s * s;
            cs += n as f64 * c * s;
            cy += c * y;
            sy += s * y;
        }
        let determinant = cc * ss - cs * cs;
        if !determinant.is_finite() || determinant <= 1e-10 * (cc + ss).powi(2) {
            return Err(Error::InvalidValue(
                "calibration phases do not identify both quadratures".into(),
            ));
        }
        let a = (cy * ss - sy * cs) / determinant;
        let b = (sy * cc - cy * cs) / determinant;
        let survival = clicks as f64 / attempts as f64;
        let visibility = a.hypot(b) / survival;
        if !visibility.is_finite() || visibility < minimum_visibility {
            return Err(Error::InvalidValue(
                "phase is not identifiable at this fringe visibility".into(),
            ));
        }
        Ok(Self {
            survival,
            visibility: visibility.min(1.),
            phase_offset: (-b).atan2(a),
            attempts,
            projected_visibility: visibility > 1.,
        })
    }
    /// Predict a fresh commanded phase, retaining no clicks.
    pub fn probabilities(&self, phase: f64) -> Result<[f64; 3]> {
        if !phase.is_finite()
            || !self.survival.is_finite()
            || !(0. ..=1.).contains(&self.survival)
            || !self.visibility.is_finite()
            || !(0. ..=1.).contains(&self.visibility)
            || !self.phase_offset.is_finite()
            || !(phase + self.phase_offset).is_finite()
        {
            return Err(Error::InvalidValue(
                "nonfinite calibration prediction".into(),
            ));
        }
        let fringe = self.visibility * (phase + self.phase_offset).cos();
        Ok([
            1. - self.survival,
            self.survival * (1. + fringe) / 2.,
            self.survival * (1. - fringe) / 2.,
        ])
    }
}
fn identity(id: &str) -> Result<()> {
    if id.is_empty() || id.len() > 4096 {
        Err(Error::InvalidValue("invalid calibration identity".into()))
    } else {
        Ok(())
    }
}
fn weights(values: impl Iterator<Item = f64>) -> Result<()> {
    let mut total = 0.;
    for p in values {
        if !p.is_finite() || !(0.0..=1.).contains(&p) {
            return Err(Error::InvalidValue(
                "invalid joint calibration probability".into(),
            ));
        }
        total += p;
    }
    if (total - 1.).abs() > 1e-12 {
        Err(Error::InvalidValue(
            "joint calibration distribution is incomplete".into(),
        ))
    } else {
        Ok(())
    }
}
