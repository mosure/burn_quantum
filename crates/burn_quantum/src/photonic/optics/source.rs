//! Finite calibrated photon-number mixtures; no discarded probability is renormalized.
use super::{FockBasis, FockDensity, PhotonSector};
use crate::{ComplexTensor, Error, ResourceLimits, Result};
use burn::tensor::{DType, Device, Tensor, TensorData};
use serde::{Deserialize, Serialize};

/// Complete finite source photon-number distribution and immutable calibration identity.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "SourceData", into = "SourceData")]
pub struct CalibratedSource(SourceData);
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceData {
    probabilities: Vec<f64>,
    calibration: String,
}
impl CalibratedSource {
    /// Validate probabilities P(n), n=0..=64, with total one to 1e-12.
    pub fn new(probabilities: Vec<f64>, calibration: String) -> Result<Self> {
        Self::try_from(SourceData {
            probabilities,
            calibration,
        })
    }
    /// A specifically declared vacuum/single/pair model from mean photon count and g2.
    /// Uses p2=g2*mean²/2, p1=mean-2p2; infeasible moments are rejected.
    pub fn vacuum_single_pair(mean: f64, g2: f64, calibration: String) -> Result<Self> {
        if !mean.is_finite() || !g2.is_finite() || mean < 0. || g2 < 0. {
            return Err(Error::InvalidValue("invalid source moments".into()));
        }
        let p2 = 0.5 * g2 * mean * mean;
        let p1 = mean - 2. * p2;
        Self::new(vec![1. - p1 - p2, p1, p2], calibration)
    }
    /// Complete photon-number weights, including failed emission.
    pub fn probabilities(&self) -> &[f64] {
        &self.0.probabilities
    }
    /// Immutable calibration identifier supplied by the caller.
    pub fn calibration(&self) -> &str {
        &self.0.calibration
    }
    /// Mean emitted photon count per attempted preparation.
    pub fn mean(&self) -> f64 {
        self.0
            .probabilities
            .iter()
            .enumerate()
            .map(|(n, p)| n as f64 * p)
            .sum()
    }
    /// P(at least one photon); this is not heralded logical-resource yield.
    pub fn emission_probability(&self) -> f64 {
        1. - self.0.probabilities[0]
    }
    /// Normally ordered second moment divided by mean²; None for a vacuum source.
    pub fn g2(&self) -> Option<f64> {
        let mean = self.mean();
        (mean > 0.).then(|| {
            self.0
                .probabilities
                .iter()
                .enumerate()
                .map(|(n, p)| n.saturating_sub(1) as f64 * n as f64 * p)
                .sum::<f64>()
                / (mean * mean)
        })
    }
}
impl TryFrom<SourceData> for CalibratedSource {
    type Error = Error;
    fn try_from(v: SourceData) -> Result<Self> {
        if v.probabilities.is_empty()
            || v.probabilities.len() > 65
            || v.calibration.is_empty()
            || v.calibration.len() > 4096
            || v.probabilities
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.).contains(p))
            || (v.probabilities.iter().sum::<f64>() - 1.).abs() > 1e-12
        {
            return Err(Error::InvalidValue(
                "source needs a complete probability distribution and calibration identity".into(),
            ));
        }
        Ok(Self(v))
    }
}
impl From<CalibratedSource> for SourceData {
    fn from(v: CalibratedSource) -> Self {
        v.0
    }
}
impl FockDensity {
    /// Prepare independent calibrated mode sources as an exact diagonal mixture.
    /// Reject bases omitting any nonzero source mass; no silent postselection.
    pub fn source_product(
        basis: FockBasis,
        sources: &[CalibratedSource],
        batch: usize,
        device: &Device,
        dtype: DType,
        limits: ResourceLimits,
    ) -> Result<Self> {
        if sources.len() != basis.modes() {
            return Err(Error::Shape(
                "one photon-number source is required per fine mode".into(),
            ));
        }
        // Structural support checking rejects even arbitrarily small omitted tails.
        let minimum: usize = sources
            .iter()
            .map(|s| s.probabilities().iter().position(|p| *p > 0.).unwrap())
            .sum();
        let maximum: usize = sources
            .iter()
            .map(|s| s.probabilities().iter().rposition(|p| *p > 0.).unwrap())
            .sum();
        let contains_support = match basis.sector() {
            PhotonSector::Fixed(n) => minimum == n && maximum == n,
            PhotonSector::UpTo(n) => maximum <= n,
        };
        if !contains_support {
            return Err(Error::Unsupported(
                "source support exceeds the supplied Fock basis; preserve missing sectors".into(),
            ));
        }
        let d = basis.size();
        let size = d.checked_mul(d).ok_or(Error::Overflow)?;
        limits.check_state(batch, size)?;
        let weights: Vec<f64> = basis
            .occupations()
            .iter()
            .map(|o| {
                o.iter()
                    .zip(sources)
                    .map(|(&n, s)| s.probabilities().get(n).copied().unwrap_or(0.))
                    .product()
            })
            .collect();
        if (weights.iter().sum::<f64>() - 1.).abs() > 1e-12 {
            return Err(Error::Unsupported(
                "source support exceeds the supplied Fock basis; preserve missing sectors".into(),
            ));
        }
        let mut entries = vec![0.; batch * size];
        for b in 0..batch {
            for (i, &p) in weights.iter().enumerate() {
                entries[b * size + i * d + i] = p;
            }
        }
        let re = Tensor::from_data(TensorData::new(entries, [batch, d, d]), (device, dtype));
        Self::from_entries(
            basis,
            ComplexTensor::new(re.clone(), re.zeros_like())?,
            limits,
        )
    }
}
