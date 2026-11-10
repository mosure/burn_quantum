//! Trace-preserving attenuation into vacuum, including asymmetric rail loss.
use crate::{Error, KrausChannel, Matrix, Result};
use serde::{Deserialize, Serialize};

/// Independent intensity transmission probabilities for the logical-zero/one rails.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "[f64; 2]", into = "[f64; 2]")]
pub struct RailTransmission([f64; 2]);
impl RailTransmission {
    /// Perfect transmission on both rails.
    pub const IDEAL: Self = Self([1., 1.]);
    /// Validate intensity (not amplitude) transmissions in `[0,1]`.
    pub fn new(zero: f64, one: f64) -> Result<Self> {
        if [zero, one]
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        {
            return Err(Error::InvalidValue(
                "rail transmission must be finite and in [0,1]".into(),
            ));
        }
        Ok(Self([zero, one]))
    }
    /// Equal transmission on both rails.
    pub fn uniform(value: f64) -> Result<Self> {
        Self::new(value, value)
    }
    /// Ordered rail transmission probabilities.
    pub fn values(self) -> [f64; 2] {
        self.0
    }
    /// Compose independent attenuation along consecutive path segments.
    pub fn then(self, next: Self) -> Self {
        Self([self.0[0] * next.0[0], self.0[1] * next.0[1]])
    }
    /// Apply a per-tick storage transmission for an integer number of ticks.
    pub fn over_ticks(self, mut ticks: u64) -> Self {
        let mut out = Self::IDEAL;
        let mut factor = self;
        while ticks > 0 {
            if ticks & 1 == 1 {
                out = out.then(factor);
            }
            factor = factor.then(factor);
            ticks >>= 1;
        }
        out
    }
    /// CPTP channel with separate environment records for loss from each rail.
    ///
    /// K0=diag(sqrt(eta0),sqrt(eta1),1), K1=sqrt(1-eta0)|vac><0|,
    /// K2=sqrt(1-eta1)|vac><1|. Vacuum is absorbing. Loss parameters are
    /// constant; gradients through an input density matrix are retained.
    pub fn channel(self) -> Result<KrausChannel> {
        let [a, b] = self.0;
        KrausChannel::new(vec![
            Matrix::real(3, vec![a.sqrt(), 0., 0., 0., b.sqrt(), 0., 0., 0., 1.])?,
            Matrix::real(3, vec![0., 0., 0., 0., 0., 0., (1. - a).sqrt(), 0., 0.])?,
            Matrix::real(3, vec![0., 0., 0., 0., 0., 0., 0., (1. - b).sqrt(), 0.])?,
        ])
    }
}
impl TryFrom<[f64; 2]> for RailTransmission {
    type Error = Error;
    fn try_from(v: [f64; 2]) -> Result<Self> {
        Self::new(v[0], v[1])
    }
}
impl From<RailTransmission> for [f64; 2] {
    fn from(v: RailTransmission) -> Self {
        v.0
    }
}
