//! Explicit randomized-basis linear-optical erasure reference.
//!
//! This is the supplied-resource stochastic model of Bartolucci et al. (2023),
//! Supplement II B/C, not a calibrated analyzer or source-generation simulation.
//! A lost signal or boosting photon erases both fusion parities. Failure with
//! no loss erases exactly one randomly selected parity. The joint law is retained.
use crate::{Error, Result};
use serde::Serialize;

/// Joint erasure law for one logical fusion, in mask order:
/// neither erased, XX only erased, ZZ only erased, both erased.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct FusionErasureDistribution {
    probabilities: [f64; 4],
}
impl FusionErasureDistribution {
    /// Probabilities indexed by `xx_erased as usize + 2 * zz_erased as usize`.
    pub fn probabilities(&self) -> [f64; 4] {
        self.probabilities
    }
    /// Marginal XX and ZZ erasure probabilities, retaining their joint law separately.
    pub fn marginals(&self) -> [f64; 2] {
        [
            self.probabilities[1] + self.probabilities[3],
            self.probabilities[2] + self.probabilities[3],
        ]
    }
    /// Deterministic inverse CDF, given one external uniform variate in [0,1).
    /// The returned flags are `[XX erased, ZZ erased]`.
    pub fn sample(&self, uniform: f64) -> Result<[bool; 2]> {
        if !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
            return Err(Error::InvalidValue(
                "fusion erasure variate outside [0,1)".into(),
            ));
        }
        let mut cumulative = 0.;
        for (mask, probability) in self.probabilities.iter().enumerate() {
            cumulative += probability;
            if uniform < cumulative || mask == 3 {
                return Ok([mask & 1 != 0, mask & 2 != 0]);
            }
        }
        unreachable!("the last interval includes residual floating-point roundoff")
    }
}

/// A fully specified loss-only optical reference with ideal supplied resources,
/// photon counting and heralding. No Pauli, dark-count or source-generation errors
/// are inferred. Signal and boosting photons share an independent loss rate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct OpticalFusionLoss {
    photon_loss: f64,
    boost_level: u8,
    shor_encoded: bool,
}
impl OpticalFusionLoss {
    /// Level zero is unboosted (failure 1/2, two photons); level one uses a Bell
    /// ancilla (failure 1/4, four photons). Levels through eight are supported.
    ///
    /// With `shor_encoded`, four physical fusions consume two `[[4,1,2]]` blocks.
    /// Their common code orientation is independently randomized per logical
    /// fusion. The source must supply those orientations; this is a model input,
    /// not a claim that a fixed source automatically provides it.
    pub fn new(photon_loss: f64, boost_level: u8, shor_encoded: bool) -> Result<Self> {
        if !photon_loss.is_finite() || !(0.0..=1.0).contains(&photon_loss) {
            return Err(Error::InvalidValue(
                "invalid optical photon-loss probability".into(),
            ));
        }
        if boost_level > 8 {
            return Err(Error::InvalidValue(
                "fusion boost level exceeds eight".into(),
            ));
        }
        Ok(Self {
            photon_loss,
            boost_level,
            shor_encoded,
        })
    }
    /// Independent loss probability for each signal and boosting photon.
    pub fn photon_loss(&self) -> f64 {
        self.photon_loss
    }
    /// Intrinsic physical fusion failure probability, conditional on no photon loss.
    pub fn failure_probability(&self) -> f64 {
        1. / f64::from(1u16 << (self.boost_level + 1))
    }
    /// Number of physical two-input fusion attempts per logical fusion attempt.
    pub fn physical_fusions(&self) -> usize {
        if self.shor_encoded { 4 } else { 1 }
    }
    /// Consumed signal photons per attempt, including failed attempts.
    pub fn signal_photons(&self) -> usize {
        2 * self.physical_fusions()
    }
    /// Consumed boosting photons per attempt; resource-source preparation overhead
    /// and rejected source attempts are deliberately not included.
    pub fn boosting_photons(&self) -> usize {
        ((1usize << (self.boost_level + 1)) - 2) * self.physical_fusions()
    }
    /// One physical fusion's correlated XX/ZZ erasure law.
    pub fn physical_distribution(&self) -> FusionErasureDistribution {
        let photons = 1u16 << (self.boost_level + 1);
        let survival = (1. - self.photon_loss).powi(i32::from(photons));
        let failure = self.failure_probability();
        FusionErasureDistribution {
            probabilities: [
                survival * (1. - failure),
                survival * failure / 2.,
                survival * failure / 2.,
                1. - survival,
            ],
        }
    }
    /// Exact joint logical erasure distribution. For encoded fusion, enumerate all
    /// 4^4 physical erasure patterns and both code orientations. Randomization is
    /// applied to the entire encoded fusion, not independently to its two outputs.
    pub fn distribution(&self) -> FusionErasureDistribution {
        let physical = self.physical_distribution();
        if !self.shor_encoded {
            return physical;
        }
        let mut probabilities = [0.; 4];
        for pattern in 0usize..256 {
            let masks: [usize; 4] = std::array::from_fn(|q| (pattern >> (2 * q)) & 3);
            let weight: f64 = masks.iter().map(|&m| physical.probabilities[m]).product();
            let x: [bool; 4] = masks.map(|m| m & 1 == 0);
            let z: [bool; 4] = masks.map(|m| m & 2 == 0);
            let known_x = (x[0] && x[2]) || (x[1] && x[3]);
            let known_z = (z[0] || z[2]) && (z[1] || z[3]);
            let standard = usize::from(!known_x) + 2 * usize::from(!known_z);
            let dual = usize::from(!known_z) + 2 * usize::from(!known_x);
            probabilities[standard] += weight / 2.;
            probabilities[dual] += weight / 2.;
        }
        FusionErasureDistribution { probabilities }
    }
    /// Invert the marginal erasure law for this boost/encoding configuration.
    /// `None` means that the target is below the unavoidable zero-loss failure
    /// floor. This is an algebraic model conversion, not a physical threshold.
    pub fn photon_loss_at_marginal(&self, target: f64) -> Result<Option<f64>> {
        if !target.is_finite() || !(0.0..=1.0).contains(&target) {
            return Err(Error::InvalidValue(
                "invalid marginal erasure target".into(),
            ));
        }
        let marginal = |loss: f64| {
            let physical = 1.
                - (1. - self.failure_probability() / 2.)
                    * (1. - loss).powi(i32::from(1u16 << (self.boost_level + 1)));
            if self.shor_encoded {
                ((1. - (1. - physical).powi(2)).powi(2) + 1. - (1. - physical.powi(2)).powi(2)) / 2.
            } else {
                physical
            }
        };
        if target < marginal(0.) {
            return Ok(None);
        }
        let (mut lower, mut upper) = (0., 1.);
        for _ in 0..64 {
            let midpoint = (lower + upper) / 2.;
            if marginal(midpoint) < target {
                lower = midpoint;
            } else {
                upper = midpoint;
            }
        }
        Ok(Some((lower + upper) / 2.))
    }
}
