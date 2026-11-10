//! Checked binary syndrome/logical relations and exact erasure decoding.
//!
//! Includes binary-channel erasure decoding, signed measurement relations, and an
//! explicit periodic four-star FBQC reference. No hardware threshold is inferred.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
mod encoded;
pub mod fusion_surgery;
mod gf2;
mod magic;
mod memory;
mod network;
mod open;
mod optical_erasure;
mod planar;
mod planar_study;
mod relations;
mod six_ring;
mod study;
mod surgery;
pub mod threshold;
pub use encoded::{
    EncodedFusion, EncodedFusionRecovery, ShorBlock, ShorEncodedResource, ShorOrientation,
};
pub use magic::{
    MagicDistillation, MagicDistillationConfig, MagicDistillationCost, MagicInjection,
    MagicInjectionBranch, MagicRecord, MagicState, ReedMuller15,
};
pub use memory::{MemoryBudgets, PhysicalMemoryStudy, PhysicalToricMemory, ToricFusionCode};
pub use network::{FourStarTorus, MatchingDecode, SyndromeGraph};
pub use open::{PatchBoundary, PatchShot, PhotonLossBurst, SixRingPatch};
pub use optical_erasure::{FusionErasureDistribution, OpticalFusionLoss};
pub use planar::{PlanarCode, PlanarCost, PlanarDecodeBudget, PlanarReadout, PlanarRecovery};
pub use planar_study::{PlanarMemoryNoise, PlanarMemoryStudy, run_planar_memory};
pub use relations::{MeasurementRelations, ParityInference, ParityRelation};
pub use six_ring::SixRingTorus;
pub use study::{
    MemoryStudy, MemoryStudyTimings, NetworkNoise, WilsonInterval, run_four_star_memory,
    run_four_star_memory_profiled,
};
pub use surgery::{
    MergeBasis, PlanarSurgery, SurgeryAbort, SurgeryAttempt, SurgeryCheckRecord, SurgeryCnotRecord,
    SurgeryDetectorLimits, SurgeryDetectorModel, SurgeryDistance, SurgeryFault, SurgeryLimits,
    SurgeryMeasurement, SurgeryPhase, SurgeryRecord, SurgeryRecovery,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CodeData {
    version: u32,
    variables: usize,
    checks: Vec<Vec<bool>>,
    logicals: Vec<Vec<bool>>,
}
/// Binary relations: syndrome H e and logical-frame flips L e for an error vector e.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "CodeData", into = "CodeData")]
pub struct ParityCode(CodeData);
/// Exact decoding status under the assumption that errors occur only at erased sites.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErasureDecode {
    /// All solutions have the same logical action, even if physical corrections differ.
    Decoded {
        /// One correction supported entirely inside the erasure set.
        correction: Vec<bool>,
        /// Logical frame flips L correction.
        logical_flips: Vec<bool>,
    },
    /// The erased variables support at least one undetectable logical change.
    Ambiguous {
        /// Logical row indices whose values are not fixed by the syndrome.
        logicals: Vec<usize>,
    },
    /// The syndrome cannot be explained by the declared erasures.
    Inconsistent,
}
impl ParityCode {
    /// Validate dense binary check/logical rows; up to 4096 variables and 2^22 entries.
    /// Empty check or logical collections are allowed; every present row has full width.
    pub fn new(variables: usize, checks: Vec<Vec<bool>>, logicals: Vec<Vec<bool>>) -> Result<Self> {
        if variables == 0 || variables > 4096 {
            return Err(Error::InvalidDimensions(
                "parity-code width must be in 1..=4096".into(),
            ));
        }
        let rows = checks
            .len()
            .checked_add(logicals.len())
            .ok_or(Error::Overflow)?;
        let required = rows.checked_mul(variables).ok_or(Error::Overflow)?;
        if required > 1 << 22 {
            return Err(Error::ResourceLimit {
                resource: "parity-code entries",
                required,
                limit: 1 << 22,
            });
        }
        if checks
            .iter()
            .chain(&logicals)
            .any(|row| row.len() != variables)
        {
            return Err(Error::Shape("parity-code row width".into()));
        }
        Ok(Self(CodeData {
            version: 1,
            variables,
            checks,
            logicals,
        }))
    }
    /// Odd-distance repetition memory for X errors: adjacent ZZ checks and logical Z0.
    /// This protects only one error channel and is not a full fault-tolerant FBQC code.
    pub fn repetition(distance: usize) -> Result<Self> {
        if !(3..=4095).contains(&distance) || distance.is_multiple_of(2) {
            return Err(Error::InvalidDimensions(
                "repetition distance must be odd in 3..=4095".into(),
            ));
        }
        // Delegate the same memory bound before allocating rows.
        let required = distance.checked_mul(distance).ok_or(Error::Overflow)?;
        if required > 1 << 22 {
            return Err(Error::ResourceLimit {
                resource: "parity-code entries",
                required,
                limit: 1 << 22,
            });
        }
        let checks = (0..distance - 1)
            .map(|i| {
                let mut row = vec![false; distance];
                row[i] = true;
                row[i + 1] = true;
                row
            })
            .collect();
        let mut logical = vec![false; distance];
        logical[0] = true;
        Self::new(distance, checks, vec![logical])
    }
    /// Number of independent error coordinates, not necessarily physical photon count.
    pub fn variables(&self) -> usize {
        self.0.variables
    }
    /// Validated check rows.
    pub fn checks(&self) -> &[Vec<bool>] {
        &self.0.checks
    }
    /// Validated logical-frame rows.
    pub fn logicals(&self) -> &[Vec<bool>] {
        &self.0.logicals
    }
    fn check_vector(&self, value: &[bool]) -> Result<()> {
        if value.len() != self.variables() {
            Err(Error::Shape("error/erasure vector width".into()))
        } else {
            Ok(())
        }
    }
    /// Compute the syndrome for an explicit fault vector.
    pub fn syndrome(&self, errors: &[bool]) -> Result<Vec<bool>> {
        self.check_vector(errors)?;
        Ok(self.checks().iter().map(|r| dot(r, errors)).collect())
    }
    /// Compute changes to declared logical observables.
    pub fn logical_flips(&self, errors: &[bool]) -> Result<Vec<bool>> {
        self.check_vector(errors)?;
        Ok(self.logicals().iter().map(|r| dot(r, errors)).collect())
    }
    /// Solve H e = syndrome over erased coordinates, detecting logical ambiguity.
    ///
    /// Uses bounded host GF(2) elimination. Correlated erasures remain one supplied
    /// mask; the decoder does not replace them by independent per-check randomness.
    /// Unknown non-erased errors require a different decoder/noise model.
    pub fn decode_erasures(&self, syndrome: &[bool], erased: &[bool]) -> Result<ErasureDecode> {
        self.check_vector(erased)?;
        if syndrome.len() != self.checks().len() {
            return Err(Error::Shape("syndrome width".into()));
        }
        let unknown: Vec<_> = erased
            .iter()
            .enumerate()
            .filter_map(|(i, &e)| e.then_some(i))
            .collect();
        let n = unknown.len();
        let matrix: Vec<Vec<bool>> = self
            .checks()
            .iter()
            .map(|r| unknown.iter().map(|&q| r[q]).collect())
            .collect();
        let Some((solution, kernel)) = gf2::solve(&matrix, n, syndrome) else {
            return Ok(ErasureDecode::Inconsistent);
        };
        let ambiguous: Vec<_> = self
            .logicals()
            .iter()
            .enumerate()
            .filter_map(|(l, logical)| {
                kernel
                    .iter()
                    .any(|v| {
                        v.iter()
                            .zip(&unknown)
                            .fold(false, |s, (&b, &q)| s ^ (b && logical[q]))
                    })
                    .then_some(l)
            })
            .collect();
        if !ambiguous.is_empty() {
            return Ok(ErasureDecode::Ambiguous {
                logicals: ambiguous,
            });
        }
        let mut correction = vec![false; self.variables()];
        for (&q, b) in unknown.iter().zip(solution) {
            correction[q] = b;
        }
        let logical_flips = self.logical_flips(&correction)?;
        Ok(ErasureDecode::Decoded {
            correction,
            logical_flips,
        })
    }
}
fn dot(a: &[bool], b: &[bool]) -> bool {
    a.iter().zip(b).fold(false, |s, (&x, &y)| s ^ (x && y))
}
impl TryFrom<CodeData> for ParityCode {
    type Error = Error;
    fn try_from(data: CodeData) -> Result<Self> {
        if data.version != 1 {
            return Err(Error::Unsupported("unknown parity-code schema".into()));
        }
        Self::new(data.variables, data.checks, data.logicals)
    }
}
impl From<ParityCode> for CodeData {
    fn from(value: ParityCode) -> Self {
        value.0
    }
}
