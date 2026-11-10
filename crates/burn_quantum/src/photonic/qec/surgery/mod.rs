//! Local doubled-seam planar lattice surgery, with explicit merge/split records.
//!
//! This gauge-fixing realization joins corresponding boundary edges with weight-two
//! checks and replaces adjacent opposing checks by weight-six products. All supports
//! have bounded geometric diameter. It realizes the rough XX / smooth ZZ instruments
//! of Horsman et al., arXiv:1111.4022, using a doubled seam rather than their inserted
//! data strip. Ideal repeated checks do not establish a circuit-noise threshold.
mod detectors;
mod distance;
pub use detectors::{
    SurgeryCheckRecord, SurgeryDetectorLimits, SurgeryDetectorModel, SurgeryFault,
};
mod execution;
mod geometry;
mod recovery;
use super::PlanarCode;
use crate::photonic::stabilizer::PauliProduct;
pub use distance::SurgeryDistance;
pub use execution::{
    SurgeryAbort, SurgeryAttempt, SurgeryCnotRecord, SurgeryMeasurement, SurgeryPhase,
    SurgeryRecord,
};
pub use recovery::SurgeryRecovery;
/// Which boundary and joint logical parity are measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeBasis {
    /// Smooth boundaries, logical Z_A Z_B.
    ZZ,
    /// Rough boundaries, logical X_A X_B.
    XX,
}
/// Bounds checked before constructing geometry or scheduling measurements.
#[derive(Clone, Copy, Debug)]
pub struct SurgeryLimits {
    /// Total data sites in the two patches, excluding measurement ancillas/references.
    pub max_data_qubits: usize,
    /// Maximum distinct checks in either phase.
    pub max_checks: usize,
    /// Total ideal check measurements over d merge and d split rounds.
    pub max_measurements: usize,
    /// BFS vertex-edge visits used to establish spatial distances.
    pub max_distance_work: usize,
}
impl Default for SurgeryLimits {
    fn default() -> Self {
        Self {
            max_data_qubits: 2048,
            max_checks: 4096,
            max_measurements: 1 << 20,
            max_distance_work: 1 << 26,
        }
    }
}
/// Two distance-d patches joined and split by local commuting check measurements.
/// Relative indices are patch A followed by patch B; execution remaps either patch.
#[derive(Clone, Debug)]
pub struct PlanarSurgery {
    code: PlanarCode,
    basis: MergeBasis,
    merged: Vec<PauliProduct>,
    split: Vec<PauliProduct>,
    boundary: Vec<[usize; 2]>,
    coordinates: Vec<[isize; 2]>,
    distance: SurgeryDistance,
}
impl PlanarSurgery {
    /// Original independent patch code and its reversible encoder.
    pub fn code(&self) -> &PlanarCode {
        &self.code
    }
    /// Logical parity measured by this instrument.
    pub fn basis(&self) -> MergeBasis {
        self.basis
    }
    /// Full merged commuting generators. The first d entries are the seam gauges.
    pub fn merged_checks(&self) -> &[PauliProduct] {
        &self.merged
    }
    /// The d local seam checks, whose product is the measured logical parity.
    pub fn seam_checks(&self) -> &[PauliProduct] {
        &self.merged[..self.code.distance()]
    }
    /// Independent patch generators in order A:X, A:Z, B:X, B:Z.
    pub fn split_checks(&self) -> &[PauliProduct] {
        &self.split
    }
    /// Index pairs in split_checks for the opposing checks replaced during merge.
    pub fn boundary_checks(&self) -> &[[usize; 2]] {
        &self.boundary
    }
    /// Integer physical data coordinates for the doubled seam realization.
    pub fn coordinates(&self) -> &[[isize; 2]] {
        &self.coordinates
    }
    /// Exact shortest protected-logical spatial paths and d temporal repetitions.
    pub fn distance_certificate(&self) -> &SurgeryDistance {
        &self.distance
    }
    /// Total check measurements, excluding preparation and later logical readout.
    pub fn measurements(&self) -> usize {
        self.code.distance() * (self.merged.len() + self.split.len())
    }
}
