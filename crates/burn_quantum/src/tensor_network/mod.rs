//! Bounded matrix-product-state (MPS) simulation for qubits and mixed qudits.
//!
//! Site contractions execute on the selected native Burn device. Complex SVD is
//! a bounded host operation, with explicit readback; this engine does not claim
//! reverse-mode derivatives through SVD. Parameter-shift/finite-shot optimizers
//! remain applicable. Storage scales with Schmidt rank, not Hilbert dimension;
//! highly entangled circuits can still require exponential resources.
//!
//! Truncation occurs only at a mixed-canonical center. Its accumulated pure-state
//! distance bound excludes floating-point error, reported separately as norm
//! drift. See Vidal, <https://arxiv.org/abs/quant-ph/0301063>.
mod engine;
mod evolution;
mod measurement;
mod readout;
mod state;
pub use engine::MpsSimulator;
pub mod distributed;
mod tensor;

use crate::{Error, Result};
use serde::{Deserialize, Serialize};
pub use state::{MpsCheckpoint, MpsState, SiteData, TruncationReport};

/// Allocation and approximation contracts, checked before local work.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MpsLimits {
    /// Maximum number of sites (does not multiply their local dimensions).
    pub max_sites: usize,
    /// Maximum Schmidt dimension after a gate.
    pub max_bond: usize,
    /// Maximum complex entries in any site, contraction, or SVD input/factor.
    pub max_tensor_elements: usize,
    /// Maximum sum of persistent site entries.
    pub max_state_elements: usize,
    /// Maximum accumulated trace distance due to truncation/conditioning, `[0,1]`.
    /// Zero requests exact ranks, up to a 1e-12 floating-point comparison floor.
    pub max_truncation_distance: f64,
    /// Bound on host SVD iterations; failure is recoverable and atomic.
    pub svd_iterations: usize,
}
impl Default for MpsLimits {
    fn default() -> Self {
        Self {
            max_sites: 100_000,
            max_bond: 128,
            max_tensor_elements: 1 << 20,
            max_state_elements: 1 << 24,
            max_truncation_distance: 0.,
            svd_iterations: 10_000,
        }
    }
}
impl MpsLimits {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.max_sites == 0
            || self.max_bond == 0
            || self.max_tensor_elements == 0
            || self.max_state_elements == 0
            || self.svd_iterations == 0
            || !self.max_truncation_distance.is_finite()
            || !(0.0..=1.).contains(&self.max_truncation_distance)
        {
            return Err(Error::InvalidValue("invalid MPS limits".into()));
        }
        Ok(())
    }
    pub(crate) fn tensor(&self, n: usize) -> Result<()> {
        limit("MPS local tensor", n, self.max_tensor_elements)
    }
}
pub(crate) fn limit(resource: &'static str, required: usize, limit: usize) -> Result<()> {
    if required > limit {
        Err(Error::ResourceLimit {
            resource,
            required,
            limit,
        })
    } else {
        Ok(())
    }
}
