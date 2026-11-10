//! Physical six-ring constructions and exact bounded channel certificates.
//!
//! `SixRingBoundaryNetwork` builds a genuine local fusion graph with mixed-basis
//! staircase boundaries and a time-dependent seam. Its signed code and logical
//! channel are derived from the resource graph; end-code distance is distinct
//! from noisy spacetime fault distance or a threshold.
//!
//! `SixRingSurgery` separately lowers ordinary planar Pauli checks by gate
//! teleportation. Each CNOT consumes one ring and two unboosted analyzers, so
//! this reference has exponentially small yield. Physical attempts and explicit
//! conditioned channel references remain separate.
mod boundary;
mod certificate;
mod instrument;
mod lowering;
pub use instrument::{
    OpticalCheckAttempt, OpticalCheckFailure, OpticalGateRecord, OpticalSurgeryCost,
    SixRingCheckInstrument, SixRingSampling,
};
pub use lowering::{OpticalSurgeryRun, SixRingSurgery, SixRingSurgeryLimits};

pub use boundary::SixRingBoundaryNetwork;
pub use certificate::{
    BoundaryCertificate, BoundaryCertificateLimits, BoundaryCode, BoundaryDistance,
};

mod surface;
pub use surface::SixRingSurfaceNetwork;
