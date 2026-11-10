//! Photonic/FBQC models with explicit physical representations and execution budgets.
//!
//! Encoded dual-rail sites use `|10>`, `|01>`, `|00>`; [`optics`] separately models
//! bounded bosonic Fock sectors and calibrated optical instruments. Dense native
//! Burn execution, host stabilizer/frame sampling, logical/resource compilers,
//! QEC and causal/worker runtimes retain their own capability and noise contracts.
//! Compilation does not allocate a global quantum state. Ideal resource supply,
//! physical failures, erasure recovery and hardware qualification are distinct
//! assumptions; see the individual modules and repository acceptance roadmap.
pub mod compiler;
pub mod encoding;
pub mod fabric;
pub mod factories;
pub mod fusion;
pub mod interop;
pub mod learning;
pub mod loss;
pub mod optics;
pub mod program;
pub mod qec;
pub mod resource;
pub mod runtime;
pub mod simulator;
pub mod stabilizer;

pub use encoding::DualRail;
pub use fabric::{Fabric, FabricLink, FabricNode, NodeId, NodeKind, Route, RouteHop};
pub use fusion::{FusionBranch, FusionOutcome, FusionParities, Parity, StandardFusion};
pub use loss::RailTransmission;
pub use program::{
    CompileOptions, CompiledFusionNetwork, FusionProgram, FusionRequest, PhotonId, ResourceInstance,
};
pub use resource::ResourceState;
pub use simulator::{DenseFusionSimulator, FusionRun, FusionShot, SimulationLimits};
