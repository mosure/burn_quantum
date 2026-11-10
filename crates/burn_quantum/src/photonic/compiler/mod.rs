//! Logical measurement-pattern compilation and a native Burn exact reference.
//!
//! The compiler implements one-bit teleportation J(a)=H Rz(a), with measured
//! outcomes controlling physical X corrections before subsequent commands.
//! Measurement-pattern graph edges denote ideal logical entanglement. The separate
//! physical compiler realizes supported gates using routed Choi resources and
//! destructive fusions, calibrated heralded resources and complete optical instruments.
mod execution;
mod fusion_lowering;
mod lowering;
mod parity_retry;
mod physical;
mod synthesis;
use crate::{Angle, Circuit, ResourceLimits, StateVector};
use burn::tensor::Tensor;
pub use execution::PatternSimulator;
pub use fusion_lowering::{HeraldedGatePlan, TeleportationPorts};
pub use parity_retry::{ParityRetryBranch, ParityRetryPlan, ParityRetryRun};
pub use physical::PhysicalRailOptions;
pub use physical::{
    CalibratedOutcome, PhysicalCalibration, PhysicalCircuit, PhysicalGate, PhysicalOutcome,
    PhysicalPlacement, PhysicalRun, ResourceFactoryModel,
};
use serde::Serialize;
pub use synthesis::{GateSynthesis, SynthesisOptions};

/// Stable physical site identity within one compiled measurement pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct PatternSite(pub usize);
/// Checked compiler expansion limits, independent of dense execution limits.
#[derive(Clone, Copy, Debug)]
pub struct PatternCompileLimits {
    /// Total input and freshly prepared sites.
    pub max_sites: usize,
    /// Total preparation, entanglement, measurement and correction commands.
    pub max_commands: usize,
}
impl Default for PatternCompileLimits {
    fn default() -> Self {
        Self {
            max_sites: 4096,
            max_commands: 32768,
        }
    }
}
/// A causal ideal measurement command; source programs are the persistence input.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum PatternCommand {
    /// Prepare a new |+> qubit with a fresh identity.
    PreparePlus {
        /// Fresh site.
        site: PatternSite,
    },
    /// Apply ideal graph-state CZ between two currently live sites.
    Entangle {
        /// Distinct live sites.
        sites: [PatternSite; 2],
    },
    /// Rotate by Rz(angle), then measure X and destructively remove the site.
    /// Equivalently measure in the equatorial basis with azimuth -angle.
    Measure {
        /// Consumed site.
        site: PatternSite,
        /// Preserved affine parameter/input expression.
        angle: Angle,
        /// Classical result index (false=+1, true=-1).
        bit: usize,
    },
    /// Apply X iff an earlier measurement bit is true; this is real feed-forward.
    CorrectX {
        /// Live correction target.
        site: PatternSite,
        /// Earlier classical result index.
        bit: usize,
    },
}
/// Immutable causal pattern compiled from a validated qubit circuit.
///
/// The logical input/output order is preserved. Global phase may differ when
/// decomposing constant unitaries; the implemented quantum channel is unchanged.
/// Serialization is inspection-only; persist the source circuit and recompile.
#[derive(Clone, Debug, Serialize)]
pub struct MeasurementPattern {
    version: u32,
    source: Circuit,
    commands: Vec<PatternCommand>,
    outputs: Vec<PatternSite>,
    sites: usize,
    measurements: usize,
    synthesis_error: f64,
}
impl MeasurementPattern {
    /// Synthesize bounded qubit gates/multicontrols then compile native operations.
    /// Symbolic bindings and reversed rotation generators are preserved. Resource,
    /// gate-expansion and unsupported-dimension failures are atomic.
    pub fn compile(source: &Circuit, limits: PatternCompileLimits) -> crate::Result<Self> {
        let synthesized = GateSynthesis::compile(
            source,
            SynthesisOptions {
                max_native_gates: limits.max_commands,
                ..Default::default()
            },
        )?;
        let mut pattern = lowering::compile(synthesized.circuit(), limits)?;
        pattern.source = source.clone();
        pattern.synthesis_error = synthesized.numerical_error();
        Ok(pattern)
    }
    /// Original validated circuit, including binding counts and readout wire order.
    pub fn source(&self) -> &Circuit {
        &self.source
    }
    /// Commands in causal execution order.
    pub fn commands(&self) -> &[PatternCommand] {
        &self.commands
    }
    /// Final physical sites in logical output order.
    pub fn outputs(&self) -> &[PatternSite] {
        &self.outputs
    }
    /// Total prepared/input site count, not simultaneous dense-state width.
    pub fn sites(&self) -> usize {
        self.sites
    }
    /// Number of measurements, hence number of binary feed-forward outcomes.
    pub fn measurements(&self) -> usize {
        self.measurements
    }
    /// Host universal-synthesis reconstruction diagnostic, separate from tensor error.
    pub fn synthesis_error(&self) -> f64 {
        self.synthesis_error
    }
}
/// Explicit exact-execution limits; branch/state limits are not a total memory cap.
#[derive(Clone, Copy, Debug)]
pub struct PatternSimulationLimits {
    /// Shared tensor and index budgets.
    pub dense: ResourceLimits,
    /// Maximum enumerated outcomes (2 to the number of measurements).
    pub max_branches: usize,
}
impl Default for PatternSimulationLimits {
    fn default() -> Self {
        Self {
            dense: ResourceLimits::default(),
            max_branches: 4096,
        }
    }
}
/// One corrected output with its measurement transcript and absolute probability.
#[derive(Clone, Debug)]
pub struct PatternBranch {
    pub(crate) outcomes: Vec<bool>,
    pub(crate) state: StateVector,
}
impl PatternBranch {
    /// Outcome bits in command order.
    pub fn outcomes(&self) -> &[bool] {
        &self.outcomes
    }
    /// Subnormalized corrected output in logical wire order. Norm squared is weight.
    pub fn state(&self) -> &StateVector {
        &self.state
    }
    /// Absolute branch weight, retaining its autodiff graph.
    pub fn probability(&self) -> Tensor<1> {
        self.state.norm_squared()
    }
    /// Normalize explicitly after host validation; retain denominator gradients.
    pub fn conditional_state(&self, min_probability: f64) -> crate::Result<StateVector> {
        crate::error::tolerance(min_probability)?;
        let p = self.probability();
        if crate::complex::read(p.clone())?
            .iter()
            .any(|v| !v.is_finite() || *v <= min_probability)
        {
            return Err(crate::Error::InvalidValue(
                "impossible or too-small pattern branch".into(),
            ));
        }
        let scale = p.sqrt().reshape([self.state.batch_size(), 1]);
        let amplitudes = crate::ComplexTensor::new(
            self.state.amplitudes().real() / scale.clone(),
            self.state.amplitudes().imag() / scale,
        )?;
        let limits = ResourceLimits {
            max_state_elements: self.state.batch_size() * self.state.register().size(),
            ..Default::default()
        };
        StateVector::from_amplitudes(self.state.register().clone(), amplitudes, limits)
    }
}
