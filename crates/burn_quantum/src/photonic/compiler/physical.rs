//! Universal gate synthesis, finite physical schedules, calibrated resources and fusions.
use super::{GateSynthesis, SynthesisOptions};
use crate::photonic::fabric::{
    FabricConstraints, PacketRequest, PacketScheduler, RailPacketScheduler, RailRoute,
    WavelengthConstraints,
};
use crate::photonic::{CompileOptions, Fabric, NodeId, NodeKind, RailTransmission, Route};
use crate::{Circuit, Error, Operation, Result};
use serde::Serialize;
mod calibrated_execution;
mod calibration;
pub(super) mod execution;
mod scheduling;
pub use calibrated_execution::CalibratedOutcome;
pub use calibration::{PhysicalCalibration, ResourceFactoryModel};
pub use execution::{PhysicalOutcome, PhysicalRun};

/// Fixed causal consumption times and phase-stable optical rail channels.
#[derive(Clone, Debug, Serialize)]
pub struct PhysicalRailOptions {
    /// One tick per fusion in the synthesized gate sequence.
    pub fusion_ticks: Vec<u64>,
    /// Wavelength capacity, switch configuration, skew and route objective.
    pub constraints: WavelengthConstraints,
    /// Matched corresponding rail channels for every input/resource.
    pub wavelengths: [usize; 2],
}

/// Placement of injected data, supplied gate resources, and destructive analyzers.
#[derive(Clone, Debug, Serialize)]
pub struct PhysicalPlacement {
    /// Source of each logical input, in wire order.
    pub inputs: Vec<NodeId>,
    /// Source supplying each gate's Choi resource after its declared generation window.
    pub resource: NodeId,
    /// Shared analyzer; fusions are serialized within and between gates.
    pub analyzer: NodeId,
    /// Classical latency after a gate's final measurement before correcting outputs.
    pub feedback_ticks: u64,
}
/// One physically teleported native gate, after bounded universal synthesis.
#[derive(Clone, Debug, Serialize)]
pub struct PhysicalGate {
    pub(super) operation: Operation,
    pub(super) wires: Vec<usize>,
    pub(super) input: Vec<RailTransmission>,
    pub(super) resource_measured: Vec<RailTransmission>,
    pub(super) resource_outputs: Vec<RailTransmission>,
    routes: Vec<Route>,
    rail_routes: Vec<RailRoute>,
    start: u64,
    finish: u64,
}
impl PhysicalGate {
    /// Logical operation encoded in the resource before calibrated preparation noise.
    pub fn operation(&self) -> &Operation {
        &self.operation
    }
    /// Logical data wires in resource input/output order.
    pub fn wires(&self) -> &[usize] {
        &self.wires
    }
    /// Input/resource-half routes in fusion order.
    pub fn routes(&self) -> &[Route] {
        &self.routes
    }
    /// Independent rail paths when wavelength-aware compilation was requested.
    /// Aggregate `routes()` then contains times/attenuation with empty hop lists.
    pub fn rail_routes(&self) -> &[RailRoute] {
        &self.rail_routes
    }
    /// Resource birth and gate start tick.
    pub fn start_tick(&self) -> u64 {
        self.start
    }
    /// Completion including classical feedback and output waiting loss.
    pub fn finish_tick(&self) -> u64 {
        self.finish
    }
}
/// Checked physical circuit with explicit resources, loss, fusions and early termination.
///
/// Constant local gates and symbolic rotations are universal together with CNOT.
/// Dense unitaries (up to four qubits by default) and structured multicontrols are
/// synthesized into native gates before any resources are scheduled. Two-qubit
/// resources remain CNOT/CZ/SWAP, whose corrections are local Paulis. Calibrated
/// compilation models heralded source attempts, complete optical POVMs, emitted
/// resource noise and costs. Local feed-forward corrections remain ideal.
#[derive(Clone, Debug, Serialize)]
pub struct PhysicalCircuit {
    version: u32,
    pub(super) source: Circuit,
    fabric: Fabric,
    placement: PhysicalPlacement,
    options: CompileOptions,
    pub(super) gates: Vec<PhysicalGate>,
    pub(super) final_storage: Vec<RailTransmission>,
    synthesis_error: f64,
    constraints: Option<FabricConstraints>,
    calibration: Option<PhysicalCalibration>,
    rail_options: Option<PhysicalRailOptions>,
    finish: u64,
}
impl PhysicalCircuit {
    /// Original logical circuit and its binding metadata.
    pub fn source(&self) -> &Circuit {
        &self.source
    }
    /// Teleported gates and resource/routing provenance.
    pub fn gates(&self) -> &[PhysicalGate] {
        &self.gates
    }
    /// End of the all-success execution path, including feedback.
    pub fn finish_tick(&self) -> u64 {
        self.finish
    }
    /// Maximum newly supplied resource photons; aborting attempts can use fewer.
    pub fn resource_photons(&self) -> usize {
        if let Some(model) = &self.calibration {
            self.gates
                .iter()
                .map(|g| {
                    model
                        .resource(g.wires.len())
                        .unwrap()
                        .factory()
                        .max_photons()
                })
                .sum()
        } else {
            2 * self.gates.iter().map(|g| g.wires.len()).sum::<usize>()
        }
    }
    /// Host synthesis reconstruction diagnostic; physical noise is accounted separately.
    pub fn synthesis_error(&self) -> f64 {
        self.synthesis_error
    }
    /// Finite buffer/configuration provenance, when requested.
    pub fn constraints(&self) -> Option<&FabricConstraints> {
        self.constraints.as_ref()
    }
    /// Wavelength/skew/capacity provenance, when independent rail routing is requested.
    pub fn rail_options(&self) -> Option<&PhysicalRailOptions> {
        self.rail_options.as_ref()
    }
    /// Complete source/instrument snapshot, including all effect matrices and noise channels.
    pub fn calibration(&self) -> Option<&PhysicalCalibration> {
        self.calibration.as_ref()
    }
}
