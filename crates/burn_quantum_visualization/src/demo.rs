//! Small, actual experiments with explicit scientific scope and cached trajectories.
mod catalog;
mod optimization;
use crate::{
    model::*,
    simulation::{TraceLimits, simulate},
};
use burn::tensor::Device;
use burn_quantum::{
    Angle, Circuit, Matrix, Operation, Register, Result, Rotation, Unitary, gate::Axis, photonic::*,
};
pub use catalog::{DemoConfig, Preview, build_configured, preview};
use serde::{Deserialize, Serialize};

/// Stable gallery identifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DemoKind {
    /// Bell-pair entanglement.
    Bell,
    /// Configurable GHZ correlations.
    Ghz,
    /// Phase-to-population interference.
    Interference,
    /// Qutrit Fourier interference.
    Qutrit,
    /// Actual ideal resource preparation and compiled photonic routing.
    PhotonicRouting,
    /// Qubits & qutrits.
    MixedRegister,
    /// Quantum teleportation.
    Teleportation,
    /// Toffoli / controlled X.
    Toffoli,
    /// Quantum Fourier transform.
    Qft,
    /// Deutsch–Jozsa.
    DeutschJozsa,
    /// Quantum phase estimation.
    Qpe,
    /// Grover search.
    Grover,
    /// QAOA · MaxCut.
    Qaoa,
    /// Bit-flip error correction.
    BitFlip,
    /// A measured one-parameter quantum-natural-SPSA optimization.
    Optimization,
}
impl DemoKind {
    /// Stable gallery order.
    pub fn all() -> &'static [Self] {
        &[
            Self::Bell,
            Self::Ghz,
            Self::Interference,
            Self::Qutrit,
            Self::PhotonicRouting,
            Self::MixedRegister,
            Self::Teleportation,
            Self::Toffoli,
            Self::Qft,
            Self::DeutschJozsa,
            Self::Qpe,
            Self::Grover,
            Self::Qaoa,
            Self::BitFlip,
            Self::Optimization,
        ]
    }
    /// URL/export identifier.
    pub fn id(self) -> &'static str {
        match self {
            Self::Bell => "bell",
            Self::Ghz => "ghz",
            Self::Interference => "interference",
            Self::Qutrit => "qutrit",
            Self::PhotonicRouting => "photonic",
            Self::MixedRegister => "mixed",
            Self::Teleportation => "teleportation",
            Self::Toffoli => "toffoli",
            Self::Qft => "qft",
            Self::DeutschJozsa => "deutsch-jozsa",
            Self::Qpe => "qpe",
            Self::Grover => "grover",
            Self::Qaoa => "qaoa",
            Self::BitFlip => "bit-flip",
            Self::Optimization => "optimization",
        }
    }
    /// Parse a gallery identifier without silently changing an unknown experiment.
    pub fn from_id(id: &str) -> Option<Self> {
        Self::all().iter().copied().find(|kind| kind.id() == id)
    }
    /// Gallery heading.
    pub fn title(self) -> &'static str {
        match self {
            Self::Bell => "Bell entanglement",
            Self::Ghz => "GHZ correlations",
            Self::Interference => "Phase interference",
            Self::Qutrit => "Qutrit Fourier",
            Self::PhotonicRouting => "Photonic routing",
            Self::MixedRegister => "Qubits & qutrits",
            Self::Teleportation => "Quantum teleportation",
            Self::Toffoli => "Toffoli / controlled X",
            Self::Qft => "Quantum Fourier transform",
            Self::DeutschJozsa => "Deutsch–Jozsa",
            Self::Qpe => "Quantum phase estimation",
            Self::Grover => "Grover search",
            Self::Qaoa => "QAOA · MaxCut",
            Self::BitFlip => "Bit-flip error correction",
            Self::Optimization => "Quantum natural SPSA",
        }
    }
    /// Physical assumptions and experiment purpose.
    pub fn description(self) -> &'static str {
        match self {
            Self::Bell => {
                "A Hadamard and CNOT prepare a Bell pair. Each entangled qubit has a maximally mixed reduced state at the Bloch-ball center, despite the pure global state."
            }
            Self::Ghz => {
                "Configurable GHZ preparation. Local reduced states become mixed while the joint distribution retains only the all-zero and all-one correlations."
            }
            Self::Interference => {
                "H, Rz(theta), H converts relative phase into population. The binding theta=pi gives destructive interference at zero. Complex amplitudes retain global phase."
            }
            Self::Qutrit => {
                "A dimension-three Fourier transform, a sign on level two, and inverse Fourier transform. Qutrit reduced density matrices and probabilities are shown without assigning a qubit Bloch sphere."
            }
            Self::PhotonicRouting => {
                "Two ideally supplied dual-rail Bell resources and their actual compiled packet/fusion schedule. Quantum frames show resource preparation only; link transmission is scheduling metadata, not an applied loss channel. No fusion success or output quantum state is assumed."
            }
            Self::MixedRegister => {
                "Two entangled pairs with different local dimensions, alongside two pure qubits. Qudit panels show populations; their density matrices and purity retain coherence information."
            }
            Self::Teleportation => {
                "A shared Bell pair, two projective measurements and classical X/Z corrections transfer Ry(1.1) Rz(0.7) input states to Bob. Each transfer has three quantum wires and two classical bits."
            }
            Self::Toffoli => {
                "A Toffoli flips its target only when both controls are one. Increase controls to explore a structured multi-controlled X without a dense exponential gate matrix."
            }
            Self::Qft => {
                "The positive-sign QFT includes final SWAPs, with wire zero most significant. Compare forward/inverse phase patterns for a computational-basis input."
            }
            Self::DeutschJozsa => {
                "An n-input parity or constant-one promise oracle acts on a |−> ancilla. Hadamards reveal a balanced oracle through a nonzero input-register result."
            }
            Self::Qpe => {
                "Counting qubits control powers of P(2π phase) on eigenstate |1>, followed by inverse QFT with bit reversal. Compare exactly representable and off-grid phases."
            }
            Self::Grover => {
                "Uniform preparation, a single marked-state phase oracle and diffusion amplify one target. Compare near-optimal iterations with over-rotation; this simulator remains exponential."
            }
            Self::Qaoa => {
                "Parameterized cost and mixer layers for a ring graph. Inspect the circuit and measured cut expectation as graph size and depth grow; angles are fixed, not trained in this example."
            }
            Self::BitFlip => {
                "Odd-length repetition encoding, injected X errors and coherent majority decoding. The logical state returns to q0, while other wires retain the syndrome. Corrects bit flips only, up to half the code distance rounded down."
            }
            Self::Optimization => {
                "One-parameter Ry ansatz minimizes measured Z expectation using the core QN-SPSA fidelity-metric estimator. Six async Burn loss/overlap probes per update, scalar running metric and absolute-value regularization; no shot noise. Frames are actual iterates, including the input. Seed 731; final circuit binding retained."
            }
        }
    }
}
/// Compute one document on an explicitly selected Burn device. Never called by a draw loop.
pub async fn build(
    kind: DemoKind,
    device: &Device,
    limits: TraceLimits,
) -> Result<VisualizationDocument> {
    build_configured(kind, DemoConfig::for_demo(kind), device, limits).await
}
async fn build_basic(
    kind: DemoKind,
    device: &Device,
    limits: TraceLimits,
) -> Result<VisualizationDocument> {
    if kind == DemoKind::Optimization {
        return optimization::build(device, limits).await;
    }
    let (mut circuit, mut labels, parameters) = match kind {
        DemoKind::Ghz => (Circuit::new(Register::qubits(4)?, 0, 0), vec![], vec![]),
        DemoKind::Interference => (
            Circuit::new(Register::qubits(1)?, 1, 0),
            vec![],
            vec![std::f64::consts::PI],
        ),
        DemoKind::Qutrit => (Circuit::new(Register::new(vec![3])?, 0, 0), vec![], vec![]),
        DemoKind::PhotonicRouting => (
            Circuit::new(Register::new(vec![3; 4])?, 0, 0),
            vec![],
            vec![],
        ),
        _ => (Circuit::new(Register::qubits(2)?, 0, 0), vec![], vec![]),
    };
    let mut fabric = None;
    match kind {
        DemoKind::Bell | DemoKind::Ghz => {
            circuit.unitary(vec![0], Unitary::h())?;
            labels.push("H on q0".into());
            for q in 1..circuit.register().wires() {
                circuit.unitary(vec![q - 1, q], Unitary::cnot())?;
                labels.push(format!("CNOT q{} → q{q}", q - 1));
            }
        }
        DemoKind::Interference => {
            circuit.unitary(vec![0], Unitary::h())?;
            circuit.rotate(0, Rotation::qubit(Axis::Z, Angle::parameter(0)))?;
            circuit.unitary(vec![0], Unitary::h())?;
            labels = vec![
                "H: create paths".into(),
                "Rz(theta): relative phase".into(),
                "H: recombine paths".into(),
            ];
        }
        DemoKind::Qutrit => {
            circuit.unitary(vec![0], Unitary::fourier(3)?)?;
            circuit.unitary(
                vec![0],
                Unitary::new(Matrix::real(3, vec![1., 0., 0., 0., 1., 0., 0., 0., -1.])?)?,
            )?;
            circuit.unitary(vec![0], Unitary::fourier(3)?.adjoint())?;
            labels = vec!["F3".into(), "Phase on level 2".into(), "F3 inverse".into()];
        }
        DemoKind::PhotonicRouting => {
            let resource = ResourceState::bell_pair();
            for offset in [0, 2] {
                for op in resource.circuit()?.operations() {
                    let mapped = match op {
                        Operation::Unitary { wires, gate } => Operation::Unitary {
                            wires: wires.iter().map(|q| q + offset).collect(),
                            gate: gate.clone(),
                        },
                        Operation::PhaseFlip { controls } => Operation::PhaseFlip {
                            controls: controls.iter().map(|&(q, l)| (q + offset, l)).collect(),
                        },
                        _ => {
                            return Err(burn_quantum::Error::Unsupported(
                                "unexpected resource preparation instruction".into(),
                            ));
                        }
                    };
                    circuit.push(mapped)?;
                    labels.push(format!(
                        "Bell resource {}: operation {}",
                        offset / 2 + 1,
                        labels.len() + 1
                    ));
                }
            }
            let mut program = FusionProgram::new();
            let a = program.add_resource(resource.clone(), NodeId(0))?;
            let b = program.add_resource(resource, NodeId(1))?;
            program.fuse(a[1], b[0], NodeId(3))?;
            let physical = Fabric::new(
                vec![
                    NodeKind::Source,
                    NodeKind::Source,
                    NodeKind::Switch,
                    NodeKind::Fusion,
                ]
                .into_iter()
                .map(|kind| FabricNode {
                    kind,
                    storage: RailTransmission::uniform(0.999).unwrap(),
                })
                .collect(),
                [(0, 2, 2, 0.98), (1, 2, 3, 0.97), (2, 3, 2, 0.96)]
                    .into_iter()
                    .map(|(from, to, ticks, eta)| {
                        Ok(FabricLink {
                            from: NodeId(from),
                            to: NodeId(to),
                            ticks,
                            transmission: RailTransmission::uniform(eta)?,
                        })
                    })
                    .collect::<Result<_>>()?,
            )?;
            fabric = Some(FabricSnapshot::from_compiled(
                &program.compile(&physical, CompileOptions::default())?,
            ));
        }
        _ => unreachable!("configurable demos are handled by the catalog"),
    }
    let trace = simulate(&circuit, &parameters, &[], &labels, device, limits).await?;
    let document = VisualizationDocument {
        schema_version: 1,
        title: kind.title().into(),
        description: kind.description().into(),
        circuit,
        parameters,
        inputs: vec![],
        operation_labels: labels,
        trace,
        fabric,
        optimization: vec![],
        protocol: None,
    };
    document.validate_with(limits)?;
    Ok(document)
}
