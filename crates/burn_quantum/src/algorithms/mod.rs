//! Reusable canonical circuit constructions. All wires use big-endian register order.
//! These are dense-simulation-sized templates, not claims of classical quantum speedup.
mod deutsch_jozsa;
mod grover;
mod phase_estimation;
mod qaoa;
mod qft;
mod repetition;
mod teleportation;
mod toffoli;
use crate::{
    Angle, Circuit, Error, Operation, Register, Result, Rotation, Unitary,
    gate::Axis,
    protocol::{Instruction, Protocol},
};
pub use deutsch_jozsa::*;
pub use grover::*;
pub use phase_estimation::*;
pub use qaoa::*;
pub use qft::*;
pub use repetition::*;
pub use teleportation::*;
pub use toffoli::*;

/// A circuit with bindings and educational labels; protocols explicitly add measurement steps.
#[derive(Clone, Debug)]
pub struct AlgorithmCircuit {
    /// Reusable checked circuit, retaining symbolic variational parameters.
    pub circuit: Circuit,
    /// Binding values for the demonstrated experiment.
    pub parameters: Vec<f64>,
    /// One label per unitary circuit operation.
    pub labels: Vec<String>,
    /// Dynamic instruction sequence; absent for ordinary unitary circuits.
    pub protocol: Option<Protocol>,
}
impl AlgorithmCircuit {
    fn new(qubits: usize, parameters: Vec<f64>) -> Result<Self> {
        if !(1..=16).contains(&qubits) {
            return Err(Error::InvalidValue(
                "algorithm templates require 1..=16 qubits".into(),
            ));
        }
        Ok(Self {
            circuit: Circuit::new(Register::qubits(qubits)?, parameters.len(), 0),
            parameters,
            labels: vec![],
            protocol: None,
        })
    }
    fn push(&mut self, op: Operation, label: impl Into<String>) -> Result<()> {
        let index = self.circuit.operations().len();
        self.circuit.push(op)?;
        self.labels.push(label.into());
        if let Some(p) = &mut self.protocol {
            p.instructions.push(Instruction::Gate {
                operation: index,
                conditions: vec![],
            });
        }
        Ok(())
    }
    fn gate(&mut self, wires: Vec<usize>, gate: Unitary, label: impl Into<String>) -> Result<()> {
        self.push(Operation::Unitary { wires, gate }, label)
    }
    fn rotate(
        &mut self,
        wire: usize,
        axis: Axis,
        angle: Angle,
        label: impl Into<String>,
    ) -> Result<()> {
        self.push(
            Operation::Rotate {
                wire,
                rotation: Rotation::qubit(axis, angle),
            },
            label,
        )
    }
    fn cx(&mut self, control: usize, target: usize) -> Result<()> {
        self.push(
            Operation::ControlledX {
                controls: vec![(control, 1)],
                target,
            },
            format!("CX q{control} → q{target}"),
        )
    }
}
fn width(n: usize, min: usize, max: usize) -> Result<()> {
    if (min..=max).contains(&n) {
        Ok(())
    } else {
        Err(Error::InvalidValue(format!(
            "algorithm size must be {min}..={max}"
        )))
    }
}
