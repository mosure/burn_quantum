//! Versioned, serializable circuit structure and differentiable parameter binding.
mod execution;
mod operation;
mod serialization;

pub use operation::{Angle, Operation, Rotation};
use serialization::CircuitData;

use crate::{Error, Register, Result, Unitary};
use serde::{Deserialize, Serialize};

/// Validated circuit specification, independent of device and tensor values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "CircuitData", into = "CircuitData")]
pub struct Circuit {
    register: Register,
    parameters: usize,
    inputs: usize,
    operations: Vec<Operation>,
}
impl Circuit {
    /// Create an empty circuit with declared parameter and feature counts.
    pub fn new(register: Register, parameters: usize, inputs: usize) -> Self {
        Self {
            register,
            parameters,
            inputs,
            operations: Vec::new(),
        }
    }
    /// Register layout.
    pub fn register(&self) -> &Register {
        &self.register
    }
    /// Trainable parameter count.
    pub fn num_parameters(&self) -> usize {
        self.parameters
    }
    /// Per-example feature count.
    pub fn num_inputs(&self) -> usize {
        self.inputs
    }
    /// Instructions in execution order.
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }
    /// Append an instruction after validation; failure leaves the circuit unchanged.
    pub fn push(&mut self, operation: Operation) -> Result<&mut Self> {
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }
    /// Append a constant unitary.
    pub fn unitary(&mut self, wires: Vec<usize>, gate: Unitary) -> Result<&mut Self> {
        self.push(Operation::Unitary { wires, gate })
    }
    /// Append an analytic rotation.
    pub fn rotate(&mut self, wire: usize, rotation: Rotation) -> Result<&mut Self> {
        self.push(Operation::Rotate { wire, rotation })
    }
    /// Append a structured controlled X without constructing a dense gate matrix.
    pub fn controlled_x(
        &mut self,
        controls: Vec<(usize, usize)>,
        target: usize,
    ) -> Result<&mut Self> {
        self.push(Operation::ControlledX { controls, target })
    }
    /// Append a structured phase flip without constructing a dense gate matrix.
    pub fn phase_flip(&mut self, controls: Vec<(usize, usize)>) -> Result<&mut Self> {
        self.push(Operation::PhaseFlip { controls })
    }
    fn validate_operation(&self, op: &Operation) -> Result<()> {
        match op {
            Operation::Unitary { wires, gate } => {
                if self.register.local_size(wires)? != gate.matrix().dimension() {
                    return Err(Error::Shape(
                        "unitary dimension differs from target dimension".into(),
                    ));
                }
            }
            Operation::Rotate { wire, rotation } => {
                let d = self.register.local_size(&[*wire])?;
                let [a, b] = rotation.levels;
                if a == b || a >= d || b >= d {
                    return Err(Error::Basis);
                }
                rotation.angle.validate(self.parameters, self.inputs)?;
            }
            Operation::ControlledX { controls, target } => {
                crate::control::validate_x(&self.register, controls, *target)?
            }
            Operation::PhaseFlip { controls } => {
                crate::control::validate_controls(&self.register, controls)?
            }
        }
        Ok(())
    }
}
