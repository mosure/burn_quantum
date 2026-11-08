//! Checked wire-format conversions; instruction changes must preserve schema rules.
use super::{Circuit, Operation};
use crate::{Error, Register, Result};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub(super) struct CircuitData {
    version: u32,
    register: Register,
    parameters: usize,
    inputs: usize,
    operations: Vec<Operation>,
}

impl TryFrom<CircuitData> for Circuit {
    type Error = Error;
    fn try_from(data: CircuitData) -> Result<Self> {
        if !matches!(data.version, 1 | 2) {
            return Err(Error::Unsupported(format!(
                "circuit schema version {}",
                data.version
            )));
        }
        let mut circuit = Self::new(data.register, data.parameters, data.inputs);
        for op in data.operations {
            if data.version == 1
                && matches!(
                    op,
                    Operation::ControlledX { .. } | Operation::PhaseFlip { .. }
                )
            {
                return Err(Error::Unsupported(
                    "structured controlled gates require circuit schema v2".into(),
                ));
            }
            circuit.push(op)?;
        }
        Ok(circuit)
    }
}
impl From<Circuit> for CircuitData {
    fn from(c: Circuit) -> Self {
        Self {
            version: if c.operations.iter().any(|op| {
                matches!(
                    op,
                    Operation::ControlledX { .. } | Operation::PhaseFlip { .. }
                )
            }) {
                2
            } else {
                1
            },
            register: c.register,
            parameters: c.parameters,
            inputs: c.inputs,
            operations: c.operations,
        }
    }
}
