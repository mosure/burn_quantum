//! Explicit measurement and classical feed-forward over an ordinary unitary circuit.
//! A branch is selected by supplied outcomes, not sampled or silently postselected.
use crate::{Circuit, ComplexTensor, DenseSimulator, Error, Result, StateVector};
use burn::tensor::{Tensor, TensorData};
use serde::{Deserialize, Serialize};

/// One step in a measurement protocol. Operation indices reference the accompanying circuit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Instruction {
    /// Apply an operation iff all initialized classical conditions match.
    Gate {
        /// Index of the next circuit operation.
        operation: usize,
        /// Classical `(bit, value)` conditions. Empty means unconditional.
        conditions: Vec<(usize, usize)>,
    },
    /// Computational-basis projective measurement of a qubit into a fresh classical bit.
    Measure {
        /// Qubit measured, retained in the register in its collapsed state.
        wire: usize,
        /// Classical destination, in measurement order.
        bit: usize,
    },
}
/// Ordered dynamic instructions; the associated circuit stores all quantum gates once.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Protocol {
    /// Executable ordered steps.
    pub instructions: Vec<Instruction>,
}
impl Protocol {
    /// Validate operation ordering, qubit measurements and initialized classical controls.
    pub fn validate(&self, circuit: &Circuit) -> Result<usize> {
        let mut gates = 0;
        let mut bits = 0;
        for instruction in &self.instructions {
            match instruction {
                Instruction::Gate {
                    operation,
                    conditions,
                } => {
                    if *operation != gates
                        || *operation >= circuit.operations().len()
                        || conditions.iter().enumerate().any(|(i, &(b, v))| {
                            b >= bits || v > 1 || conditions[..i].iter().any(|c| c.0 == b)
                        })
                    {
                        return Err(Error::InvalidValue(
                            "invalid protocol gate or classical condition".into(),
                        ));
                    }
                    gates += 1;
                }
                Instruction::Measure { wire, bit } => {
                    if *bit != bits || circuit.register().dimensions().get(*wire) != Some(&2) {
                        return Err(Error::InvalidValue(
                            "protocol measurements require qubits and fresh ordered bits".into(),
                        ));
                    }
                    bits += 1;
                }
            }
        }
        if gates != circuit.operations().len() {
            return Err(Error::Shape(
                "protocol must reference every circuit gate exactly once".into(),
            ));
        }
        Ok(bits)
    }
}

/// Incremental execution of one explicit measurement branch, on the input state's device.
/// Conditional probabilities multiply into `probability`; impossible branches are errors.
/// Normalization stays on device, preserving gradients of the conditional state for a fixed
/// positive-probability branch. Outcome selection and the scalar likelihood report are not differentiable.
pub struct BranchExecution<'a> {
    circuit: &'a Circuit,
    protocol: &'a Protocol,
    engine: &'a DenseSimulator,
    state: StateVector,
    parameters: Option<Tensor<1>>,
    inputs: Option<Tensor<2>>,
    outcomes: Vec<usize>,
    classical: Vec<usize>,
    probability: f64,
    next: usize,
}
impl<'a> BranchExecution<'a> {
    /// Validate a protocol and supplied outcomes before executing any instruction.
    pub fn new(
        circuit: &'a Circuit,
        protocol: &'a Protocol,
        engine: &'a DenseSimulator,
        state: StateVector,
        outcomes: Vec<usize>,
        parameters: Option<Tensor<1>>,
        inputs: Option<Tensor<2>>,
    ) -> Result<Self> {
        let bits = protocol.validate(circuit)?;
        if state.register() != circuit.register()
            || state.batch_size() != 1
            || outcomes.len() != bits
            || outcomes.iter().any(|v| *v > 1)
        {
            return Err(Error::Shape("measurement branch requires one matching state and one binary outcome per measurement".into()));
        }
        Ok(Self {
            circuit,
            protocol,
            engine,
            state,
            parameters,
            inputs,
            outcomes,
            classical: vec![],
            probability: 1.,
            next: 0,
        })
    }
    /// Current normalized conditional state. The caller supplies a normalized input.
    pub fn state(&self) -> &StateVector {
        &self.state
    }
    /// Outcomes already measured; future requested outcomes are not exposed as classical data.
    pub fn classical(&self) -> &[usize] {
        &self.classical
    }
    /// Probability of the selected branch up to the current instruction.
    pub fn probability(&self) -> f64 {
        self.probability
    }
    /// Number of completed instructions.
    pub fn completed(&self) -> usize {
        self.next
    }
    /// Execute one instruction, including actual collapse/normalization. Returns false at the end.
    pub async fn advance(&mut self) -> Result<bool> {
        let Some(instruction) = self.protocol.instructions.get(self.next) else {
            return Ok(false);
        };
        match instruction {
            Instruction::Gate {
                operation,
                conditions,
            } => {
                if conditions
                    .iter()
                    .all(|&(bit, value)| self.classical[bit] == value)
                {
                    let mut step = Circuit::new(
                        self.circuit.register().clone(),
                        self.circuit.num_parameters(),
                        self.circuit.num_inputs(),
                    );
                    step.push(self.circuit.operations()[*operation].clone())?;
                    self.state = step.run_from(
                        self.engine,
                        self.state.clone(),
                        self.parameters.clone(),
                        self.inputs.clone(),
                    )?;
                }
            }
            Instruction::Measure { wire, bit } => {
                let outcome = self.outcomes[*bit];
                let branch_probability = self
                    .state
                    .marginal(&[*wire], self.engine.limits)?
                    .slice([0..1, outcome..outcome + 1]);
                let probabilities = branch_probability
                    .clone()
                    .into_data_async()
                    .await
                    .map_err(|e| Error::Readback(format!("{e:?}")))?
                    .convert::<f64>()
                    .try_to_vec::<f64>()
                    .map_err(|e| Error::Readback(format!("{e:?}")))?;
                let probability = probabilities[0];
                if !probability.is_finite() || probability <= 0. || probability > 1.0005 {
                    return Err(Error::InvalidValue(
                        "impossible or invalid measurement branch".into(),
                    ));
                }
                let n = self.state.register().size();
                let stride: usize = self.state.register().dimensions()[wire + 1..]
                    .iter()
                    .product();
                let mask: Vec<f64> = (0..n)
                    .map(|i| if i / stride % 2 == outcome { 1. } else { 0. })
                    .collect();
                let mask = Tensor::from_data(
                    TensorData::new(mask, [1, n]),
                    (&self.state.device(), self.state.amplitudes().dtype()),
                );
                self.state = StateVector::from_amplitudes(
                    self.state.register().clone(),
                    ComplexTensor::new(
                        self.state.amplitudes().real() * mask.clone()
                            / branch_probability.clone().sqrt(),
                        self.state.amplitudes().imag() * mask / branch_probability.sqrt(),
                    )?,
                    self.engine.limits,
                )?;
                self.classical.push(outcome);
                self.probability *= probability;
            }
        }
        self.next += 1;
        Ok(true)
    }
}
