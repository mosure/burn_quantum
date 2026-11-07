//! Versioned, serializable circuit structure and differentiable parameter binding.
use crate::complex::compatible;
use crate::gate::{Axis, rotation};
use crate::{DenseSimulator, Error, Register, Result, StateVector, Unitary};
use burn::tensor::{DType, Device, Tensor, TensorData};
use serde::{Deserialize, Serialize};

/// An affine angle expression. References may be shared across gate occurrences.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Angle {
    /// Constant angle in radians.
    Constant(f64),
    /// `scale * parameters[index] + offset`.
    Parameter {
        /// Parameter-vector index.
        index: usize,
        /// Chain-rule scale.
        scale: f64,
        /// Constant phase offset.
        offset: f64,
    },
    /// `scale * inputs[batch, index] + offset`.
    Input {
        /// Input-feature index.
        index: usize,
        /// Feature scale.
        scale: f64,
        /// Constant phase offset.
        offset: f64,
    },
}
impl Angle {
    /// Reference a trainable parameter without affine scaling.
    pub fn parameter(index: usize) -> Self {
        Self::Parameter {
            index,
            scale: 1.,
            offset: 0.,
        }
    }
    /// Reference a per-example input feature without affine scaling.
    pub fn input(index: usize) -> Self {
        Self::Input {
            index,
            scale: 1.,
            offset: 0.,
        }
    }
    fn validate(&self, parameters: usize, inputs: usize) -> Result<()> {
        let valid = match self {
            Self::Constant(x) => x.is_finite(),
            Self::Parameter {
                index,
                scale,
                offset,
            } => *index < parameters && scale.is_finite() && offset.is_finite(),
            Self::Input {
                index,
                scale,
                offset,
            } => *index < inputs && scale.is_finite() && offset.is_finite(),
        };
        if valid {
            Ok(())
        } else {
            Err(Error::InvalidValue(
                "invalid angle expression or binding index".into(),
            ))
        }
    }
    fn bind(
        &self,
        parameters: &Option<Tensor<1>>,
        inputs: &Option<Tensor<2>>,
        batch: usize,
        device: &Device,
        dtype: DType,
    ) -> Tensor<1> {
        match self {
            Self::Constant(x) => Tensor::from_data(TensorData::new(vec![*x], [1]), (device, dtype)),
            Self::Parameter {
                index,
                scale,
                offset,
            } => {
                parameters
                    .as_ref()
                    .unwrap()
                    .clone()
                    .slice_dim(0, *index..*index + 1)
                    * *scale
                    + *offset
            }
            Self::Input {
                index,
                scale,
                offset,
            } => {
                inputs
                    .as_ref()
                    .unwrap()
                    .clone()
                    .slice([0..batch, *index..*index + 1])
                    .reshape([batch])
                    * *scale
                    + *offset
            }
        }
    }
}

/// Analytic two-level SU(2) rotation, including qubit Pauli rotations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rotation {
    /// Rotation generator axis.
    pub axis: Axis,
    /// Ordered levels defining the embedded Pauli basis.
    pub levels: [usize; 2],
    /// Angle expression.
    pub angle: Angle,
}
impl Rotation {
    /// A qubit rotation in the |0>, |1> subspace.
    pub fn qubit(axis: Axis, angle: Angle) -> Self {
        Self {
            axis,
            levels: [0, 1],
            angle,
        }
    }
}

/// Closed set of executable v1 instructions. Arbitrary local unitaries use `Unitary`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Operation {
    /// Constant unitary on ordered target wires.
    Unitary {
        /// Local basis follows this target order.
        wires: Vec<usize>,
        /// Checked local operator.
        gate: Unitary,
    },
    /// Differentiable rotation on one finite-dimensional subsystem.
    Rotate {
        /// Target wire.
        wire: usize,
        /// Generator, levels, and binding.
        rotation: Rotation,
    },
}

#[derive(Serialize, Deserialize)]
struct CircuitData {
    version: u32,
    register: Register,
    parameters: usize,
    inputs: usize,
    operations: Vec<Operation>,
}

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
        }
        Ok(())
    }
    /// Execute on the portable dense engine. Bindings remain ordinary Burn tensors.
    pub fn run(
        &self,
        engine: &DenseSimulator,
        parameters: Option<Tensor<1>>,
        inputs: Option<Tensor<2>>,
        batch: usize,
        device: &Device,
    ) -> Result<StateVector> {
        self.run_shifted(engine, parameters, inputs, batch, device, None)
    }
    pub(crate) fn run_shifted(
        &self,
        engine: &DenseSimulator,
        parameters: Option<Tensor<1>>,
        inputs: Option<Tensor<2>>,
        batch: usize,
        device: &Device,
        shift: Option<(usize, f64)>,
    ) -> Result<StateVector> {
        if batch == 0 {
            return Err(Error::Shape("batch must be positive".into()));
        }
        match &parameters {
            Some(p) if p.dims() == [self.parameters] && p.device() == *device => {}
            None if self.parameters == 0 => {}
            _ => return Err(Error::Shape("parameter shape or device mismatch".into())),
        }
        match &inputs {
            Some(x) if x.dims() == [batch, self.inputs] && x.device() == *device => {}
            None if self.inputs == 0 => {}
            _ => return Err(Error::Shape("input shape or device mismatch".into())),
        }
        if let (Some(p), Some(x)) = (&parameters, &inputs) {
            compatible(p, x)?;
        }
        let dtype = parameters
            .as_ref()
            .map(|x| x.dtype())
            .or_else(|| inputs.as_ref().map(|x| x.dtype()))
            .unwrap_or_else(|| device.settings().float_dtype.into());
        // Validate every resource-dependent instruction before allocating the state.
        engine.limits.check_state(batch, self.register.size())?;
        if !self.operations.is_empty() {
            engine.limits.check_plan(self.register.size())?;
        }
        for op in &self.operations {
            match op {
                Operation::Unitary { gate, .. } => {
                    engine.limits.check_matrix(1, gate.matrix().dimension())?
                }
                Operation::Rotate { wire, rotation } => {
                    let d = self.register.dimensions()[*wire];
                    let entries = crate::error::checked_product([d, d])?;
                    if entries > crate::gate::MAX_BUILTIN_MATRIX_ELEMENTS {
                        return Err(Error::ResourceLimit {
                            resource: "built-in rotation",
                            required: entries,
                            limit: crate::gate::MAX_BUILTIN_MATRIX_ELEMENTS,
                        });
                    }
                    let gb = if matches!(rotation.angle, Angle::Input { .. }) {
                        batch
                    } else {
                        1
                    };
                    engine
                        .limits
                        .check_matrix(gb, self.register.dimensions()[*wire])?;
                }
            }
        }
        let mut state =
            StateVector::zero(self.register.clone(), batch, device, dtype, engine.limits)?;
        for (i, op) in self.operations.iter().enumerate() {
            state = match op {
                Operation::Unitary { wires, gate } => engine.apply(&state, wires, gate)?,
                Operation::Rotate {
                    wire,
                    rotation: spec,
                } => {
                    let mut angle = spec.angle.bind(&parameters, &inputs, batch, device, dtype);
                    if let Some((target, delta)) = shift
                        && target == i
                    {
                        angle = angle + delta;
                    }
                    let gate = rotation(
                        spec.axis,
                        self.register.dimensions()[*wire],
                        spec.levels,
                        angle,
                    )?;
                    engine.apply_tensor(&state, &[*wire], &gate)?
                }
            };
        }
        Ok(state)
    }
}
impl TryFrom<CircuitData> for Circuit {
    type Error = Error;
    fn try_from(data: CircuitData) -> Result<Self> {
        if data.version != 1 {
            return Err(Error::Unsupported(format!(
                "circuit schema version {}",
                data.version
            )));
        }
        let mut circuit = Self::new(data.register, data.parameters, data.inputs);
        for op in data.operations {
            circuit.push(op)?;
        }
        Ok(circuit)
    }
}
impl From<Circuit> for CircuitData {
    fn from(c: Circuit) -> Self {
        Self {
            version: 1,
            register: c.register,
            parameters: c.parameters,
            inputs: c.inputs,
            operations: c.operations,
        }
    }
}
