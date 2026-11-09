//! Dense-engine binding, resource preflight, and execution on existing or zero states.
use super::{Angle, Circuit, Operation};
use crate::complex::compatible;
use crate::gate::rotation;
use crate::{DenseSimulator, Error, Result, StateVector};
use burn::tensor::{DType, Device, Tensor, TensorData};

impl Angle {
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

impl Circuit {
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
        let dtype = self.validate_bindings(&parameters, &inputs, batch, device)?;
        self.preflight(engine, batch)?;
        let state = StateVector::zero(self.register.clone(), batch, device, dtype, engine.limits)?;
        self.apply_bound(engine, state, parameters, inputs, shift)
    }
    /// Execute on an existing state, preserving its batch, precision and autodiff graph.
    /// Register, bindings and allocation budgets are validated before execution.
    pub fn run_from(
        &self,
        engine: &DenseSimulator,
        state: StateVector,
        parameters: Option<Tensor<1>>,
        inputs: Option<Tensor<2>>,
    ) -> Result<StateVector> {
        if state.register() != &self.register {
            return Err(Error::Shape(
                "circuit and initial-state registers differ".into(),
            ));
        }
        self.validate_bindings(&parameters, &inputs, state.batch_size(), &state.device())?;
        if let Some(p) = &parameters {
            compatible(&state.amplitudes().real(), p)?;
        }
        if let Some(x) = &inputs {
            compatible(&state.amplitudes().real(), x)?;
        }
        self.preflight(engine, state.batch_size())?;
        self.apply_bound(engine, state, parameters, inputs, None)
    }
    fn validate_bindings(
        &self,
        parameters: &Option<Tensor<1>>,
        inputs: &Option<Tensor<2>>,
        batch: usize,
        device: &Device,
    ) -> Result<DType> {
        if batch == 0 {
            return Err(Error::Shape("batch must be positive".into()));
        }
        match parameters {
            Some(p) if p.dims() == [self.parameters] && p.device() == *device => {}
            None if self.parameters == 0 => {}
            _ => return Err(Error::Shape("parameter shape or device mismatch".into())),
        }
        match inputs {
            Some(x) if x.dims() == [batch, self.inputs] && x.device() == *device => {}
            None if self.inputs == 0 => {}
            _ => return Err(Error::Shape("input shape or device mismatch".into())),
        }
        if let (Some(p), Some(x)) = (parameters, inputs) {
            compatible(p, x)?;
        }
        Ok(parameters
            .as_ref()
            .map(|x| x.dtype())
            .or_else(|| inputs.as_ref().map(|x| x.dtype()))
            .unwrap_or_else(|| device.settings().float_dtype.into()))
    }
    fn preflight(&self, engine: &DenseSimulator, batch: usize) -> Result<()> {
        // Validate every resource-dependent instruction before allocating the state.
        engine.limits.check_state(batch, self.register.size())?;
        if !self.operations.is_empty() {
            engine.limits.check_plan(self.register.size())?;
            if self.register.size() > i32::MAX as usize {
                return Err(Error::Unsupported(
                    "dense gather indices exceed signed 32-bit range".into(),
                ));
            }
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
                Operation::ControlledX { .. } | Operation::PhaseFlip { .. } => {}
            }
        }
        Ok(())
    }
    fn apply_bound(
        &self,
        engine: &DenseSimulator,
        mut state: StateVector,
        parameters: Option<Tensor<1>>,
        inputs: Option<Tensor<2>>,
        shift: Option<(usize, f64)>,
    ) -> Result<StateVector> {
        let device = state.device();
        let dtype = state.amplitudes().dtype();
        let batch = state.batch_size();
        for (i, op) in self.operations.iter().enumerate() {
            state = match op {
                Operation::Unitary { wires, gate } => engine.apply(&state, wires, gate)?,
                Operation::Rotate {
                    wire,
                    rotation: spec,
                } => {
                    let mut angle = spec.angle.bind(&parameters, &inputs, batch, &device, dtype);
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
                Operation::ControlledX { controls, target } => {
                    engine.controlled_x(&state, controls, *target)?
                }
                Operation::PhaseFlip { controls } => engine.phase_flip(&state, controls)?,
            };
        }
        Ok(state)
    }
}
