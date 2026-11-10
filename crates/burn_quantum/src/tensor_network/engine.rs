use super::{MpsLimits, MpsState, limit};
use crate::{
    Angle, Capabilities, Circuit, Error, Matrix, Observable, Operation, QuantumBackend,
    ResourceLimits, Result, Unitary,
};
use burn::tensor::{DType, Device, Tensor, TensorData};
use num_complex::Complex64;

/// Canonical quantum-backend adapter. Batches are independent MPS chains;
/// probability export remains explicitly dense and bounded.
#[derive(Clone, Copy, Debug, Default)]
pub struct MpsSimulator {
    /// Per-chain tensor-network limits.
    pub limits: MpsLimits,
    /// Budgets for batched storage and requested dense readouts.
    pub readout_limits: ResourceLimits,
}
impl MpsSimulator {
    fn validate_batch(&self, state: &[MpsState]) -> Result<()> {
        let first = state
            .first()
            .ok_or_else(|| Error::Shape("empty MPS batch".into()))?;
        let dimensions = first.dimensions();
        let device = first.device();
        let dtype = first.sites[0].dtype();
        let mut storage = 0usize;
        for chain in state {
            if chain.dimensions() != dimensions {
                return Err(Error::Shape("heterogeneous MPS batch dimensions".into()));
            }
            if chain.device() != device || chain.sites[0].dtype() != dtype {
                return Err(Error::TensorContext);
            }
            storage = storage
                .checked_add(chain.storage_elements())
                .ok_or(Error::Overflow)?;
        }
        limit(
            "MPS batched state",
            storage,
            self.readout_limits.max_state_elements,
        )?;
        self.readout_limits.check_state(state.len(), 1)?;
        Ok(())
    }
}
impl QuantumBackend for MpsSimulator {
    type State = Vec<MpsState>;
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            mixed_qudits: true,
            reverse_mode: false,
            density_matrices: false,
            dynamic_control: true,
        }
    }
    fn execute(
        &self,
        circuit: &Circuit,
        parameters: Option<Tensor<1>>,
        inputs: Option<Tensor<2>>,
        batch: usize,
        device: &Device,
    ) -> Result<Self::State> {
        self.limits.validate()?;
        if batch == 0
            || parameters.as_ref().map(|x| x.dims()[0]).unwrap_or(0) != circuit.num_parameters()
            || inputs.as_ref().map(|x| x.dims()).unwrap_or([batch, 0])
                != [batch, circuit.num_inputs()]
        {
            return Err(Error::Shape("MPS circuit binding mismatch".into()));
        }
        if parameters.as_ref().is_some_and(|x| x.is_require_grad())
            || inputs.as_ref().is_some_and(|x| x.is_require_grad())
        {
            return Err(Error::Unsupported(
                "MPS circuit has no reverse-mode SVD; use parameter shift".into(),
            ));
        }
        if parameters.as_ref().is_some_and(|x| x.device() != *device)
            || inputs.as_ref().is_some_and(|x| x.device() != *device)
        {
            return Err(Error::TensorContext);
        }
        if let (Some(p), Some(x)) = (&parameters, &inputs) {
            crate::complex::compatible(p, x)?;
        }
        let dtype = parameters
            .as_ref()
            .map(|x| x.dtype())
            .or_else(|| inputs.as_ref().map(|x| x.dtype()))
            .unwrap_or(DType::F32);
        let p = parameters
            .map(crate::complex::read)
            .transpose()?
            .unwrap_or_default();
        let x = inputs
            .map(crate::complex::read)
            .transpose()?
            .unwrap_or_default();
        if p.iter().chain(x.iter()).any(|v| !v.is_finite()) {
            return Err(Error::InvalidValue("nonfinite MPS circuit binding".into()));
        }
        let dimensions = circuit.register().dimensions();
        let minimum = dimensions
            .iter()
            .try_fold(0usize, |n, d| n.checked_add(*d).ok_or(Error::Overflow))?;
        self.readout_limits.check_state(batch, minimum)?;
        let mut out = Vec::with_capacity(batch);
        let mut count = 0usize;
        for b in 0..batch {
            let mut state = MpsState::basis(
                dimensions,
                &vec![0; dimensions.len()],
                device,
                dtype,
                self.limits,
            )?;
            for op in circuit.operations() {
                match op {
                    Operation::Unitary { wires, gate } => state.apply(wires, gate)?,
                    Operation::Rotate { wire, rotation } => {
                        let angle = match rotation.angle {
                            Angle::Constant(v) => v,
                            Angle::Parameter {
                                index,
                                scale,
                                offset,
                            } => p[index] * scale + offset,
                            Angle::Input {
                                index,
                                scale,
                                offset,
                            } => x[b * circuit.num_inputs() + index] * scale + offset,
                        };
                        let d = dimensions[*wire];
                        let [a, z] = rotation.levels;
                        let elements = crate::error::checked_product([d, d])?;
                        self.limits.tensor(elements)?;
                        let mut values = vec![Complex64::new(0., 0.); elements];
                        for i in 0..d {
                            values[i * d + i] = 1.0.into();
                        }
                        let c = (angle / 2.).cos();
                        let s = (angle / 2.).sin();
                        values[a * d + a] = c.into();
                        values[z * d + z] = c.into();
                        match rotation.axis {
                            crate::gate::Axis::X => {
                                values[a * d + z] = Complex64::new(0., -s);
                                values[z * d + a] = Complex64::new(0., -s);
                            }
                            crate::gate::Axis::Y => {
                                values[a * d + z] = (-s).into();
                                values[z * d + a] = s.into();
                            }
                            crate::gate::Axis::Z => {
                                values[a * d + a] = Complex64::new(c, -s);
                                values[z * d + z] = Complex64::new(c, s);
                            }
                        }
                        state.apply(&[*wire], &Unitary::new(Matrix::new(d, values)?)?)?;
                    }
                    Operation::ControlledX { controls, target } => {
                        let mut gate = Unitary::x();
                        let mut wires = vec![*target];
                        for &(w, level) in controls.iter().rev() {
                            let n = gate
                                .matrix()
                                .dimension()
                                .checked_mul(dimensions[w])
                                .ok_or(Error::Overflow)?;
                            self.limits.tensor(crate::error::checked_product([n, n])?)?;
                            gate = gate.controlled(dimensions[w], level)?;
                            wires.insert(0, w);
                        }
                        state.apply(&wires, &gate)?;
                    }
                    Operation::PhaseFlip { controls } => {
                        let wires: Vec<_> = controls.iter().map(|v| v.0).collect();
                        if wires.is_empty() {
                            let elements =
                                crate::error::checked_product([dimensions[0], dimensions[0]])?;
                            self.limits.tensor(elements)?;
                            state.apply(
                                &[0],
                                &Unitary::new(Matrix::real(
                                    dimensions[0],
                                    (0..elements)
                                        .map(|i| {
                                            if i / dimensions[0] == i % dimensions[0] {
                                                -1.
                                            } else {
                                                0.
                                            }
                                        })
                                        .collect(),
                                )?)?,
                            )?;
                        } else {
                            let dims: Vec<_> = wires.iter().map(|w| dimensions[*w]).collect();
                            let r = crate::Register::new(dims)?;
                            let n = r.size();
                            self.limits.tensor(crate::error::checked_product([n, n])?)?;
                            let selected =
                                r.encode(&controls.iter().map(|v| v.1).collect::<Vec<_>>())?;
                            let mut values = vec![0.; n * n];
                            for i in 0..n {
                                values[i * n + i] = if i == selected { -1. } else { 1. };
                            }
                            state.apply(&wires, &Unitary::new(Matrix::real(n, values)?)?)?;
                        }
                    }
                }
            }
            count = count
                .checked_add(state.storage_elements())
                .ok_or(Error::Overflow)?;
            limit(
                "MPS batched state",
                count,
                self.readout_limits.max_state_elements,
            )?;
            out.push(state);
        }
        Ok(out)
    }
    fn probabilities(&self, state: &Self::State) -> Result<Tensor<2>> {
        self.validate_batch(state)?;
        let n = crate::error::checked_product(state[0].dimensions())?;
        self.readout_limits.check_state(state.len(), n)?;
        let probs = state
            .iter()
            .map(|s| s.to_dense(self.readout_limits).map(|d| d.probabilities()))
            .collect::<Result<Vec<_>>>()?;
        Ok(Tensor::cat(probs, 0))
    }
    fn expectation(&self, state: &Self::State, observable: &Observable) -> Result<Tensor<1>> {
        self.validate_batch(state)?;
        let values = state
            .iter()
            .map(|s| s.expectation(observable))
            .collect::<Result<Vec<_>>>()?;
        Ok(Tensor::from_data(
            TensorData::new(values, [state.len()]),
            (&state[0].device(), state[0].sites[0].dtype()),
        ))
    }
}
