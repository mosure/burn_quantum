//! Bounded asynchronous Burn readback. Simulation is explicitly requested, never a render callback.
use crate::model::*;
use burn::tensor::{DType, Device, Tensor, TensorData};
use burn_quantum::{
    Circuit, DenseSimulator, DensityMatrix, Error, Register, ResourceLimits, Result, StateVector,
};
use serde::{Deserialize, Serialize};

/// Preflight limits for interactive workloads and retained host snapshots.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TraceLimits {
    /// Maximum register wires.
    pub max_wires: usize,
    /// Maximum pure-state Hilbert dimension.
    pub max_dimension: usize,
    /// Maximum dimension of one subsystem.
    pub max_local_dimension: usize,
    /// Maximum retained snapshots, including the input.
    pub max_frames: usize,
    /// Maximum aggregate amplitudes plus local density entries across frames.
    pub max_trace_elements: usize,
    /// Maximum entries in a mixed-state readout source.
    pub max_density_elements: usize,
    /// Maximum entries in one dense circuit gate.
    pub max_matrix_elements: usize,
}
impl Default for TraceLimits {
    fn default() -> Self {
        Self {
            max_wires: 16,
            max_dimension: 65536,
            max_local_dimension: 16,
            max_frames: 257,
            max_trace_elements: 1_048_576,
            max_density_elements: 65536,
            max_matrix_elements: 65536,
        }
    }
}
impl TraceLimits {
    /// Reject excessive sizes before constructing a state or retaining a timeline.
    pub fn check(&self, register: &Register, frames: usize) -> Result<()> {
        let local = register.dimensions().iter().try_fold(0usize, |sum, &d| {
            sum.checked_add(d.checked_mul(d).ok_or(Error::Overflow)?)
                .ok_or(Error::Overflow)
        })?;
        let total = register
            .size()
            .checked_add(local)
            .and_then(|v| v.checked_mul(frames))
            .ok_or(Error::Overflow)?;
        for (resource, required, limit) in [
            ("visualization wires", register.wires(), self.max_wires),
            ("visualization state", register.size(), self.max_dimension),
            (
                "visualization local dimension",
                *register.dimensions().iter().max().unwrap(),
                self.max_local_dimension,
            ),
            ("visualization frames", frames, self.max_frames),
            (
                "visualization retained elements",
                total,
                self.max_trace_elements,
            ),
        ] {
            if required > limit {
                return Err(Error::ResourceLimit {
                    resource,
                    required,
                    limit,
                });
            }
        }
        Ok(())
    }
    /// Matching bounded core simulator resources.
    pub fn resources(&self) -> ResourceLimits {
        ResourceLimits {
            max_state_elements: self.max_dimension.max(self.max_density_elements),
            max_matrix_elements: self.max_matrix_elements,
            max_plan_indices: self.max_density_elements.saturating_mul(2),
        }
    }
}
/// Portable asynchronous readback; works when synchronous GPU readback is unavailable.
pub async fn read_tensor<const D: usize>(tensor: Tensor<D>) -> Result<Vec<f64>> {
    tensor
        .into_data_async()
        .await
        .map_err(|e| Error::Readback(format!("{e:?}")))?
        .convert::<f64>()
        .try_to_vec::<f64>()
        .map_err(|e| Error::Readback(format!("{e:?}")))
}
async fn subsystem(wire: usize, density: DensityMatrix) -> Result<SubsystemState> {
    let dimension = density.register().size();
    let re = read_tensor(density.entries().real()).await?;
    let im = read_tensor(density.entries().imag()).await?;
    let values: Vec<_> = re
        .into_iter()
        .zip(im)
        .map(|(re, im)| ComplexValue { re, im })
        .collect();
    let probabilities = (0..dimension)
        .map(|i| values[i * dimension + i].re)
        .collect();
    let purity = values.iter().map(|v| v.re * v.re + v.im * v.im).sum();
    let bloch = (dimension == 2).then(|| {
        [
            2. * values[1].re,
            -2. * values[1].im,
            values[0].re - values[3].re,
        ]
    });
    Ok(SubsystemState {
        wire,
        dimension,
        probabilities,
        density: values,
        bloch,
        purity,
    })
}
/// Snapshot one unbatched pure state, using the core simulator's partial trace.
pub async fn snapshot_pure(
    state: &StateVector,
    step: usize,
    label: String,
    limits: TraceLimits,
) -> Result<TimelineFrame> {
    limits.check(state.register(), 1)?;
    if state.batch_size() != 1 {
        return Err(Error::Shape(
            "visualization requires one displayed batch row".into(),
        ));
    }
    let re = read_tensor(state.amplitudes().real()).await?;
    let im = read_tensor(state.amplitudes().imag()).await?;
    let amplitudes: Vec<_> = re
        .into_iter()
        .zip(im)
        .map(|(re, im)| ComplexValue { re, im })
        .collect();
    let probabilities: Vec<_> = amplitudes
        .iter()
        .map(|v| v.re * v.re + v.im * v.im)
        .collect();
    let mut subsystems = Vec::with_capacity(state.register().wires());
    for wire in 0..state.register().wires() {
        subsystems
            .push(subsystem(wire, state.reduced_density(&[wire], limits.resources())?).await?);
    }
    Ok(TimelineFrame {
        step,
        label,
        norm: probabilities.iter().sum(),
        probabilities,
        amplitudes: Some(amplitudes),
        subsystems,
    })
}
/// Snapshot a mixed-state source. Its physical validity remains the caller's core density contract.
/// No amplitudes are assigned to a mixed state; all Bloch vectors come from actual partial traces.
pub async fn snapshot_density(
    state: &DensityMatrix,
    step: usize,
    label: String,
    limits: TraceLimits,
) -> Result<TimelineFrame> {
    limits.check(state.register(), 1)?;
    let required = state
        .register()
        .size()
        .checked_mul(state.register().size())
        .ok_or(Error::Overflow)?;
    if required > limits.max_density_elements {
        return Err(Error::ResourceLimit {
            resource: "visualization density",
            required,
            limit: limits.max_density_elements,
        });
    }
    if state.batch_size() != 1 {
        return Err(Error::Shape(
            "visualization requires one displayed batch row".into(),
        ));
    }
    let probabilities = read_tensor(state.probabilities()).await?;
    let mut subsystems = Vec::with_capacity(state.register().wires());
    for wire in 0..state.register().wires() {
        subsystems.push(subsystem(wire, state.reduced(&[wire], limits.resources())?).await?);
    }
    Ok(TimelineFrame {
        step,
        label,
        norm: probabilities.iter().sum(),
        probabilities,
        amplitudes: None,
        subsystems,
    })
}
/// Execute each operation once and cache its actual output. Inputs are one feature row.
pub async fn simulate(
    circuit: &Circuit,
    parameters: &[f64],
    inputs: &[f64],
    labels: &[String],
    device: &Device,
    limits: TraceLimits,
) -> Result<ExperimentTrace> {
    let frames = circuit
        .operations()
        .len()
        .checked_add(1)
        .ok_or(Error::Overflow)?;
    limits.check(circuit.register(), frames)?;
    if labels.len() != circuit.operations().len()
        || parameters.len() != circuit.num_parameters()
        || inputs.len() != circuit.num_inputs()
        || parameters.iter().chain(inputs).any(|x| !x.is_finite())
    {
        return Err(Error::Shape("visualization labels or bindings".into()));
    }
    for op in circuit.operations() {
        let entries = match op {
            burn_quantum::Operation::Unitary { gate, .. } => Some(gate.matrix().entries().len()),
            burn_quantum::Operation::Rotate { wire, .. } => Some(
                circuit.register().dimensions()[*wire]
                    .checked_pow(2)
                    .ok_or(Error::Overflow)?,
            ),
            _ => None,
        };
        if let Some(required) = entries
            && required > limits.max_matrix_elements
        {
            return Err(Error::ResourceLimit {
                resource: "visualization gate",
                required,
                limit: limits.max_matrix_elements,
            });
        }
    }
    let parameter_tensor = (!parameters.is_empty()).then(|| {
        Tensor::from_data(
            TensorData::new(parameters.to_vec(), [parameters.len()]),
            (device, DType::F32),
        )
    });
    let input_tensor = (!inputs.is_empty()).then(|| {
        Tensor::from_data(
            TensorData::new(inputs.to_vec(), [1, inputs.len()]),
            (device, DType::F32),
        )
    });
    let engine = DenseSimulator::new(limits.resources());
    let mut state = StateVector::zero(
        circuit.register().clone(),
        1,
        device,
        DType::F32,
        engine.limits,
    )?;
    let mut snapshots = Vec::with_capacity(frames);
    snapshots.push(snapshot_pure(&state, 0, "Input |0…0>".into(), limits).await?);
    for (i, op) in circuit.operations().iter().enumerate() {
        let mut step = Circuit::new(
            circuit.register().clone(),
            circuit.num_parameters(),
            circuit.num_inputs(),
        );
        step.push(op.clone())?;
        state = step.run_from(
            &engine,
            state,
            parameter_tensor.clone(),
            input_tensor.clone(),
        )?;
        snapshots.push(snapshot_pure(&state, i + 1, labels[i].clone(), limits).await?);
    }
    Ok(ExperimentTrace {
        kind: TraceKind::Circuit,
        frames: snapshots,
        backend: format!("{device:?}"),
    })
}
