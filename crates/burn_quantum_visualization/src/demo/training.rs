//! Reconstruct a single probe state per measured optimizer iterate, never an entire dataset.
use super::*;
use crate::simulation::snapshot_pure;
use burn::tensor::{DType, Tensor, TensorData};
use burn_quantum::DenseSimulator;

pub(super) struct Iterate {
    pub parameters: Vec<f64>,
    pub loss: f64,
    pub evaluations: usize,
}
/// Let the browser paint and handle input between CPU training/readback chunks.
/// A resolved CPU tensor future alone does not yield to the browser event loop.
pub(super) async fn yield_to_ui(_: usize) {
    #[cfg(all(target_arch = "wasm32", feature = "viewer"))]
    {
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            if let Some(window) = web_sys::window()
                && window
                    .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0)
                    .is_ok()
            {
                return;
            }
            let _ = resolve.call0(&wasm_bindgen::JsValue::NULL);
        });
        let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    }
}
pub(super) fn preview(circuit: &Circuit, frames: usize, limits: TraceLimits) -> Result<Preview> {
    limits.check(circuit.register(), frames)?;
    // Optimization frames are retained, but the full circuit must also satisfy
    // the document's gate/timeline bound before a costly training run begins.
    let circuit_frames = circuit.operations().len() + 1;
    if circuit_frames > limits.max_frames {
        return Err(burn_quantum::Error::ResourceLimit {
            resource: "visualization circuit frames",
            required: circuit_frames,
            limit: limits.max_frames,
        });
    }
    let r = circuit.register();
    Ok(Preview {
        wires: r.wires(),
        dimension: r.size(),
        steps: frames - 1,
        snapshot_bytes: frames
            * (24 * r.size()
                + r.dimensions()
                    .iter()
                    .map(|d| 16 * d * d + 8 * d)
                    .sum::<usize>()),
    })
}

pub(super) async fn trace(
    circuit: &Circuit,
    inputs: &[f64],
    iterates: Vec<Iterate>,
    device: &Device,
    limits: TraceLimits,
) -> Result<(ExperimentTrace, Vec<OptimizationPoint>)> {
    limits.check(circuit.register(), iterates.len())?;
    let device = device.clone().without_autodiff();
    let engine = DenseSimulator::new(limits.resources());
    let inputs = (!inputs.is_empty()).then(|| {
        Tensor::<2>::from_data(
            TensorData::new(inputs.to_vec(), [1, inputs.len()]),
            (&device, DType::F32),
        )
    });
    let mut frames = Vec::with_capacity(iterates.len());
    let mut points = Vec::with_capacity(iterates.len());
    for (iteration, point) in iterates.into_iter().enumerate() {
        let parameters = Tensor::<1>::from_data(
            TensorData::new(point.parameters.clone(), [point.parameters.len()]),
            (&device, DType::F32),
        );
        let state = circuit.run(&engine, Some(parameters), inputs.clone(), 1, &device)?;
        let frame = snapshot_pure(
            &state,
            iteration,
            format!("Adam update {iteration}"),
            limits,
        )
        .await?;
        points.push(OptimizationPoint {
            iteration,
            parameters: point.parameters,
            loss: point.loss,
            metric: vec![],
            bloch: frame.subsystems.iter().map(|s| s.bloch).collect(),
            probabilities: frame.probabilities.clone(),
            loss_evaluations: point.evaluations,
            fidelity_evaluations: 0,
        });
        frames.push(frame);
        yield_to_ui(iteration).await;
    }
    Ok((
        ExperimentTrace {
            kind: TraceKind::Optimization,
            frames,
            backend: format!("{device:?}"),
        },
        points,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_quantum::qml::benchmarks::wine::{self, WineConfig};

    #[test]
    fn preview_rejects_circuit_budget_before_training() {
        let config = WineConfig {
            qubits: 8,
            depth: 3,
            ..Default::default()
        };
        let circuit = wine::ansatz(&config).unwrap().0;
        let limits = TraceLimits {
            max_frames: 80,
            ..Default::default()
        };
        assert!(limits.check(circuit.register(), config.steps + 1).is_ok());
        assert!(matches!(
            preview(&circuit, config.steps + 1, limits),
            Err(burn_quantum::Error::ResourceLimit {
                resource: "visualization circuit frames",
                ..
            })
        ));
        assert!(preview(&circuit, config.steps + 1, TraceLimits::default()).is_ok());
    }
}
