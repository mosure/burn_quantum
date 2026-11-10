use super::*;
use crate::simulation::{read_tensor, snapshot_pure};
use burn::tensor::{DType, Tensor, TensorData};
use burn_quantum::{
    DenseSimulator, StateVector, optim::qn_spsa_sample, photonic::runtime::shot_seed,
};

pub(super) async fn build(device: &Device, limits: TraceLimits) -> Result<VisualizationDocument> {
    const UPDATES: usize = 24;
    let mut circuit = Circuit::new(Register::qubits(1)?, 1, 0);
    circuit.rotate(0, Rotation::qubit(Axis::Y, Angle::parameter(0)))?;
    limits.check(circuit.register(), UPDATES + 1)?;
    let engine = DenseSimulator::new(limits.resources());
    let prepare = |angle: f64| -> Result<StateVector> {
        circuit.run(
            &engine,
            Some(Tensor::from_data(
                TensorData::new(vec![angle], [1]),
                (device, DType::F32),
            )),
            None,
            1,
            device,
        )
    };
    let mut theta = 0.35;
    let mut metric = 0.25;
    let mut state = prepare(theta)?;
    let mut frame = snapshot_pure(&state, 0, "Initial parameter".into(), limits).await?;
    let point =
        |iteration: usize, theta: f64, metric: f64, frame: &TimelineFrame| OptimizationPoint {
            iteration,
            parameters: vec![theta],
            loss: frame.probabilities[0] - frame.probabilities[1],
            metric: vec![metric],
            bloch: frame.subsystems.iter().map(|s| s.bloch).collect(),
            probabilities: frame.probabilities.clone(),
            loss_evaluations: 1 + 3 * iteration,
            fidelity_evaluations: 4 * iteration,
        };
    let mut points = vec![point(0, theta, metric, &frame)];
    let mut frames = vec![frame];
    for iteration in 0..UPDATES {
        let eps = 0.08;
        let d1 = if shot_seed(731, iteration as u64, 0) & 1 == 0 {
            1.
        } else {
            -1.
        };
        let d2 = if shot_seed(731, iteration as u64, 1) & 1 == 0 {
            1.
        } else {
            -1.
        };
        let plus = prepare(theta + eps * d1)?;
        let minus = prepare(theta - eps * d1)?;
        let plus_prob = read_tensor(plus.probabilities()).await?;
        let minus_prob = read_tensor(minus.probabilities()).await?;
        let both = prepare(theta + eps * (d1 + d2))?;
        let crossed = prepare(theta + eps * (-d1 + d2))?;
        let fidelities = [
            read_tensor(state.fidelity(&plus)?).await?[0],
            read_tensor(state.fidelity(&minus)?).await?[0],
            read_tensor(state.fidelity(&both)?).await?[0],
            read_tensor(state.fidelity(&crossed)?).await?[0],
        ];
        let sample = qn_spsa_sample(
            eps,
            &[d1],
            &[d2],
            [plus_prob[0] - plus_prob[1], minus_prob[0] - minus_prob[1]],
            fidelities,
            1,
        )?;
        metric += (sample.metric[0] - metric) / (iteration + 2) as f64;
        theta -= 0.12 * sample.gradient[0] / (metric.abs() + 0.05);
        state = prepare(theta)?;
        frame = snapshot_pure(
            &state,
            iteration + 1,
            format!("QN-SPSA update {}", iteration + 1),
            limits,
        )
        .await?;
        points.push(point(iteration + 1, theta, metric, &frame));
        frames.push(frame);
    }
    let kind = DemoKind::Optimization;
    let document = VisualizationDocument {
        schema_version: 1,
        title: kind.title().into(),
        description: kind.description().into(),
        circuit,
        parameters: vec![theta],
        inputs: vec![],
        operation_labels: vec!["Ry(theta)".into()],
        trace: ExperimentTrace {
            kind: TraceKind::Optimization,
            frames,
            backend: format!("{device:?}"),
        },
        fabric: None,
        optimization: points,
        protocol: None,
        study: None,
    };
    document.validate_with(limits)?;
    Ok(document)
}
