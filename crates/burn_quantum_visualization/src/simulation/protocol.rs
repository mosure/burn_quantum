//! Measurement trajectories share the core protocol executor with nonvisual clients.
use super::*;
use burn_quantum::{
    algorithms::AlgorithmCircuit,
    protocol::{BranchExecution, Instruction},
};
/// Simulate one explicitly selected measurement branch, recording its actual likelihood.
pub async fn simulate_protocol(
    algorithm: &AlgorithmCircuit,
    outcomes: Vec<usize>,
    device: &Device,
    limits: TraceLimits,
) -> Result<(ExperimentTrace, ProtocolRecord)> {
    let program = algorithm
        .protocol
        .as_ref()
        .ok_or_else(|| Error::InvalidValue("missing protocol".into()))?;
    limits.check(algorithm.circuit.register(), program.instructions.len() + 1)?;
    let engine = DenseSimulator::new(limits.resources());
    let initial = StateVector::zero(
        algorithm.circuit.register().clone(),
        1,
        device,
        DType::F32,
        engine.limits,
    )?;
    let parameters = (!algorithm.parameters.is_empty()).then(|| {
        Tensor::from_data(
            TensorData::new(algorithm.parameters.clone(), [algorithm.parameters.len()]),
            (device, DType::F32),
        )
    });
    let mut run = BranchExecution::new(
        &algorithm.circuit,
        program,
        &engine,
        initial,
        outcomes.clone(),
        parameters,
        None,
    )?;
    let mut frames = vec![snapshot_pure(run.state(), 0, "Input |0…0>".into(), limits).await?];
    let mut probabilities = vec![1.];
    while run.advance().await? {
        let step = run.completed();
        let label = match &program.instructions[step - 1] {
            Instruction::Gate {
                operation,
                conditions,
            } => format!(
                "{}{}",
                algorithm.labels[*operation],
                if conditions.iter().all(|&(b, v)| run.classical()[b] == v) {
                    ""
                } else {
                    " (condition false; skipped)"
                }
            ),
            Instruction::Measure { wire, bit } => format!(
                "Measure q{wire} → c{bit}={} · branch probability {:.4}",
                outcomes[*bit],
                run.probability()
            ),
        };
        frames.push(snapshot_pure(run.state(), step, label, limits).await?);
        probabilities.push(run.probability());
        if step % 4 == 0 {
            yield_to_browser().await;
        }
    }
    Ok((
        ExperimentTrace {
            kind: TraceKind::Protocol,
            frames,
            backend: format!("{device:?}"),
        },
        ProtocolRecord {
            program: program.clone(),
            outcomes,
            probabilities,
        },
    ))
}
