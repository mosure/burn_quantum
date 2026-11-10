//! Costed 15-to-1 magic distillation, followed by native T injection on a Bell pair.
use burn_quantum::{photonic::qec::*, *};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};
use serde::Serialize;
#[derive(Serialize)]
struct Run {
    #[serde(flatten)]
    report: Report,
    distillation: MagicDistillation,
}
fn run() -> ExampleResult<Run> {
    let device = execution_device()?;
    let engine = DenseSimulator::default();
    let distillation = ReedMuller15::distill(&MagicDistillationConfig {
        phase_errors: [0.01; 15],
        source_availability: [0.995; 15],
        syndrome_readout_errors: [0.0001; 4],
        ..Default::default()
    })?;
    // Independently execute the actual code encoder/transversal gate/decoder.
    let decoded = engine.execute(
        &ReedMuller15::fault_circuit(0, true)?,
        None,
        None,
        1,
        &device,
    )?;
    decoded.validate(3e-6)?;
    let logical = decoded.reduced_density(&[0], engine.limits)?;
    let ideal_magic =
        MagicState::new(0., 1.)?.density(&device, logical.entries().dtype(), engine.limits)?;
    let code_overlap = (logical.entries().real() * ideal_magic.entries().real()
        + logical.entries().imag() * ideal_magic.entries().imag())
    .sum()
    .into_scalar::<f64>();
    ensure(
        (code_overlap - 1.).abs() < 3e-6,
        "native code implementation failed",
    )?;
    let mut bell = Circuit::new(Register::qubits(2)?, 0, 0);
    bell.unitary(vec![0], Unitary::h())?
        .unitary(vec![0, 1], Unitary::cnot())?;
    let input = DensityMatrix::from_pure(
        &engine.execute(&bell, None, None, 1, &device)?,
        engine.limits,
    )?;
    let injected = distillation.resource()?.inject(&input, 0, engine.limits)?;
    let completed = injected.completed(engine.limits)?;
    let ideal = input.apply(&engine, &[0], &Unitary::t())?;
    let success = values(completed.trace())?[0];
    let fidelity = (completed.entries().real() * ideal.entries().real()
        + completed.entries().imag() * ideal.entries().imag())
    .sum()
    .into_scalar::<f64>()
        / success;
    let error = distillation.output_phase_error.unwrap();
    ensure(
        (success - distillation.acceptance).abs() < 3e-6,
        "source yield lost",
    )?;
    ensure(
        (fidelity - (1. - error)).abs() < 3e-6 && error < 0.0001,
        "distillation failed to improve the consumed resource",
    )?;
    let report = Report::new(
        "photonic_magic_distillation",
        0,
        0,
        &[
            ("raw_magic_error", 0.01),
            ("distilled_magic_error", error),
            ("acceptance", success),
            ("conditional_entangled_gate_fidelity", fidelity),
            ("native_encoded_state_fidelity", code_overlap),
            (
                "false_herald_probability",
                distillation.false_herald_probability,
            ),
            (
                "raw_source_attempts_per_output",
                distillation.cost.raw_attempts_per_output.unwrap(),
            ),
            (
                "expected_measurements_per_attempt",
                distillation.cost.expected_measurements,
            ),
        ],
    )?;
    Ok(Run {
        report,
        distillation,
    })
}
fn main() -> ExampleResult<()> {
    println!("{}", serde_json::to_string_pretty(&run()?)?);
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn distilled_resources_improve_entangled_gate_with_complete_costs() {
        super::run().unwrap();
    }
}
