//! Compile a symbolic phase circuit into a causal ideal measurement pattern.
//! This tests the logical layer; it does not assume deterministic photonic CZ hardware.
use burn::tensor::{DType, Tensor};
use burn_quantum::{gate::Axis, photonic::compiler::*, *};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};

fn run() -> ExampleResult<Report> {
    let device = execution_device()?;
    let engine = DenseSimulator::default();
    let mut prep = Circuit::new(Register::qubits(2)?, 0, 0);
    prep.unitary(vec![0], Unitary::h())?;
    prep.unitary(vec![0, 1], Unitary::cnot())?;
    let input = prep.run_from(
        &engine,
        StateVector::zero(Register::qubits(2)?, 1, &device, DType::F32, engine.limits)?,
        None,
        None,
    )?;
    let mut logical = Circuit::new(Register::qubits(2)?, 1, 0);
    logical.rotate(0, Rotation::qubit(Axis::Z, Angle::parameter(0)))?;
    logical.unitary(vec![0, 1], Unitary::cnot())?;
    logical.unitary(vec![0], Unitary::h())?;
    let pattern = MeasurementPattern::compile(&logical, PatternCompileLimits::default())?;
    let theta = 0.7f64;
    let branches = PatternSimulator::default().execute_all(
        &pattern,
        input,
        Some(Tensor::from_data([theta as f32], &device)),
        None,
    )?;
    let expected = [
        (theta / 2.).cos().powi(2),
        0.,
        (theta / 2.).sin().powi(2),
        0.,
    ];
    let mut error = 0.0f64;
    let mut total = 0.;
    for branch in &branches {
        total += values(branch.probability())?[0];
        let probabilities = values(branch.conditional_state(1e-8)?.probabilities())?;
        for (actual, expected) in probabilities.iter().zip(expected) {
            error = error.max((actual - expected).abs());
        }
    }
    ensure(
        error < 3e-5 && (total - 1.).abs() < 3e-5,
        "causal measurement pattern disagrees with analytic phase circuit",
    )?;
    ensure(
        pattern.measurements() == 5 && branches.len() == 32,
        "unexpected compiler expansion",
    )?;
    Report::new(
        "photonic_teleportation",
        0,
        1,
        &[
            ("ideal_pattern_sites", pattern.sites() as f64),
            ("measurements", pattern.measurements() as f64),
            ("branches", branches.len() as f64),
            ("total_probability", total),
            ("max_conditional_probability_error", error),
            ("expected_zero_probability", expected[0]),
        ],
    )
}
fn main() -> ExampleResult<()> {
    run()?.print()
}
#[cfg(test)]
mod tests {
    #[test]
    fn corrected_measurement_branches_match_analytic_phase_readout() {
        super::run().unwrap();
    }
}
