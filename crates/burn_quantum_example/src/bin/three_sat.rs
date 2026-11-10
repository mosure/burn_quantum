//! Exact-three-literal Boolean configuration constraints with reversible Grover amplification.
//! Usage: three_sat [3..6 variables] [sat|unsat]. This is a bounded reference
//! instance, not a scalable classical SAT solver or evidence of quantum advantage.
use burn_quantum::{DenseSimulator, ResourceLimits, sat::benchmarks::ThreeSatBenchmark};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};

fn run(variables: usize, contradictory: bool) -> ExampleResult<serde_json::Value> {
    let device = execution_device()?;
    let benchmark = ThreeSatBenchmark::new(variables, contradictory, ResourceLimits::default())?;
    let state = benchmark
        .circuit()
        .run(&DenseSimulator::default(), None, None, 1, &device)?;
    state.validate(5e-5)?;
    let readout = benchmark.readout(&values(state.probabilities())?)?;
    ensure(
        (readout.success_probability - readout.analytic_success_probability).abs() < 5e-5,
        "3SAT amplification disagrees with the independent analytic law",
    )?;
    ensure(
        readout.clean_ancilla_probability > 1. - 5e-5,
        "3SAT oracle left dirty work qubits",
    )?;
    ensure(
        readout.reference_proves_unsatisfiable == contradictory,
        "3SAT reference count changed",
    )?;
    ensure(
        contradictory || readout.verified_witness.is_some(),
        "3SAT peak failed classical witness verification",
    )?;
    let execution = Report::new(
        "three_sat",
        0,
        1,
        &[
            ("variables", variables as f64),
            ("clauses", readout.clauses as f64),
            ("simulated_qubits", readout.total_qubits as f64),
            (
                "uniform_success_probability",
                readout.uniform_success_probability,
            ),
            ("success_probability", readout.success_probability),
            (
                "clean_ancilla_probability",
                readout.clean_ancilla_probability,
            ),
        ],
    )?;
    let mut report = serde_json::to_value(execution)?;
    report["formula"] = serde_json::to_value(benchmark.formula())?;
    report["readout"] = serde_json::to_value(readout)?;
    report["interpretation"] = "One actual Grover round. The peak is read from the simulated distribution, not sampled. Satisfying witnesses are classically checked; UNSAT is established only by the exhaustive small-instance reference. Dense storage is exponential in variables plus clause ancillas.".into();
    Ok(report)
}

fn main() -> ExampleResult<()> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!(
            "Usage: three_sat [3..6 variables] [sat|unsat]\nDefault: 5 variables, satisfiable exact-3SAT. BURN_QUANTUM_DEVICE selects CPU/WGPU/CUDA."
        );
        return Ok(());
    }
    ensure(
        arguments.len() <= 2,
        "Usage: three_sat [3..6 variables] [sat|unsat]",
    )?;
    let variables = arguments
        .first()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(5);
    let contradictory = match arguments.get(1).map(String::as_str).unwrap_or("sat") {
        "sat" => false,
        "unsat" => true,
        _ => return Err("variant must be sat or unsat".into()),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&run(variables, contradictory)?)?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn exact_three_sat_amplifies_and_contradictions_require_classical_proof() {
        super::run(5, false).unwrap();
        super::run(4, true).unwrap();
    }
}
