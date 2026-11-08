//! Reversible-oracle Boolean SAT search, plus an analytic amplitude-amplification check.
//! With an optional DIMACS filename, solve that problem and print its verified outcome.
use burn_quantum::{
    ResourceLimits,
    sat::{Cnf, GroverSat, SatOutcome, SearchConfig},
};
use burn_quantum_example::{ExampleResult, Report, ensure, execution_device, values};

const SEED: u64 = 7;
const ITERATIONS: usize = 2;
// Exactly one model, x1=x2=x3=true. Clauses are compiled without supplying that model.
const DEMO: &str = "p cnf 3 4\n1 2 0\n-1 2 0\n-2 3 0\n1 -3 0\n";

fn run() -> ExampleResult<Report> {
    let device = execution_device()?;
    let limits = ResourceLimits::default();
    let solver = GroverSat::new(Cnf::from_dimacs(DEMO)?, limits)?;
    let initial = solver.initial_state(&device)?;
    let baseline = values(initial.marginal(&[0, 1, 2], limits)?)?[7];
    let state = solver.amplify(ITERATIONS, &device)?;
    state.validate(1e-5)?;
    let probability = values(state.marginal(&[0, 1, 2], limits)?)?[7];
    // Independent analytic result: sin²(5 asin(sqrt(1/8))) = 121/128.
    let expected = 121.0 / 128.0;
    let clean_ancillas = values(state.marginal(&[3, 4, 5, 6], limits)?)?[0];
    ensure(
        (baseline - 0.125).abs() < 2e-5 && (probability - expected).abs() < 2e-5,
        "Grover amplification disagrees with the analytic probability",
    )?;
    ensure(
        clean_ancillas > 1. - 1e-5,
        "SAT oracle left work qubits populated",
    )?;
    let result = solver.search(
        SearchConfig {
            seed: SEED,
            ..Default::default()
        },
        &device,
    )?;
    let SatOutcome::Satisfiable { ref assignment } = result.outcome else {
        return Err("SAT search failed to find the demo witness".into());
    };
    ensure(
        assignment == &[true, true, true] && solver.formula().evaluate(assignment)?,
        "returned assignment does not satisfy the CNF",
    )?;
    Report::new(
        "grover_sat",
        SEED,
        ITERATIONS,
        &[
            ("variables", 3.),
            ("simulated_qubits", solver.num_qubits() as f64),
            ("initial_success_probability", baseline),
            ("success_probability", probability),
            ("analytic_success_probability", expected),
            ("clean_ancilla_probability", clean_ancillas),
            ("verified_witness", 1.),
            ("search_trials", result.trials as f64),
            ("search_oracle_calls", result.oracle_calls as f64),
        ],
    )
}

fn main() -> ExampleResult<()> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.is_empty() {
        return run()?.print();
    }
    if arguments.len() != 1 {
        return Err("Usage: grover_sat [problem.cnf]".into());
    }
    if arguments[0] == "--help" || arguments[0] == "-h" {
        println!(
            "Usage: grover_sat [problem.cnf]\nNo filename runs the verified demo. Select CPU/GPU with BURN_QUANTUM_DEVICE.\nA bounded failed search reports inconclusive, never a general UNSAT proof."
        );
        return Ok(());
    }
    let formula = Cnf::from_dimacs(&std::fs::read_to_string(&arguments[0])?)?;
    let solver = GroverSat::new(formula, ResourceLimits::default())?;
    let result = solver.search(
        SearchConfig {
            seed: SEED,
            ..Default::default()
        },
        &execution_device()?,
    )?;
    let execution = Report::new(
        "grover_sat",
        SEED,
        result.oracle_calls,
        &[
            ("variables", solver.formula().num_variables() as f64),
            (
                "normalized_clauses",
                solver.formula().clauses().len() as f64,
            ),
            ("simulated_qubits", solver.num_qubits() as f64),
        ],
    )?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({"execution":execution,"search":result}))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn amplifies_and_finds_a_verified_sat_witness() {
        super::run().unwrap();
    }
}
