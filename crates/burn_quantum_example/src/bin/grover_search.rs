//! Single-target Grover search; optional qubit count 2..=10.
use burn_quantum::algorithms::{grover, grover_iterations};
use burn_quantum_example::*;
fn run(n: usize) -> ExampleResult<()> {
    let rounds = grover_iterations(n)?;
    let marked = (1 << n) - 3;
    let p = values(algorithm_state(&grover(n, marked, rounds)?)?.probabilities())?;
    let theta = (1. / ((1usize << n) as f64).sqrt()).asin();
    let predicted = ((2 * rounds + 1) as f64 * theta).sin().powi(2);
    ensure(
        (p[marked] - predicted).abs() < 5e-5,
        "amplitude amplification mismatch",
    )?;
    Report::new(
        "grover_search",
        0,
        rounds,
        &[
            ("qubits", n as f64),
            ("success", p[marked]),
            ("analytic_success", predicted),
        ],
    )?
    .print()
}

fn main() -> ExampleResult<()> {
    run(algorithm_size(6, 2, 10)?)
}
#[cfg(test)]
mod tests {
    #[test]
    fn canonical_numerical_acceptance() {
        super::run(6).unwrap();
    }
}
