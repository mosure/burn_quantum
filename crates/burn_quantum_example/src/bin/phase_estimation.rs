//! QPE of eigenphase 3/8; optional precision 3..=12, most significant bit first.
use burn_quantum::algorithms::phase_estimation;
use burn_quantum_example::*;
fn run(n: usize) -> ExampleResult<()> {
    let p = values(algorithm_state(&phase_estimation(n, 3. / 8.)?)?.probabilities())?;
    let target = 3 * (1 << n) / 8;
    let success = p[2 * target] + p[2 * target + 1];
    ensure((success - 1.).abs() < 5e-5, "QPE peak mismatch")?;
    Report::new(
        "phase_estimation",
        0,
        0,
        &[
            ("precision_bits", n as f64),
            ("phase", target as f64 / (1 << n) as f64),
            ("success", success),
        ],
    )?
    .print()
}

fn main() -> ExampleResult<()> {
    run(algorithm_size(5, 3, 12)?)
}
#[cfg(test)]
mod tests {
    #[test]
    fn canonical_numerical_acceptance() {
        super::run(5).unwrap();
    }
}
