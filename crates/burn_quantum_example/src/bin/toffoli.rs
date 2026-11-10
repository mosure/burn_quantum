//! Toffoli/generalized controlled-X on superposed controls; optional controls 2..=12.
use burn_quantum::algorithms::controlled_x_demo;
use burn_quantum_example::*;
fn run(n: usize) -> ExampleResult<()> {
    let p = values(algorithm_state(&controlled_x_demo(n, true)?)?.probabilities())?;
    let target_one: f64 = p.iter().skip(1).step_by(2).sum();
    ensure(
        (target_one - 1. / (1usize << n) as f64).abs() < 5e-5,
        "controlled-X correlation mismatch",
    )?;
    Report::new(
        "toffoli",
        0,
        0,
        &[
            ("controls", n as f64),
            ("target_one_probability", target_one),
        ],
    )?
    .print()
}

fn main() -> ExampleResult<()> {
    run(algorithm_size(2, 2, 12)?)
}
#[cfg(test)]
mod tests {
    #[test]
    fn canonical_numerical_acceptance() {
        super::run(2).unwrap();
    }
}
