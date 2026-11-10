//! Balanced parity and constant-one promise oracles; optional input width 2..=12.
use burn_quantum::algorithms::deutsch_jozsa;
use burn_quantum_example::*;
fn run(n: usize) -> ExampleResult<()> {
    let mut balanced = 0.;
    let mut constant = 0.;
    for mask in [0, (1 << n) - 1] {
        let state = algorithm_state(&deutsch_jozsa(n, mask, true)?)?;
        let p = values(state.probabilities())?;
        if mask == 0 {
            constant = p[0] + p[1];
        } else {
            balanced = p[0] + p[1];
        }
    }
    ensure(
        (constant - 1.).abs() < 5e-5 && balanced < 5e-5,
        "promise classification failed",
    )?;
    Report::new(
        "deutsch_jozsa",
        0,
        0,
        &[
            ("input_bits", n as f64),
            ("constant_zero_probability", constant),
            ("balanced_zero_probability", balanced),
        ],
    )?
    .print()
}

fn main() -> ExampleResult<()> {
    run(algorithm_size(5, 2, 12)?)
}
#[cfg(test)]
mod tests {
    #[test]
    fn canonical_numerical_acceptance() {
        super::run(5).unwrap();
    }
}
