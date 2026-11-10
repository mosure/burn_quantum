//! Odd repetition code with an X on the input wire; optional odd code length 3..=9.
use burn_quantum::{ResourceLimits, algorithms::bit_flip_code};
use burn_quantum_example::*;
fn run(n: usize) -> ExampleResult<()> {
    let state = algorithm_state(&bit_flip_code(n, &[0], 1.1, 0.7)?)?;
    let rho = state.reduced_density(&[0], ResourceLimits::default())?;
    let re = values(rho.entries().real())?;
    let im = values(rho.entries().imag())?;
    let b = [2. * re[1], -2. * im[1], re[0] - re[3]];
    let expected = [
        1.1f64.sin() * 0.7f64.cos(),
        1.1f64.sin() * 0.7f64.sin(),
        1.1f64.cos(),
    ];
    let error = b
        .into_iter()
        .zip(expected)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f64::max);
    ensure(error < 5e-5, "logical recovery failed")?;
    Report::new(
        "bit_flip_code",
        0,
        0,
        &[("code_length", n as f64), ("logical_bloch_error", error)],
    )?
    .print()
}

fn main() -> ExampleResult<()> {
    run(algorithm_size(5, 3, 9)?)
}
#[cfg(test)]
mod tests {
    #[test]
    fn canonical_numerical_acceptance() {
        super::run(5).unwrap();
    }
}
